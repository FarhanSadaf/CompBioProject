import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed, parallel_config
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


METHODS = ["random_pairing", "background", "p99", "p75", "p50", "p25", "p1"]


def rank_background(positives, repertoire):
    ranks = {}
    candidates = np.array(sorted(repertoire.tcr_beta.unique()))
    for (hla, peptide), group in positives.groupby(["hla", "peptide"], sort=True):
        binders = group.tcr_beta.unique().tolist()
        distances = process.cdist(candidates.tolist(), binders, scorer=Levenshtein.distance, dtype=np.int16)
        ranks[(hla, peptide)] = pd.DataFrame({
            "tcr_beta": candidates, "distance": distances.min(axis=1),
        })
    return ranks


def make_training_set(positives, repertoire, method, seed, ratio=5, ranks=None, excluded_pairs=None):
    rng = np.random.default_rng(seed)
    negatives = []
    for (hla, peptide), group in positives.groupby(["hla", "peptide"], sort=True):
        known = positives.loc[positives.peptide.eq(peptide), "tcr_beta"]
        if method == "random_pairing":
            pool = positives.loc[positives.hla.eq(hla), ["tcr_beta"]].drop_duplicates()
        elif method == "background":
            pool = repertoire[["tcr_beta"]].copy()
        elif method.startswith("p") and ranks is not None:
            percentile = int(method[1:])
            if percentile not in [1, 25, 50, 75, 99]:
                raise ValueError(f"Unknown hardness percentile: {method}")
            pool = ranks[(hla, peptide)]
            cutoff = pool.distance.quantile(percentile / 100, interpolation="higher")
            pool = pool.loc[pool.distance.le(cutoff)].copy()
            pool["distance_cutoff"] = cutoff
        else:
            raise ValueError(f"Unknown negative method or missing distance rankings: {method}")
        pool = pool.loc[~pool.tcr_beta.isin(known)].reset_index(drop=True)
        if excluded_pairs is not None:
            blocked = excluded_pairs.loc[
                excluded_pairs.hla.eq(hla) & excluded_pairs.peptide.eq(peptide), "tcr_beta"
            ]
            pool = pool.loc[~pool.tcr_beta.isin(blocked)].reset_index(drop=True)
        if pool.empty:
            raise ValueError(f"No negative candidates for {hla}, {peptide}, {method}.")
        pool = pool.copy()
        pool["pool_size"] = len(pool)
        # Use every candidate once before repeating pairs to keep the 5:1 row ratio.
        order = np.resize(rng.permutation(len(pool)), ratio * len(group))
        sampled = pool.iloc[order].copy()
        sampled["peptide"], sampled["hla"], sampled["label"] = peptide, hla, 0
        negatives.append(sampled)
    columns = ["tcr_beta", "peptide", "hla", "label"]
    train = pd.concat([positives.assign(label=1)[columns], *negatives], ignore_index=True)
    train["method"] = method
    train.insert(0, "row_id", np.arange(len(train)))
    return train


def embed_sequences(sequences, batch_size=32, device="cpu", cache=None):
    sequences = sorted(set(sequences))
    if cache is not None and Path(cache).exists():
        with np.load(cache, allow_pickle=False) as stored:
            embeddings = {sequence: stored[sequence] for sequence in stored.files}
        if set(embeddings) != set(sequences):
            raise ValueError("The embedding cache does not match the data. Remove it and rerun.")
        return embeddings

    import torch
    from transformers import AutoModel, AutoTokenizer

    name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(name)
    encoder = AutoModel.from_pretrained(name, use_safetensors=True).to(device).eval()
    encoder.requires_grad_(False)
    embeddings = {}
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        tokens = tokenizer(batch, padding=True, return_tensors="pt", return_special_tokens_mask=True)
        special = tokens.pop("special_tokens_mask").to(device).bool()
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].bool() & ~special
            vectors = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        embeddings.update(zip(batch, vectors.cpu().numpy()))
    if cache is not None:
        np.savez_compressed(cache, **embeddings)
    return embeddings


def pair_features(table, embeddings, inputs=("tcr_beta", "peptide", "hla")):
    blocks = []
    for column in ["tcr_beta", "peptide"]:
        block = np.stack([embeddings[sequence] for sequence in table[column]])
        blocks.append(block if column in inputs else np.zeros_like(block))
    hla = np.column_stack([table.hla.eq("HLA-A*02:01"), table.hla.eq("HLA-B*40:01")])
    blocks.append(hla if "hla" in inputs else np.zeros_like(hla))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def mlp_scores(train, test, embeddings, seed, inputs=("tcr_beta", "peptide", "hla"), epochs=10):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(pair_features(train, embeddings, inputs))
    x_test = scaler.transform(pair_features(test, embeddings, inputs))
    model = MLPClassifier(
        hidden_layer_sizes=(128,), activation="relu", solver="adam", alpha=0.0001,
        batch_size=256, learning_rate_init=0.001, early_stopping=False, random_state=seed,
    )
    for _ in range(epochs):
        model.partial_fit(x_train, train.label.to_numpy(), classes=np.array([0, 1]))
    return model.predict_proba(x_test)[:, 1]


def boosting_scores(train, test, seed):
    hasher = HashingVectorizer(
        analyzer="char", ngram_range=(3, 3), n_features=256,
        lowercase=False, alternate_sign=False, norm=None,
    )
    features = []
    for table in [train, test]:
        tcr = hasher.transform(table.tcr_beta).toarray()
        peptide = hasher.transform(table.peptide).toarray()
        hla = np.column_stack([table.hla.eq("HLA-A*02:01"), table.hla.eq("HLA-B*40:01")])
        features.append(np.concatenate([tcr, peptide, hla], axis=1))
    model = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, learning_rate=0.1,
        early_stopping=False, random_state=seed,
    )
    model.fit(features[0], train.label)
    return model.predict_proba(features[1])[:, 1]


def structure_candidates(hard_negatives, count=200):
    candidates = hard_negatives.drop_duplicates(["hla", "peptide", "tcr_beta"])
    candidates = candidates.sort_values(["distance", "hla", "peptide", "tcr_beta"]).head(count)
    candidates = candidates[["tcr_beta", "peptide", "hla", "distance"]].reset_index(drop=True)
    candidates.insert(0, "row_id", np.arange(len(candidates)))
    return candidates


def run_experiments(positives, repertoire, test_sets, embeddings, seeds,
                    ratio=5, epochs=10, permutations=1000, bootstraps=1000, output="results",
                    methods=None, excluded_pairs=None, n_jobs=1):
    from evaluation.evaluate import record_fit, save_results

    output = Path(output)
    methods = METHODS if methods is None else methods
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError("Choose one or more of the seven negative constructions.")
    def run_fold(fold):
        fits, peptides, predictions = [], [], []
        hardness_checks, hardest = [], []
        train_positive = positives.loc[positives.fold.ne(fold)]
        test = test_sets[fold]
        assert not set(train_positive.peptide) & set(test.peptide)
        ranks = rank_background(train_positive, repertoire)
        for seed in seeds:
            run_seed = seed + 100003 * fold
            for method in methods:
                train = make_training_set(
                    train_positive, repertoire, method, run_seed, ratio, ranks, excluded_pairs,
                )
                scores = mlp_scores(train, test, embeddings, run_seed, epochs=epochs)
                details = dict(experiment="negative_swap", method=method, control="full",
                               model="ESM2_MLP", fold=fold, seed=seed)
                result, per_peptide, predicted = record_fit(
                    train, test, scores, details, permutations, bootstraps,
                )
                fits.append(result)
                peptides.append(per_peptide)
                predictions.append(predicted)
                if method.startswith("p"):
                    negative = train.loc[train.label.eq(0)]
                    checks = negative.groupby(["hla", "peptide"]).agg(
                        negative_rows=("tcr_beta", "size"), unique_pairs=("tcr_beta", "nunique"),
                        eligible_pool=("pool_size", "first"), cutoff=("distance_cutoff", "first"),
                        minimum_distance=("distance", "min"), maximum_distance=("distance", "max"),
                    ).reset_index()
                    hardness_checks.append(checks.assign(fold=fold, seed=seed, method=method))
                    scores = boosting_scores(train, test, run_seed)
                    details = {**details, "experiment": "robustness", "model": "3mer_boosting"}
                    result, per_peptide, predicted = record_fit(
                        train, test, scores, details, permutations, bootstraps,
                    )
                    fits.append(result)
                    peptides.append(per_peptide)
                    predictions.append(predicted)
                    if method == "p1" and seed == seeds[0]:
                        hardest.append(negative[["tcr_beta", "peptide", "hla", "distance"]].copy())
            print(f"Negatives: fold {fold}, seed {seed} finished")
        return fits, peptides, predictions, hardness_checks, hardest

    fits, peptides, predictions = [], [], []
    hardness_checks, hardest = [], []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(run_fold)(fold) for fold in sorted(test_sets)
        )
        for fold_fits, fold_peptides, fold_predictions, fold_checks, fold_hardest in completed:
            fits.extend(fold_fits)
            peptides.extend(fold_peptides)
            predictions.extend(fold_predictions)
            hardness_checks.extend(fold_checks)
            hardest.extend(fold_hardest)
            summary = save_results(fits, peptides, predictions, output, "negatives")
            if hardness_checks:
                pd.concat(hardness_checks, ignore_index=True).to_csv(output / "hardness_checks.csv", index=False)
    if hardest:
        candidates = structure_candidates(pd.concat(hardest, ignore_index=True), count=None)
        candidates.to_csv(output / "structure_pool.csv", index=False)
    return summary


def plot_hardness(folder="results"):
    import matplotlib.pyplot as plt

    folder = Path(folder)
    table = pd.read_csv(folder / "negatives_summary.csv")
    main = table.loc[table.experiment.eq("negative_swap")].set_index("method")
    order = ["p99", "p75", "p50", "p25", "p1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, ["auroc", "auprc"]):
        ax.errorbar(range(5), main.loc[order, metric + "_mean"],
                    yerr=main.loc[order, metric + "_std"], marker="o", capsize=3)
        ax.set_xticks(range(5), order)
        ax.set_xlabel("Distance cutoff: easy to hard")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(folder / "hardness.png", dpi=180)
    plt.show()


def plot_negative_swap(folder="results"):
    import matplotlib.pyplot as plt

    folder = Path(folder)
    table = pd.read_csv(folder / "negatives_summary.csv")
    main = table.loc[table.experiment.eq("negative_swap")].set_index("method")
    comparison = main.loc[["random_pairing", "background", "p1"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, ["auroc", "auprc"]):
        ax.bar(range(3), comparison[metric + "_mean"], yerr=comparison[metric + "_std"], capsize=3)
        ax.set_xticks(range(3), ["Random pairing", "Background", "Hard p1"], rotation=15)
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(folder / "negative_swap.png", dpi=180)
    plt.show()
    print("AUROC range across the three constructions:",
          round(comparison.auroc_mean.max() - comparison.auroc_mean.min(), 4))
    return comparison[["auroc_mean", "auroc_std", "auprc_mean", "auprc_std"]]


def plot_robustness(folder="results"):
    import matplotlib.pyplot as plt

    folder = Path(folder)
    table = pd.read_csv(folder / "negatives_summary.csv")
    main = table.loc[table.experiment.eq("negative_swap")].set_index("method")
    robustness = table.loc[table.experiment.eq("robustness")].set_index("method")
    order = ["p99", "p75", "p50", "p25", "p1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, ["auroc", "auprc"]):
        for values, label, marker in [(main, "ESM-2 + MLP", "o"), (robustness, "3-mer + boosting", "s")]:
            ax.errorbar(range(5), values.loc[order, metric + "_mean"],
                        yerr=values.loc[order, metric + "_std"], marker=marker, capsize=3, label=label)
        ax.set_xticks(range(5), order)
        ax.set_xlabel("Distance cutoff: easy to hard")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "robustness.png", dpi=180)
    plt.show()
