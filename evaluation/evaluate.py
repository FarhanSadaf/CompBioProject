import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import average_precision_score, roc_auc_score


CONTROLS = {
    "full": ("tcr_beta", "peptide", "hla"),
    "peptide_blind": ("tcr_beta", "hla"),
    "mhc_blind": ("tcr_beta", "peptide"),
    "tcr_blind": ("peptide", "hla"),
    "tcr_only": ("tcr_beta",),
    "peptide_only": ("peptide",),
    "mhc_only": ("hla",),
}


def make_test_set(positives, repertoire, seed, ratio=5):
    rng = np.random.default_rng(seed)
    negatives = []
    background = repertoire.tcr_beta.to_numpy()
    lengths = np.array([len(sequence) for sequence in background])
    for (hla, peptide), group in positives.groupby(["hla", "peptide"], sort=True):
        used = set()
        for length, rows in group.groupby(group.tcr_beta.str.len(), sort=True):
            difference = np.abs(lengths - length)
            candidates = background[difference == difference.min()]
            # Exclude known binders even if a different background is supplied later.
            candidates = np.setdiff1d(candidates, positives.loc[positives.peptide.eq(peptide), "tcr_beta"])
            if not len(candidates):
                raise ValueError(f"No test candidates for {peptide}, CDR3 length {length}.")
            unused = np.array([tcr for tcr in candidates if tcr not in used])
            first = rng.permutation(unused)
            needed = ratio * len(rows)
            if needed > len(first):
                extra = np.resize(rng.permutation(candidates), needed - len(first))
                selected = np.concatenate([first, extra])
            else:
                selected = first[:needed]
            used.update(selected)
            negatives.append(pd.DataFrame({"tcr_beta": selected, "peptide": peptide, "hla": hla, "label": 0}))
    columns = ["tcr_beta", "peptide", "hla", "label"]
    test = pd.concat([positives.assign(label=1)[columns], *negatives], ignore_index=True)
    test.insert(0, "row_id", np.arange(len(test)))
    return test


def metrics(labels, scores):
    if len(np.unique(labels)) < 2:
        return np.array([np.nan, np.nan])
    return np.array([roc_auc_score(labels, scores), average_precision_score(labels, scores)])


def evaluate(test, scores, seed, permutations=1000, bootstraps=1000):
    y = test.label.to_numpy()
    scores = np.asarray(scores, dtype=float)
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("There must be one finite score for every test row.")
    if permutations < 1 or bootstraps < 1:
        raise ValueError("Use at least one shuffle and one bootstrap sample.")
    observed = metrics(y, scores)
    rng = np.random.default_rng(seed)
    null = np.array([metrics(rng.permutation(y), scores) for _ in range(permutations)])

    peptide_groups = list(test.reset_index(drop=True).groupby("peptide", sort=True).indices.items())
    peptide_results = []
    for peptide, index in peptide_groups:
        roc, ap = metrics(y[index], scores[index])
        peptide_results.append({
            "peptide": peptide, "positives": int(y[index].sum()),
            "negatives": int((y[index] == 0).sum()), "auroc": roc, "auprc": ap,
        })

    intervals = np.full((2, 2), np.nan)
    if len(peptide_groups) > 1:
        samples = []
        for _ in range(bootstraps):
            selected = rng.integers(0, len(peptide_groups), size=len(peptide_groups))
            index = np.concatenate([peptide_groups[i][1] for i in selected])
            samples.append(metrics(y[index], scores[index]))
        intervals = np.quantile(samples, [0.025, 0.975], axis=0)

    negative = test.loc[test.label.eq(0)]
    unique = len(negative.drop_duplicates(["hla", "peptide", "tcr_beta"]))
    result = {
        "auroc": observed[0], "auprc": observed[1],
        "chance_auroc": np.quantile(null[:, 0], 0.95),
        "chance_auprc": np.quantile(null[:, 1], 0.95),
        "auroc_low": intervals[0, 0], "auroc_high": intervals[1, 0],
        "auprc_low": intervals[0, 1], "auprc_high": intervals[1, 1],
        "test_peptides": len(peptide_groups), "negative_rows": len(negative),
        "unique_negatives": unique, "reused_negatives": len(negative) - unique,
    }
    return result, pd.DataFrame(peptide_results)


def summarize(results):
    keys = ["experiment", "method", "control", "model"]
    # Average the same five folds first, then measure spread across seeds.
    by_seed = results.groupby(keys + ["seed"])[["auroc", "auprc"]].mean().reset_index()
    summary = by_seed.groupby(keys)[["auroc", "auprc"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    by_fold = results.groupby(keys + ["fold"])[["auroc", "auprc"]].agg(["mean", "std"])
    by_fold.columns = ["_".join(column) for column in by_fold.columns]
    return summary.reset_index(), by_seed, by_fold.reset_index()


def record_fit(train, test, scores, details, permutations=1000, bootstraps=1000):
    seed = details["seed"] + 100003 * details["fold"]
    result, peptides = evaluate(test, scores, seed + 2, permutations, bootstraps)
    negatives = train.loc[train.label.eq(0)]
    unique = len(negatives.drop_duplicates(["hla", "peptide", "tcr_beta"]))
    result.update(
        train_negative_rows=len(negatives), train_unique_negatives=unique,
        train_reused_negatives=len(negatives) - unique,
    )
    predictions = pd.DataFrame({"row_id": test.row_id.to_numpy(), "score": scores})
    return {**details, **result}, peptides.assign(**details), predictions.assign(**details)


def save_results(fits, peptides, predictions, folder, prefix):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(fits)
    summary, _, _ = summarize(results)
    keys = ["experiment", "method", "control", "model"]
    counts = results.groupby(keys).agg(folds=("fold", "nunique"), seeds=("seed", "nunique")).reset_index()
    summary = summary.merge(counts, on=keys, validate="one_to_one")
    results.to_csv(folder / f"{prefix}_metrics.csv", index=False)
    summary.to_csv(folder / f"{prefix}_summary.csv", index=False)
    pd.concat(peptides, ignore_index=True).to_csv(folder / f"{prefix}_per_peptide.csv.gz", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(folder / f"{prefix}_predictions.csv.gz", index=False)
    return summary


def run_controls(positives, repertoire, test_sets, embeddings, seeds,
                 ratio=5, epochs=10, permutations=1000, bootstraps=1000, output="results", n_jobs=1):
    from hard_negatives.experiment import METHODS, make_training_set, mlp_scores, rank_background

    def run_fold(fold):
        fits, peptides, predictions = [], [], []
        train_positive = positives.loc[positives.fold.ne(fold)]
        test = test_sets[fold]
        assert not set(train_positive.peptide) & set(test.peptide)
        ranks = rank_background(train_positive, repertoire)
        for seed in seeds:
            run_seed = seed + 100003 * fold
            for method in METHODS:
                train = make_training_set(train_positive, repertoire, method, run_seed, ratio, ranks)
                for control, inputs in CONTROLS.items():
                    scores = mlp_scores(train, test, embeddings, run_seed, inputs=inputs, epochs=epochs)
                    details = dict(experiment="shortcuts", method=method, control=control,
                                   model="ESM2_MLP", fold=fold, seed=seed)
                    result, per_peptide, predicted = record_fit(
                        train, test, scores, details, permutations, bootstraps,
                    )
                    fits.append(result)
                    peptides.append(per_peptide)
                    predictions.append(predicted)
            print(f"Controls: fold {fold}, seed {seed} finished")
        return fits, peptides, predictions

    fits, peptides, predictions = [], [], []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(run_fold)(fold) for fold in sorted(test_sets)
        )
        for fold_fits, fold_peptides, fold_predictions in completed:
            fits.extend(fold_fits)
            peptides.extend(fold_peptides)
            predictions.extend(fold_predictions)
            summary = save_results(fits, peptides, predictions, output, "controls")
    return summary


def plot_controls(folder="results", method="p1"):
    import matplotlib.pyplot as plt

    folder = Path(folder)
    summary = pd.read_csv(folder / "controls_summary.csv")
    summary = summary.loc[summary.method.eq(method)].set_index("control")
    if summary.empty:
        raise ValueError(f"No control results for {method}.")
    metrics_table = pd.read_csv(folder / "controls_metrics.csv")
    metrics_table = metrics_table.loc[metrics_table.method.eq(method)]
    chance = metrics_table.groupby("control")[["chance_auroc", "chance_auprc"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["auroc", "auprc"]):
        ax.errorbar(range(len(summary)), summary[metric + "_mean"],
                    yerr=summary[metric + "_std"], fmt="o", capsize=3)
        ax.plot(range(len(summary)), chance.loc[summary.index, "chance_" + metric],
                "x", label="Mean shuffled threshold")
        ax.set_xticks(range(len(summary)), summary.index, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.suptitle(method)
    fig.subplots_adjust(top=0.88)
    fig.savefig(folder / f"shortcuts_{method}.png", dpi=180)
    plt.show()
    return summary[["auroc_mean", "auroc_std", "auprc_mean", "auprc_std"]].join(chance)
