from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate import metrics


def prepare_common_benchmark(folder="results", receptors=None):
    folder = Path(folder)
    pairs = pd.read_csv(folder / "test_pairs.csv.gz")
    if pairs.groupby("peptide").fold.nunique().gt(1).any():
        raise ValueError("A peptide crosses the saved fold boundaries.")
    # These are epiTCR's released input limits; the main study keeps its original cohort.
    if receptors is None:
        from preprocessing.sequences import prepare_receptors
        receptors = prepare_receptors(output=folder / "receptor_sequences.csv")
    if receptors.tcr_beta.duplicated().any():
        raise ValueError("Use one unambiguous beta sequence per CDR3.")
    length_ok = pairs.tcr_beta.str.len().between(8, 19) & pairs.peptide.str.len().between(8, 11)
    beta_ok = pairs.tcr_beta.isin(receptors.tcr_beta)
    supported = length_ok & beta_ok
    common = pairs.loc[supported].copy()
    common = common.merge(receptors[["tcr_beta", "tcr_beta_sequence"]], on="tcr_beta", validate="many_to_one")
    if set(common.fold) != set(range(5)) or not common.groupby("fold").label.nunique().eq(2).all():
        raise ValueError("The common benchmark needs both labels in every saved fold.")
    coverage = pairs.groupby(["fold", "peptide"]).size().rename("original_pairs").to_frame()
    retained = common.groupby(["fold", "peptide"]).agg(
        shared_pairs=("row_id", "size"), shared_positives=("label", "sum"),
    )
    coverage = coverage.join(retained).fillna(0)
    exclusions = pairs.assign(unsupported_length=~length_ok, missing_beta_sequence=length_ok & ~beta_ok)
    coverage = coverage.join(exclusions.groupby(["fold", "peptide"])[["unsupported_length", "missing_beta_sequence"]].sum())
    coverage.to_csv(folder / "common_benchmark_coverage.csv")
    common.to_csv(folder / "common_test_pairs.csv.gz", index=False)
    common[["fold", "row_id", "tcr_beta", "peptide", "label"]].rename(columns={
        "tcr_beta": "CDR3b", "peptide": "epitope", "label": "binder",
    }).to_csv(folder / "epitcr_common_input.csv", index=False)
    return common


def compare_predictors(scores=None, folder="results"):
    folder = Path(folder)
    pairs = pd.read_csv(folder / "common_test_pairs.csv.gz")
    scores = folder / "common_predictor_scores.csv" if scores is None else Path(scores)
    key = ["fold", "row_id"]
    if pairs.duplicated(key).any() or set(pairs.fold) != set(range(5)):
        raise ValueError("Use uniquely identified pairs from the five saved folds.")
    if not Path(scores).exists():
        raise FileNotFoundError("Run predict_common_pairs() before comparing the released predictors.")
    predicted = pd.read_csv(scores)
    if not {"predictor", "fold", "row_id", "score"}.issubset(predicted.columns):
        raise ValueError("Predictor scores need predictor, fold, row_id, score columns.")
    if set(predicted.predictor) != {"epiTCR", "TITAN"}:
        raise ValueError("Supply both released predictors' scores on the shared pairs.")
    if predicted.duplicated(["predictor", *key]).any() or not np.isfinite(predicted.score).all():
        raise ValueError("Each predictor needs one finite score per shared pair.")
    expected = set(map(tuple, pairs[key].to_numpy()))
    for name, group in predicted.groupby("predictor"):
        if set(map(tuple, group[key].to_numpy())) != expected:
            raise ValueError(f"{name} must score exactly the common benchmark pairs.")
    evaluated = predicted.merge(pairs, on=key, validate="many_to_one")
    rows = []
    for (name, fold), group in evaluated.groupby(["predictor", "fold"]):
        roc, ap = metrics(group.label, group.score)
        rows.append({"predictor": name, "fold": fold, "auroc": roc, "auprc": ap})
    external = pd.DataFrame(rows)
    external.to_csv(folder / "common_predictor_metrics.csv", index=False)
    peptide_rows = []
    for (name, fold, peptide), group in evaluated.groupby(["predictor", "fold", "peptide"]):
        roc, ap = metrics(group.label, group.score)
        peptide_rows.append({"predictor": name, "fold": fold, "peptide": peptide, "auroc": roc, "auprc": ap})
    coverage = pd.read_csv(folder / "common_benchmark_coverage.csv")
    complete = coverage.merge(pd.DataFrame({"predictor": ["epiTCR", "TITAN"]}), how="cross")
    complete.merge(pd.DataFrame(peptide_rows), on=["predictor", "fold", "peptide"],
                   how="left", validate="one_to_one").to_csv(folder / "common_predictor_per_peptide.csv", index=False)
    external_mean = external.groupby("predictor")[["auroc", "auprc"]].mean()

    local = pd.read_csv(folder / "negatives_predictions.csv.gz")
    local = local.loc[local.experiment.eq("negative_swap") & local.control.eq("full") & local.model.eq("ESM2_MLP")
                      & local.method.isin(["random_pairing", "background", "p1"])]
    if set(local.method) != {"random_pairing", "background", "p1"} or len(local.seed.unique()) < 5:
        raise ValueError("Finish all three local constructions with at least five seeds.")
    seed_sets = [set(group.seed) for _, group in local.groupby("method")]
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("Use the same seeds for all three negative constructions.")
    if local.duplicated(["method", "seed", *key]).any() or not np.isfinite(local.score).all():
        raise ValueError("The saved local predictions contain duplicate pairs or invalid scores.")
    local = local.merge(pairs, on=key, validate="many_to_one")
    local_rows = []
    for (method, seed), group in local.groupby(["method", "seed"]):
        if set(map(tuple, group[key].to_numpy())) != expected:
            raise ValueError("Every local fit must cover the same common benchmark pairs.")
        for fold, heldout in group.groupby("fold"):
            roc, ap = metrics(heldout.label, heldout.score)
            local_rows.append({"method": method, "seed": seed, "fold": fold, "auroc": roc, "auprc": ap})
    local_metrics = pd.DataFrame(local_rows)
    local_metrics.to_csv(folder / "common_negative_metrics.csv", index=False)
    seed_means = local_metrics.groupby(["method", "seed"])[["auroc", "auprc"]].mean()
    local_mean = seed_means.groupby("method").mean()
    local_sd = seed_means.groupby("method").std()
    result = pd.DataFrame({
        "metric": ["auroc", "auprc"],
        "negative_swap_range": [local_mean[m].max() - local_mean[m].min() for m in ["auroc", "auprc"]],
        "predictor_gap": [abs(external_mean.loc["epiTCR", m] - external_mean.loc["TITAN", m]) for m in ["auroc", "auprc"]],
        "largest_seed_sd": [local_sd[m].max() for m in ["auroc", "auprc"]],
    })
    result["swap_exceeds_gap"] = result.negative_swap_range > result.predictor_gap
    result["swap_exceeds_seed_sd"] = result.negative_swap_range > result.largest_seed_sd
    result.to_csv(folder / "rq5_comparison.csv", index=False)
    return external_mean, result


def compare_after_audit(folder="results"):
    folder = Path(folder)
    before = pd.read_csv(folder / "negatives_metrics.csv")
    after = pd.read_csv(folder / "after_audit/negatives_metrics.csv")
    before = before.loc[before.experiment.eq("negative_swap") & before.method.eq("p1")]
    after = after.loc[after.experiment.eq("negative_swap") & after.method.eq("p1")]
    if before.empty or after.empty:
        raise ValueError("Both the original and audit-filtered p1 results are required.")
    keys = ["fold", "seed", "method", "control", "model"]
    paired = before.merge(after, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    if len(paired) != len(before) or len(paired) != len(after):
        raise ValueError("The audit rerun must use exactly the same folds and seeds.")
    for metric in ["auroc", "auprc"]:
        paired[metric + "_change"] = paired[metric + "_after"] - paired[metric + "_before"]
    paired.to_csv(folder / "structure_effect.csv", index=False)
    return paired.groupby("seed")[["auroc_change", "auprc_change"]].mean().agg(["mean", "std"])
