import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--per-peptide", action="store_true")
    args = ap.parse_args()

    arr = np.load(args.pred)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise SystemExit(f"unexpected shape {arr.shape}, expected (2, n)")

    predictions = arr[0].ravel()
    labels = arr[1].ravel().astype(int)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    print(f"pairs  {len(labels):,}")
    print(f"labels {n_pos:,} pos / {n_neg:,} neg")

    if n_pos == 0 or n_neg == 0:
        print("single class, auroc/auprc undefined")
        sys.exit(0)

    print(f"auroc  {roc_auc_score(labels, predictions):.4f}")
    print(f"auprc  {average_precision_score(labels, predictions):.4f} "
          f"(baseline {n_pos / len(labels):.4f})")
    print(f"acc    {((predictions >= 0.5).astype(int) == labels).mean():.4f}")

    if not args.per_peptide:
        return

    if args.test is None:
        raise SystemExit("--per-peptide needs --test")

    test = pd.read_csv(args.test)
    if len(test) != len(labels):
        raise SystemExit(
            f"row mismatch: {len(test)} test rows vs {len(labels)} predictions. "
            "Run patch_drop_last.sh and rerun."
        )

    if not np.array_equal(test["label"].to_numpy().astype(int), labels):
        raise SystemExit("label order mismatch, cannot join")

    test = test.assign(score=predictions)

    rows = []
    for pep, g in test.groupby("ligand_name"):
        auc = (roc_auc_score(g["label"], g["score"])
               if g["label"].nunique() == 2 else None)
        rows.append((pep, len(g), auc))
    rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))

    print()
    print(f"{'epitope_id':<12}{'n':>8}  auroc")
    for pep, n, auc in rows:
        print(f"{pep:<12}{n:>8}  {f'{auc:.4f}' if auc is not None else 'n/a'}")


if __name__ == "__main__":
    main()
