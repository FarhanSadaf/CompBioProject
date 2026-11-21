import argparse
import sys

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--per-peptide", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.pred)

    n_pos = int((df["binder"] == 1).sum())
    n_neg = int((df["binder"] == 0).sum())

    print(f"pairs  {len(df):,}")
    print(f"labels {n_pos:,} pos / {n_neg:,} neg")

    if n_pos == 0 or n_neg == 0:
        print("single class, auroc/auprc undefined")
        sys.exit(0)

    print(f"auroc  {roc_auc_score(df['binder'], df['predict_proba']):.4f}")
    print(f"auprc  {average_precision_score(df['binder'], df['predict_proba']):.4f} "
          f"(baseline {n_pos / len(df):.4f})")
    print(f"acc    {(df['binder'] == df['binder_pred']).mean():.4f}")

    if args.per_peptide:
        rows = []
        for pep, g in df.groupby("epitope"):
            auc = (roc_auc_score(g["binder"], g["predict_proba"])
                   if g["binder"].nunique() == 2 else None)
            rows.append((pep, len(g), auc))
        rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))
        print()
        print(f"{'epitope':<16}{'n':>8}  auroc")
        for pep, n, auc in rows[:30]:
            print(f"{pep:<16}{n:>8}  {f'{auc:.4f}' if auc is not None else 'n/a'}")
        if len(rows) > 30:
            print(f"... {len(rows) - 30} more")


if __name__ == "__main__":
    main()
