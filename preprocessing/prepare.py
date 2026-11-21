from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


def prepare_data(folder, cdr3_window=(8, 30)):
    folder = Path(folder)

    # positive pairs
    tables = []
    for source, filename, cdr3, rows in [
        ("IEDB", "iedb_positives.csv", "CDR3b_extended", 9652),
        ("VDJdb", "vdjdb_positives.csv", "CDR3b", 30274),
    ]:
        table = pd.read_csv(folder / filename, keep_default_na=False)
        if len(table) != rows:
            raise ValueError(f"Unexpected row count in {filename}: {len(table)}")
        table = table[[cdr3, "Peptide", "HLA"]].copy()
        table.columns = ["tcr_beta", "peptide", "hla"]
        table["source"] = source
        tables.append(table)

    # fixed filters
    positives = pd.concat(tables, ignore_index=True)
    for column in ["tcr_beta", "peptide", "hla"]:
        positives[column] = positives[column].str.upper().str.replace(r"\s+", "", regex=True)
    positives["hla"] = positives.hla.str.replace("HLA_", "HLA-", regex=False)
    known_binders = positives[["peptide", "tcr_beta"]].copy()
    alleles = ["HLA-A*02:01", "HLA-B*40:01"]
    amino_acids = r"[ACDEFGHIKLMNPQRSTVWY]+"
    valid = (
        positives.hla.isin(alleles)
        & positives.tcr_beta.str.fullmatch(amino_acids)
        & positives.peptide.str.fullmatch(amino_acids)
        & positives.tcr_beta.str.len().between(*cdr3_window)
        & positives.peptide.str.len().ge(7)
    )
    positives = positives.loc[valid].groupby(
        ["hla", "peptide", "tcr_beta"], as_index=False, sort=True
    ).agg(source=("source", lambda values: ";".join(sorted(set(values)))))

    # peptide-grouped splits
    split_file = folder / "folds.tsv"
    if split_file.exists():
        folds = pd.read_csv(split_file, sep="\t")[["peptide", "fold"]].drop_duplicates()
    else:
        folds = positives[["peptide"]].copy()
        folds["fold"] = -1
        splitter = GroupKFold(n_splits=5)
        for fold, (_, test) in enumerate(splitter.split(positives, groups=positives.peptide)):
            folds.loc[test, "fold"] = fold
        folds = folds.drop_duplicates()
        folds.to_csv(split_file, sep="\t", index=False)
    if folds.peptide.duplicated().any():
        raise ValueError("A peptide appears in more than one saved fold.")
    positives = positives.merge(folds, on="peptide", how="left", validate="many_to_one")
    if positives.fold.isna().any() or set(positives.fold) != set(range(5)):
        raise ValueError("Every positive needs one of the five saved folds.")
    positives["fold"] = positives.fold.astype(int)

    # background repertoire
    contigs = pd.read_csv(folder / "tcr_all_contig_annotations.csv", keep_default_na=False)
    keep = contigs.chain.eq("TRB")
    for column in ["productive", "high_confidence", "is_cell"]:
        keep &= contigs[column].astype(str).str.lower().eq("true")
    repertoire = contigs.loc[keep, ["cdr3"]].rename(columns={"cdr3": "tcr_beta"}).copy()
    repertoire["tcr_beta"] = repertoire.tcr_beta.str.upper().str.replace(r"\s+", "", regex=True)
    repertoire = repertoire.loc[
        repertoire.tcr_beta.str.fullmatch(amino_acids)
        & repertoire.tcr_beta.str.len().between(*cdr3_window)
    ].drop_duplicates()
    binders = known_binders.loc[known_binders.peptide.isin(positives.peptide), "tcr_beta"]
    repertoire = repertoire.loc[~repertoire.tcr_beta.isin(binders)].reset_index(drop=True)
    if repertoire.empty:
        raise ValueError("The background repertoire is empty after filtering.")

    # leakage checks
    if positives.groupby("peptide").fold.nunique().gt(1).any():
        raise ValueError("A peptide appears in more than one fold.")
    if repertoire.tcr_beta.isin(binders).any():
        raise ValueError("A known binder remains in the background repertoire.")
    coverage = positives.groupby("fold").agg(
        positives=("peptide", "size"), peptides=("peptide", "nunique")
    )
    return positives, repertoire, coverage
