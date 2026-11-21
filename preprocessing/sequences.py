import json
import re
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO


TENX = (
    "https://cf.10xgenomics.com/samples/cell-vdj/5.0.0/"
    "sc5p_v2_hs_PBMC_10k_multi_5gex_5fb_b_t/"
    "sc5p_v2_hs_PBMC_10k_multi_5gex_5fb_b_t_vdj_t_all_contig_annotations.json"
)


def prepare_receptors(data="data", output="results/receptor_sequences.csv"):
    data, output = Path(data), Path(output)
    if output.exists():
        return pd.read_csv(output, keep_default_na=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    downloads = output.parent / "downloads"
    downloads.mkdir(exist_ok=True)

    # supplied variable domains
    prefixes, positives = {}, []
    for source, cdr3 in [("iedb", "CDR3b_extended"), ("vdjdb", "CDR3b")]:
        table = pd.read_csv(data / f"{source}_positives.csv", keep_default_na=False)
        for row in table.itertuples(index=False):
            for gene, sequence in [(row.Va, row.TCRa), (row.Vb, row.TCRb)]:
                if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{12}", sequence[:12]):
                    prefixes.setdefault(gene.split("*")[0], set()).add(sequence[:12])
            beta = getattr(row, cdr3)
            sequence = row.TCRb
            if source == "vdjdb":
                sequence = sequence.replace(row.CDR3b_extended, beta, 1)
            if (re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{8,30}", beta) and beta in sequence
                    and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{80,180}", sequence)):
                positives.append({"tcr_beta": beta, "tcr_beta_sequence": sequence,
                                  "tcr_alpha_sequence": "", "sequence_source": f"IMMREP25 {source}:TCRb"})

    # observed 10x variable domains
    path = downloads / "tenx_contigs.json"
    if not path.exists():
        response = requests.get(TENX, timeout=180)
        response.raise_for_status()
        response.json()
        path.write_bytes(response.content)
    contigs = []
    for record in json.loads(path.read_text()):
        if not all(record.get(key) for key in ["is_cell", "high_confidence", "productive", "full_length"]):
            continue
        parts = {item["feature"]["region_type"]: item for item in record["annotations"]}
        if "L-REGION+V-REGION" not in parts or "J-REGION" not in parts:
            continue
        feature = parts["L-REGION+V-REGION"]["feature"]
        chain = feature["chain"]
        amino = record.get("aa_sequence") or ""
        starts = {amino.find(prefix) for prefix in prefixes.get(feature["gene_name"], []) if prefix in amino}
        if chain not in ["TRA", "TRB"] or len(starts) != 1 or record.get("start_codon_pos") is None:
            continue
        end = (parts["J-REGION"]["contig_match_end"] - record["start_codon_pos"]) // 3
        sequence = amino[starts.pop():end]
        if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{80,180}", sequence):
            continue
        if not record.get("cdr3") or record["cdr3"] not in sequence:
            continue
        contigs.append({"barcode": record["barcode"], "chain": chain, "cdr3": record["cdr3"],
                        "sequence": sequence, "contig": record["contig_name"]})
    contigs = pd.DataFrame(contigs)
    if contigs.empty:
        raise ValueError("No complete receptor variable domains were recovered from 10x.")

    # keep unambiguous observed alpha/beta pairings
    counts = contigs.groupby(["barcode", "chain"]).sequence.nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=["TRA", "TRB"], fill_value=0)
    single = counts.index[counts.TRA.eq(1) & counts.TRB.eq(1)]
    alpha = contigs.loc[contigs.chain.eq("TRA") & contigs.barcode.isin(single)]
    alpha = alpha.drop_duplicates("barcode")[["barcode", "sequence", "contig"]]
    alpha.columns = ["barcode", "tcr_alpha_sequence", "alpha_contig"]
    beta = contigs.loc[contigs.chain.eq("TRB")].merge(alpha, on="barcode", how="left").fillna("")
    paired = beta.loc[beta.tcr_alpha_sequence.ne("")].groupby("cdr3").tcr_alpha_sequence.nunique()
    beta.loc[beta.cdr3.isin(paired.index[paired.gt(1)]), ["tcr_alpha_sequence", "alpha_contig"]] = ""
    beta["sequence_source"] = "10x:" + beta.contig + ";" + beta.alpha_contig
    beta = beta.rename(columns={"cdr3": "tcr_beta", "sequence": "tcr_beta_sequence"})
    columns = ["tcr_beta", "tcr_beta_sequence", "tcr_alpha_sequence", "sequence_source"]
    receptors = pd.concat([pd.DataFrame(positives), beta[columns]], ignore_index=True).drop_duplicates()
    unique = receptors.groupby("tcr_beta").tcr_beta_sequence.nunique()
    receptors = receptors.loc[receptors.tcr_beta.isin(unique.index[unique.eq(1)])].copy()
    receptors["paired"] = receptors.tcr_alpha_sequence.ne("")
    receptors = receptors.sort_values(["paired", "sequence_source"], ascending=[False, True])
    receptors = receptors.drop_duplicates("tcr_beta")[columns].reset_index(drop=True)
    receptors.to_csv(output, index=False)
    return receptors


def prepare_mhc(folder="results/downloads"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    urls = {f"{gene}_prot.fasta": f"https://raw.githubusercontent.com/ANHIG/IMGTHLA/3660/fasta/{gene}_prot.fasta"
            for gene in ["A", "B"]}
    urls["b2m.fasta"] = "https://rest.uniprot.org/uniprotkb/P61769.fasta"
    for filename, url in urls.items():
        path = folder / filename
        if not path.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            if not response.text.startswith(">"):
                raise ValueError(f"Expected FASTA protein sequences from {url}")
            path.write_text(response.text)
    b2m = str(SeqIO.read(folder / "b2m.fasta", "fasta").seq)[20:]
    rows = []
    for allele in ["A*02:01", "B*40:01"]:
        records = SeqIO.parse(folder / f"{allele[0]}_prot.fasta", "fasta")
        reference = next((r for r in records if r.description.split()[1].startswith(allele + ":")
                          and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{350,380}", str(r.seq))), None)
        if reference is None:
            raise ValueError(f"No complete reference protein for {allele} in IPD-IMGT/HLA 3.66.0.")
        # extracellular MHC and mature beta2m (UniProt P01892 / P61769)
        rows.append({"hla": "HLA-" + allele, "mhc_sequence": str(reference.seq)[24:308],
                     "b2m_sequence": b2m, "mhc_source": f"IPD-IMGT/HLA 3.66.0:{reference.description.split()[1]};UniProt:P61769"})
    return pd.DataFrame(rows)
