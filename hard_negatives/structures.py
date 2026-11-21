import json
import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd


def prepare_sequences(folder="results", receptors=None, mhc=None):
    from preprocessing.sequences import prepare_receptors, prepare_mhc

    folder = Path(folder)
    pool = pd.read_csv(folder / "structure_pool.csv")
    if receptors is None:
        receptors = prepare_receptors(output=folder / "receptor_sequences.csv")
    if mhc is None:
        mhc = prepare_mhc(folder / "downloads")
    paired = receptors.loc[receptors.tcr_alpha_sequence.ne("")]
    pairs = pool.merge(paired, on="tcr_beta", how="left", validate="many_to_one")
    usable = pairs.tcr_alpha_sequence.notna() & pairs.tcr_beta_sequence.notna()
    coverage = pairs.assign(usable_pair=usable).groupby(["hla", "peptide"]).agg(
        hard_pairs=("tcr_beta", "size"), with_paired_sequences=("usable_pair", "sum"),
    )
    pairs = pairs.loc[usable].sort_values(["distance", "hla", "peptide", "tcr_beta"])
    pairs = pairs.drop_duplicates(["hla", "peptide", "tcr_beta"]).head(200).reset_index(drop=True)
    if len(pairs) != 200:
        raise ValueError(f"The hard pool has only {len(pairs)} usable paired receptors; 200 are needed for the audit.")
    pairs["row_id"] = np.arange(200)
    pairs = pairs.merge(mhc, on="hla", validate="many_to_one")
    pairs["sequence_source"] = pairs.sequence_source + ";" + pairs.mhc_source
    pairs.to_csv(folder / "structure_sequences.csv", index=False)
    pairs[["row_id", "tcr_beta", "peptide", "hla", "distance"]].to_csv(folder / "structure_candidates.csv", index=False)
    selected = pairs.groupby(["hla", "peptide"]).size().rename("selected")
    coverage.join(selected).fillna(0).to_csv(folder / "structure_coverage.csv")
    return pairs


def prepare_inputs(candidates="results/structure_candidates.csv",
                   sequences="results/structure_sequences.csv", output="results/boltz_inputs"):
    pairs = pd.read_csv(candidates)
    if len(pairs) != 200 or set(pairs.row_id) != set(range(200)):
        raise ValueError("The proposal audit needs exactly 200 uniquely identified candidate pairs.")
    if pairs.duplicated(["hla", "peptide", "tcr_beta"]).any():
        raise ValueError("A biological pair occurs more than once in the audit.")
    sequence_file = Path(sequences)
    if not sequence_file.exists():
        raise FileNotFoundError(
            "Run prepare_sequences() to recover the selected receptor and MHC sequences."
        )
    chains = pd.read_csv(sequence_file, keep_default_na=False)
    columns = ["tcr_alpha_sequence", "tcr_beta_sequence", "mhc_sequence", "b2m_sequence"]
    required = {"row_id", "sequence_source", *columns}
    if not required.issubset(chains.columns):
        raise ValueError("Missing sequence columns: " + ", ".join(sorted(required - set(chains.columns))))
    if chains.row_id.duplicated().any() or set(chains.row_id) != set(pairs.row_id):
        raise ValueError("Supply sequences for all 200 candidate IDs exactly once.")
    pairs = pairs.merge(chains[["row_id", "sequence_source", *columns]], on="row_id", validate="one_to_one")
    for column, minimum in zip(columns, [80, 80, 150, 80]):
        if not (pairs[column].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")
                & pairs[column].str.len().ge(minimum)).all():
            raise ValueError(f"{column} must contain complete canonical protein/domain sequences.")
    if pairs.sequence_source.str.strip().eq("").any():
        raise ValueError("Record where each receptor and MHC sequence came from.")
    if any(row.tcr_beta not in row.tcr_beta_sequence for row in pairs.itertuples()):
        raise ValueError("A supplied beta chain does not contain its candidate CDR3beta.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = {f"pair_{int(value):03d}.yaml" for value in pairs.row_id}
    if any(path.name not in expected for path in output.glob("*.yaml")):
        raise ValueError("The input folder contains YAML files from another candidate set.")
    for row in pairs.itertuples():
        proteins = [row.tcr_alpha_sequence, row.tcr_beta_sequence,
                    row.mhc_sequence, row.b2m_sequence, row.peptide]
        payload = {"version": 1, "sequences": [
            {"protein": {"id": chain, "sequence": sequence}}
            for chain, sequence in zip("ABCDE", proteins)
        ]}
        # JSON is also valid YAML. Boltz generates MSAs when the runner is called.
        path = output / f"pair_{int(row.row_id):03d}.yaml"
        content = json.dumps(payload, indent=2) + "\n"
        if path.exists() and path.read_text() != content:
            raise ValueError("The existing structural inputs differ. Use a new input folder.")
        path.write_text(content)
    pairs.to_csv(output.parent / "structure_inputs.csv", index=False)
    return output


def fold_pairs(inputs="results/boltz_inputs", output="results/boltz",
               executable=".venv/bin/boltz", accelerator="auto", timeout=None, seed=559):
    if accelerator == "auto":
        import torch
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    program = shutil.which(executable) or str(Path(executable).resolve())
    if not Path(program).is_file():
        raise FileNotFoundError("Install requirements_combined.txt or pass the Boltz executable path.")
    inputs = Path(inputs)
    if len(list(inputs.glob("pair_*.yaml"))) != 200:
        raise ValueError("Prepare the 200 structural inputs before calling Boltz.")
    output = Path(output)
    fingerprints = {"seed": seed, "model": "boltz2", "inputs": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.glob("*.yaml")
    }}
    record = output / "input_hashes.json"
    if record.exists() and json.loads(record.read_text()) != fingerprints:
        raise ValueError("These are different inputs. Choose a new Boltz output folder.")
    if not record.exists() and output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output folder for the first run.")
    output.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(fingerprints, indent=2) + "\n")
    command = [program, "predict", str(inputs.resolve()), "--out_dir", str(Path(output).resolve()),
               "--use_msa_server", "--accelerator", accelerator, "--diffusion_samples", "1",
               "--output_format", "mmcif", "--model", "boltz2", "--seed", str(seed)]
    subprocess.run(command, check=True, timeout=timeout)


def score_structures(threshold, inputs="results/structure_inputs.csv", folder="results/boltz",
                     output="results/structure_scores.csv"):
    if not 0 <= threshold <= 1:
        raise ValueError("Choose the interface-confidence threshold before inspecting results.")
    pairs = pd.read_csv(inputs)
    if len(pairs) != 200 or pairs.row_id.duplicated().any():
        raise ValueError("The audit must cover the original 200 candidates.")
    scores = []
    hashes = json.loads((Path(folder) / "input_hashes.json").read_text())["inputs"]
    for row in pairs.itertuples():
        name = f"pair_{int(row.row_id):03d}"
        proteins = [row.tcr_alpha_sequence, row.tcr_beta_sequence,
                    row.mhc_sequence, row.b2m_sequence, row.peptide]
        payload = {"version": 1, "sequences": [
            {"protein": {"id": chain, "sequence": sequence}}
            for chain, sequence in zip("ABCDE", proteins)
        ]}
        fingerprint = hashlib.sha256((json.dumps(payload, indent=2) + "\n").encode()).hexdigest()
        if hashes.get(name + ".yaml") != fingerprint:
            raise ValueError(f"The recorded structure input does not match {name}.")
        files = list(Path(folder).rglob(f"confidence_{name}_model_0.json"))
        if len(files) != 1:
            raise ValueError(f"Expected one Boltz confidence file for {name}; found {len(files)}.")
        confidence = json.loads(files[0].read_text())
        interface = confidence["pair_chains_iptm"]
        # Cross-interface pairs only: TCR alpha/beta against MHC heavy chain/peptide.
        values = [float(interface[a][b]) for a in ["0", "1"] for b in ["2", "4"]]
        if not np.isfinite(values).all() or not all(0 <= value <= 1 for value in values):
            raise ValueError(f"Invalid interface scores for {name}.")
        score = float(np.mean(values))
        scores.append({"row_id": row.row_id, "interface_confidence": score,
                       "plausible_binder": int(score >= threshold), "threshold": threshold})
    audit = pairs[["row_id", "hla", "peptide", "tcr_beta"]].merge(
        pd.DataFrame(scores), on="row_id", validate="one_to_one",
    )
    audit.to_csv(output, index=False)
    print("Pairs audited:", len(audit), "| Fraction flagged:", audit.plausible_binder.mean())
    return audit
