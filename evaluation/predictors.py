import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


def predict_common_pairs(folder="results", predictors="tcr_predictors"):
    from rdkit import Chem

    folder, predictors = Path(folder).resolve(), Path(predictors).resolve()
    pairs = pd.read_csv(folder / "common_test_pairs.csv.gz")
    key = ["fold", "row_id"]
    if pairs.duplicated(key).any() or pairs.empty:
        raise ValueError("The common pairs need unique fold/row IDs.")
    epi = predictors / "epitcr"
    titan = predictors / "titan"
    epi_python = epi / "epitcr_env/bin/python"
    epi_model = epi / "models_unzipped/rdforestWithoutMHCModel.pickle"
    required = [epi_python, epi_model, epi / "epiTCR/predict.py",
                titan / "TITAN/trained_model/weights/best_ROC-AUC_bimodal_mca.pt"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Complete the two predictor setup steps in README.md first.")
    titan_python = titan / "titan_env/bin/python"
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if titan_python.exists():
        command = [str(titan_python)]
    elif conda and Path(conda).is_file():
        command = [conda, "run", "--no-capture-output", "-n", "titan", "python"]
    else:
        raise FileNotFoundError("The TITAN environment is missing; follow its setup guide.")

    # epiTCR input and released runner
    pairs[[*key, "tcr_beta", "peptide", "label"]].rename(columns={
        "tcr_beta": "CDR3b", "peptide": "epitope", "label": "binder",
    }).to_csv(folder / "epitcr_common_input.csv", index=False)
    environment = os.environ.copy()
    environment.update(TEST=str(folder / "epitcr_common_input.csv"), MODEL=str(epi_model),
                       OUT=str(folder / "epitcr_common_predictions.csv"),
                       PYTHONDONTWRITEBYTECODE="1")
    environment["PATH"] = str(epi_python.parent) + os.pathsep + environment.get("PATH", "")
    subprocess.run(["bash", "run_replication.sh"], cwd=epi, env=environment, check=True)

    # TITAN's tab-separated sequence and SMILES tables
    proteins = pairs[["tcr_beta", "tcr_beta_sequence"]].drop_duplicates().reset_index(drop=True)
    proteins["sequence_id"] = [f"t{i}" for i in range(len(proteins))]
    peptides = pairs[["peptide"]].drop_duplicates().reset_index(drop=True)
    peptides["ligand_name"] = [f"p{i}" for i in range(len(peptides))]
    peptides["smiles"] = [Chem.MolToSmiles(Chem.MolFromFASTA(peptide)) for peptide in peptides.peptide]
    proteins[["tcr_beta_sequence", "sequence_id"]].to_csv(folder / "titan_tcrs.csv", sep="\t", header=False, index=False)
    peptides[["smiles", "ligand_name"]].to_csv(folder / "titan_peptides.smi", sep="\t", header=False, index=False)
    labels = pairs.merge(proteins[["tcr_beta", "sequence_id"]], on="tcr_beta", validate="many_to_one")
    labels = labels.merge(peptides[["peptide", "ligand_name"]], on="peptide", validate="many_to_one")
    if not labels[key].equals(pairs[key]):
        raise ValueError("The TITAN input rows changed order.")
    labels[["ligand_name", "sequence_id", "label"]].to_csv(folder / "titan_pairs.csv")
    subprocess.run([*command, str(Path(__file__).resolve()), str(folder), str(titan / "TITAN/trained_model")],
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True)

    # join scores by the original IDs
    results = []
    for name, filename, column in [("epiTCR", "epitcr_common_predictions.csv", "predict_proba"),
                                    ("TITAN", "titan_common_predictions.csv", "score")]:
        scored = pd.read_csv(folder / filename)
        if scored.duplicated(key).any() or set(map(tuple, scored[key].to_numpy())) != set(map(tuple, pairs[key].to_numpy())):
            raise ValueError(f"{name} did not return every common pair exactly once.")
        if not np.isfinite(scored[column]).all() or not scored[column].between(0, 1).all():
            raise ValueError(f"Invalid {name} probabilities.")
        results.append(scored[[*key, column]].rename(columns={column: "score"}).assign(predictor=name))
    scores = pd.concat(results, ignore_index=True)
    scores.to_csv(folder / "common_predictor_scores.csv", index=False)
    return folder / "common_predictor_scores.csv"


def predict_titan(folder, model_folder):
    import torch
    from paccmann_predictor.models import MODEL_FACTORY
    from paccmann_predictor.utils.utils import get_device
    from pytoda.datasets import DrugAffinityDataset
    from pytoda.proteins import ProteinFeatureLanguage, ProteinLanguage
    from pytoda.smiles.smiles_language import SMILESTokenizer

    folder, model_folder = Path(folder), Path(model_folder)
    params = json.loads((model_folder / "model_params.json").read_text())
    checkpoint = model_folder / "weights/best_ROC-AUC_bimodal_mca.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch.manual_seed(123456)
    device = get_device()
    smiles = SMILESTokenizer.from_pretrained(
        str(model_folder), randomize=False,
        add_start_and_stop=params.get("ligand_start_stop_token", True),
        padding=params.get("ligand_padding", True), padding_length=params["ligand_padding_length"],
        augment=False, canonical=params.get("smiles_canonical", False),
        kekulize=params.get("smiles_kekulize", False), all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False), remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False), selfies=params.get("selfies", False),
        sanitize=params.get("sanitize", False),
    )
    language = ProteinFeatureLanguage if params.get("receptor_embedding") == "predefined" else ProteinLanguage
    protein = language.load(str(model_folder / "protein_language.pkl"))
    dataset = DrugAffinityDataset(
        drug_affinity_filepath=str(folder / "titan_pairs.csv"), smi_filepath=str(folder / "titan_peptides.smi"),
        protein_filepath=str(folder / "titan_tcrs.csv"), smiles_language=smiles, protein_language=protein,
        smiles_padding=params.get("ligand_padding", True), smiles_padding_length=params["ligand_padding_length"],
        smiles_add_start_and_stop=params.get("ligand_add_start_stop", True), smiles_augment=False,
        smiles_canonical=params.get("smiles_canonical", False), smiles_kekulize=params.get("smiles_kekulize", False),
        smiles_all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        smiles_all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        smiles_remove_bonddir=params.get("smiles_remove_bonddir", False),
        smiles_remove_chirality=params.get("smiles_remove_chirality", False), smiles_selfies=params.get("selfies", False),
        smiles_sanitize=params.get("sanitize", False),
        protein_amino_acid_dict=params.get("protein_amino_acid_dict", "iupac"),
        protein_padding=params.get("receptor_padding", True), protein_padding_length=params["receptor_padding_length"],
        protein_add_start_and_stop=params.get("receptor_add_start_stop", True),
        protein_augment_by_revert=False, drug_affinity_dtype=torch.float, backend="eager", iterate_dataset=True,
    )
    pairs = pd.read_csv(folder / "common_test_pairs.csv.gz")
    if list(dataset.drug_affinity_df.index) != list(range(len(pairs))):
        raise ValueError("TITAN dropped or reordered input rows.")
    model = MODEL_FACTORY[params.get("model_fn", "bimodal_mca")](params).to(device)
    model._associate_language(smiles)
    model._associate_language(protein)
    model.load(str(checkpoint), map_location=device)
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=False,
                                         drop_last=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for ligand, receptor, _ in loader:
            score, _ = model(ligand.to(device), receptor.to(device))
            predictions.append(score.flatten().cpu().numpy())
    pairs[["fold", "row_id"]].assign(score=np.concatenate(predictions)).to_csv(
        folder / "titan_common_predictions.csv", index=False,
    )


if __name__ == "__main__":
    predict_titan(sys.argv[1], sys.argv[2])
