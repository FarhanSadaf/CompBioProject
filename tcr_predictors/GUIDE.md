# Guide

Two published TCR–epitope predictors, set up to run on a Mac.

## Goal

My job is to run two predictors that other people published, and report what
they score. They are the outside reference point. When our team changes how
the negatives are built, we need to know whether the score moved more than the
gap between two real published models.

---

## Predictor 1: epiTCR

Paper: Pham et al. Bioinformatics 39(5), btad284, 2023.
Code: https://github.com/ddiem-ri-4D/epiTCR

```bash
cd epitcr
git clone https://github.com/ddiem-ri-4D/epiTCR.git
bash setup_mac.sh
source epitcr_env/bin/activate
python check_env.py
bash run_replication.sh
python score.py --pred epitcr_predictions.csv --per-peptide
```

Runs natively on Apple Silicon.

---

## Predictor 2: TITAN

Paper: Weber, Born & Rodriguez Martínez. Bioinformatics 37(Suppl 1),
i237–i244, 2021.
Code: https://github.com/PaccMann/TITAN

```bash
cd titan
git clone https://github.com/PaccMann/TITAN.git
bash setup_mac.sh
conda activate titan
python check_env.py
bash patch_drop_last.sh
bash run_replication.sh
python score.py \
  --pred TITAN/trained_model/results/repro.npy \
  --test TITAN/datasets/trained_model_test.csv \
  --per-peptide
```

Needs conda and Rosetta on Apple Silicon.

---

## Files in each folder

| File | What it does |
|---|---|
| `setup_mac.sh` | Builds the environment |
| `requirements_mac.txt` | Package list. Fixed version of the repo's own, which does not install |
| `check_env.py` | Checks everything works before running anything slow. Run first |
| `run_replication.sh` | Runs the model |
| `score.py` | Prints AUROC, AUPRC and accuracy |
