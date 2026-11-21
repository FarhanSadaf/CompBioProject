# Evaluation

Can a model score well when an input is hidden? These controls check random pairing,
background sampling, and all five hard-negative settings against shuffled-label chance.

## How to run

Package setup: `.venv/bin/python -m pip install -r evaluation/requirements.txt` from the project folder.

Use the cleaned tables, fixed test pairs, and frozen sequence vectors:

```python
from evaluation.evaluate import run_controls, plot_controls
summary = run_controls(positives, train_background, test_sets, embeddings, SEEDS, n_jobs=3)
plot_controls(method="random_pairing")
plot_controls(method="p1")
```

`n_jobs` sets the number of folds running at once. Workers use one numerical-library thread each;
the main process saves the results. Use `n_jobs=1` for sequential execution.

## Results

The `results/controls_*` files contain metrics, summaries, per-peptide results, and pair scores.
There are 1,000 label shuffles and 1,000 peptide bootstrap samples per fit. A single-peptide fold
has no peptide bootstrap interval. Seed changes can change the results.
Each shuffle or bootstrap metric takes about O(M log M) time for M scored rows.

First prepare the common predictor benchmark from the saved test pairs:

```python
from evaluation.comparison import prepare_common_benchmark
common_pairs = prepare_common_benchmark()
```

This uses CDR3β lengths 8–19 and peptide lengths 8–11, matching epiTCR's input limits.
The common subset also requires a longer beta sequence recovered by preprocessing. The main study
is unchanged. Coverage for every original peptide is saved, including length and sequence exclusions.
After the two predictor setup steps, generate their scores on these same pairs:

```python
from evaluation.comparison import compare_predictors
from evaluation.predictors import predict_common_pairs
scores = predict_common_pairs()
predictor_metrics, comparison = compare_predictors(scores)
```

The adapter prepares epiTCR's input, converts the peptides to SMILES with RDKit, and makes TITAN's
longer-beta sequence table. It uses the original epiTCR runner and TITAN's released model/languages
in their own environments. TITAN uses every row in order, including the last partial batch.
Jacob's files are unchanged. Scores are saved in `results/common_predictor_scores.csv` with the
original fold and row IDs; duplicate or missing predictions raise an error.

`results/rq5_comparison.csv` compares the negative-swap range, seed spread, and predictor gap.
The local scores are recomputed on exactly the same common pairs. Matching inputs does not establish
that pretrained predictors never saw them during original training; check overlap before claiming
unseen-data performance.
