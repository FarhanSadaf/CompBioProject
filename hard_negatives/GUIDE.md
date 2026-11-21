# Hard negatives

This changes how training negatives are chosen. Keep the positives, model, folds, and test pairs fixed.
Compare random pairing and background sampling with p99, p75, p50, p25, and p1. The p1 pool is hardest.

## How to run

Use Python 3.11. Install with `.venv/bin/python -m pip install -r hard_negatives/requirements.txt`
from the project folder. This includes Boltz.

1. Make the frozen sequence vectors.
2. Run the negative comparison and the 3-mer/gradient-boosting check.
3. Plot the saved scores.

```python
from hard_negatives.experiment import embed_sequences, run_experiments, plot_hardness
embeddings = embed_sequences(sequences, cache="data/embeddings.npz")
summary = run_experiments(positives, train_background, test_sets, embeddings, SEEDS, n_jobs=3)
plot_hardness()
```

Run up to three folds together with `n_jobs=3`, or choose `1` to run them in order.
Each worker uses one numerical-library thread and returns its results for saving.

The sequence step recovers observed alpha/beta variable regions from the matching 10x annotations.
It downloads the two MHC references from IPD-IMGT/HLA 3.66.0 and beta2-microglobulin from UniProt.
It selects the 200 nearest hard pairs with usable paired sequences and records exclusions.

```python
from hard_negatives.structures import prepare_sequences, prepare_inputs, fold_pairs, score_structures
prepare_sequences()
prepare_inputs()
fold_pairs()
audit = score_structures(threshold=0.6)
```

The fixed rule flags a mean TCR–MHC/peptide interface ipTM of at least 0.6. This is an exploratory
confidence rule, not a calibrated binding probability. Set the rule before seeing results.
The folding seed is 559. Boltz sends the protein sequences to its MSA server and uses a CUDA GPU
when available, otherwise CPU. Folding all 200 complexes is much slower on CPU.
Its small-molecule affinity head is not used.

## Results

- `results/negatives_*`: metrics, predictions, and summaries.
- `results/hardness_checks.csv`: actual cutoffs, pool sizes, and repeated pairs.
- `results/structure_candidates.csv`: 200 unique hard pairs.
- `results/structure_sequences.csv`: automatically recovered proteins and source records.
- `results/structure_coverage.csv`: eligible and selected pairs for each peptide.
- `results/structure_scores.csv`: the complete 200-pair audit and flagged fraction.

To check whether removing flagged negatives changes the result, use the same seeds and test pairs:

```python
blocked = audit.loc[audit.plausible_binder.eq(1), ["hla", "peptide", "tcr_beta"]]
run_experiments(positives, train_background, test_sets, embeddings, SEEDS,
                methods=["p1"], excluded_pairs=blocked, output="results/after_audit")
from evaluation.comparison import compare_after_audit
compare_after_audit()
```

The sampler replaces excluded pairs from the remaining pool to keep 5:1 rows. An empty pool raises
an error. The audit is limited to pairs with recoverable, unambiguous receptor context; its flagged
fraction is not an estimate for all negatives. Structural confidence does not confirm binding.
Predictions can vary between runs.
RapidFuzz uses O(R B L²) work and an O(R B) distance matrix for R candidates and B binders of length L.

Input and output formats follow the [Boltz documentation](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md).
