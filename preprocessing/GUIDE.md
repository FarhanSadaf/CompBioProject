# Preprocessing

In this part, we clean the positive pairs and background TCRs. We keep HLA-A*02:01 and HLA-B*40:01,
use CDR3β lengths from 8 to 30, and remove duplicate pairs. IEDB uses `CDR3b_extended` and
VDJdb uses `CDR3b` so both include the same CDR3 boundary residues.

## How to run

From the project folder, install this part's packages if needed:

```bash
.venv/bin/python -m pip install -r preprocessing/requirements.txt
```

Then run:

```python
from preprocessing.prepare import prepare_data
positives, repertoire, coverage = prepare_data("data", cdr3_window=(8, 30))
from preprocessing.sequences import prepare_receptors
receptors = prepare_receptors()
```

We reuse `data/folds.tsv`; if it is missing, `GroupKFold` creates five folds grouped by peptide
and saves the assignments.
It uses peptide groups to balance row counts as far as possible. No peptide is divided between folds.

## Results

`positives` contains the cleaned pairs and fold assignments. `repertoire` contains unique, productive,
high-confidence TRB sequences from called cells in the supplied 10x donor. We remove TCRs with a
recorded specificity for any retained peptide in either positive file, across all HLA types.
These are assumed negatives; other specificities and the donor's HLA types are unknown.

`coverage` gives the positive and peptide counts in each fold. The saved split keeps every peptide
in one fold, but allows a TCR to appear across folds. One fold contains only GILGFVFTL because that
peptide has many positives. This tests unseen peptides, not unseen TCRs.

`receptors` contains longer beta sequences for the published predictor and paired alpha/beta
variable regions for the audit. The matching 10x amino-acid annotations download once and are
cached in `results/downloads/`. We trim at the supplied V-framework and observed J boundaries.
Only single alpha/beta pairings in a cell are used; ambiguous beta sequences are excluded, and
ambiguous alpha pairings are left empty. The supplied VDJdb beta sequences have their duplicated
CDR3 boundary residues normalized. The lookup is saved in `results/receptor_sequences.csv`.

For N positive rows and R repertoire rows, the cost is O(N log N + R) time and O(N + R) memory.
