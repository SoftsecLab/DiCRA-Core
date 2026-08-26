# Reproducibility artifacts

Real per-run artifacts are distributed as release assets so large, immutable
evidence files do not enter Git history.

Use the `_anonymous.tar.gz` transport bundles and
`ANONYMOUS_SHA256SUMS.txt` during double-blind review. These copies replace
private machine paths in protocol-only files, preserve every real
`config.json` and `results.json` byte-for-byte, update dependent manifest
hashes, and pass the original independent bundle verifiers.

## Release assets

| Asset | Scope |
|---|---|
| `evidence_63_documented_dev_v1.tar.gz` | Canonical fixed-global comparison |
| `evidence_99_documented_dev_v1.tar.gz` | Fixed-global and expanding-head comparison |
| `evidence_documented_dev_selection_189.tar.gz` | Development-only hyperparameter selection |
| `evidence_documented_dev_single_variable_ablation_45.tar.gz` | Single-variable ablations |
| `evidence_documented_dev_multi_order_81.tar.gz` | Multi-class-order robustness |
| `evidence_recap_diagnostics_paired_9.tar.gz` | Paired classifier and prototype diagnostics |
| `evidence_63_documented_dev_bert_cased_v1.tar.gz` | BERT-base-cased robustness |

`recap` is retained in legacy artifact and method identifiers to preserve
published hashes. The public method name is DiCRA.

## Verification

Download the required release assets and verify them from the repository root:

```bash
sha256sum -c results/ANONYMOUS_SHA256SUMS.txt
```

Each extracted evidence bundle contains a standard-library-only verifier under
its `scripts/` directory and a `verification_report.json`. See the bundle's
manifest for the exact verification command and its mapping to paper tables.
