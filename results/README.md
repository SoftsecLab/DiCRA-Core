# Reproducibility artifacts

Real per-run artifacts are distributed as release assets so large, immutable
evidence files do not enter Git history.

Use sanitized transport bundles during double-blind review. The standalone
seven-bundle distribution uses `_anonymous.tar.gz` filenames; the five bundles
inside `data.zip` retain their unsuffixed legacy filenames. Both forms remove
private machine paths, preserve every real `config.json` and `results.json`,
and pass the independent bundle verifiers.

## Release assets

The [`v0.1.0-evidence` release](https://github.com/loveyou-3001/DiCRA-Core/releases/tag/v0.1.0-evidence)
provides `data.zip`, a verified outer archive containing the first five bundles
listed below. Its SHA256 is
`a04e0490fbc6cc5da7b56b7929133fde7c9951a4597fe3724adfcf19c001b763`.

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

The SHA256 shown above verifies `data.zip`; after extraction, run each bundle's
independent verifier. `ANONYMOUS_SHA256SUMS.txt` applies only when the seven
standalone `_anonymous.tar.gz` assets are downloaded together:

```bash
sha256sum -c results/ANONYMOUS_SHA256SUMS.txt
```

Each extracted evidence bundle contains a standard-library-only verifier under
its `scripts/` directory and a `verification_report.json`. See the bundle's
manifest for the exact verification command and its mapping to paper tables.
