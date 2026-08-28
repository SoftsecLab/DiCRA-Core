# DiCRA

Minimal research code for diagnosis-driven representation consolidation and
classifier alignment in parameter-efficient continual text classification.

## Install

```bash
conda env create -f environment.yml
conda activate dicra
```

The audited runtime uses Python 3.11. Set the model root with:

```bash
export DICRA_MODEL_ROOT=/path/to/local/models
```

## Prepare data

```bash
python scripts/data/prepare_clinc150.py \
  --raw_path /path/to/clinc150/data_full.json \
  --output_root data/clinc150 --num_tasks 15 --seed 42 --overwrite

python scripts/data/prepare_banking77.py \
  --output_root data/banking77 --num_tasks 7 --seed 42 --overwrite

python scripts/data/prepare_fewrel_acl2024.py \
  --source_pkl /path/to/FewRel-2021.pkl \
  --output_root data/fewrel_acl2024
```

## Run the canonical comparison

Preview the frozen 7-method, 3-dataset, 3-seed grid:

```bash
python scripts/run_canonical.py --dry-run
```

Run with safe resume and aggregate the matrices:

```bash
python scripts/run_canonical.py --keep-going
python scripts/run_canonical.py --report-only
```

Results are written to `outputs/canonical/summary.{json,tsv}`.

## Verify released evidence

Each supplementary evidence bundle contains its own standard-library-only
verifier. For the canonical fixed-global bundle:

```bash
python evidence_63_documented_dev_v1_anonymous/scripts/recompute_all_results.py \
  evidence_63_documented_dev_v1_anonymous
```

Expected marker: `SUBMISSION_EVIDENCE_63_RUNS_OK`.

## Repository scope

```text
main.py       DiCRA training entry point
src/          core method and evaluation implementation
baselines/    five controlled LoRA baselines
scripts/      canonical runner, data preparation, and paper figures
tests/        core unit and protocol tests
experiments/  frozen selection and artifact manifests
```

Extended ablations, robustness studies, and audit tooling are distributed in
the supplementary evidence bundles rather than the main code repository.

## License

MIT. See `THIRD_PARTY.md` for adapted-baseline and external-artifact boundaries.
