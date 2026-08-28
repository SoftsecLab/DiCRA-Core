# Paper figure scripts

Figures 1 and 2 are manually authored. The current experiment figures are
generated from verified evidence bundles:

| Paper figure | Script | Default output |
|---|---|---|
| Figure 3: robustness controls | `plot_dicra_robustness_controls.py` | `figures/dicra-robustness-controls.pdf` |
| Figure 4: component effects | `plot_dicra_ablation_effects.py` | `figures/dicra-ablation-effects.pdf` |
| Figure 5: matched diagnostic plane | `plot_dicra_diagnostic_plane.py` | `figures/dicra-diagnostic-plane.pdf` |

The scripts accept either an extracted evidence directory or its `.tar.gz`
archive. Legacy `RECAP` identifiers inside evidence files are displayed as
`DiCRA` in the figures.

```bash
python scripts/figures/plot_dicra_robustness_controls.py \
  --cross-interface-evidence evidence_99_documented_dev_v1_anonymous.tar.gz \
  --multi-order-evidence evidence_documented_dev_multi_order_81_anonymous.tar.gz \
  --ablation-evidence evidence_documented_dev_single_variable_ablation_45_anonymous.tar.gz

python scripts/figures/plot_dicra_ablation_effects.py \
  --evidence evidence_documented_dev_single_variable_ablation_45_anonymous.tar.gz

python scripts/figures/plot_dicra_diagnostic_plane.py \
  --evidence evidence_63_documented_dev_v1_anonymous.tar.gz
```

Use `--png PATH` to produce a raster preview in addition to the vector PDF.
The other plotting scripts in this directory target earlier diagnostic figures
and are not sources for Figures 3--5 in the current paper.
