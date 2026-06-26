---
name: use-tradeseq
description: How to run trajectory-based differential expression (NB-GAM) on AnnData with tradeSeq-python — a port of R/Bioconductor tradeSeq. Fit per-gene smoothers along lineages, then run the Wald battery.
---

# tradeSeq-python

Python port of [statOmics/tradeSeq](https://github.com/statOmics/tradeSeq) (1.13.12). Operates on `AnnData` instead of R's `SingleCellExperiment`; every R accessor (`slingPseudotime`, `slingCurveWeights`, …) collapses to a direct AnnData slot read.

`import tradeseq as ts` (the distribution is `tradeSeq-python`, the import name is `tradeseq`).

## Mental model — fit once, then test

tradeSeq is a **two-phase state machine over an AnnData**:

```
trajectory (you provide)        fit_gam (writes state)           tests (read state)
obsm.pseudotime      ─┐                                    ┌─ association_test
obsm.cell_weights    ─┼─► fit_gam ─► var.tradeseq_converged┼─ start_vs_end_test
layers.counts        ─┘            varm.tradeseq_beta      ├─ diff_end_test
                                   varm.tradeseq_Sigma     ├─ pattern_test / early_de_test
                                   uns.tradeseq            ├─ condition_test
                                                           ├─ predict_cells / predict_smooth
                                                           ├─ cluster_expression_patterns
                                                           └─ cascade → plot_cascade
```

1. **You supply the trajectory.** tradeSeq does **not** infer one — run slingshot / Monocle / etc. first and store `obsm['pseudotime']` and `obsm['cell_weights']` (both `cells × lineages`). `ts.load_paul15()` returns a ready 2-lineage example.
2. **`fit_gam` is the gate.** It fits a per-gene negative-binomial GAM and writes the coefficient/covariance/convergence payloads under the `tradeseq` namespace. **Every test below reads those slots — nothing works before `fit_gam`.**
3. **The tests are pure.** They return per-gene DataFrames indexed by gene name and never mutate the AnnData.

## Slot schema (namespace = `key_added`, default `"tradeseq"`)

| Written by `fit_gam` | Slot |
|---|---|
| convergence flag | `adata.var['tradeseq_converged']` |
| GAM coefficients β | `adata.varm['tradeseq_beta']` |
| coefficient covariance Σ | `adata.varm['tradeseq_Sigma']` |
| design / lpmatrix / knots / conditions | `adata.uns['tradeseq']` |

Read inputs: `adata.layers['counts']` (raw), `adata.obsm['pseudotime']`, `adata.obsm['cell_weights']`, `adata.obsm['X_umap']`, `adata.uns['slingshot']['curves']`.

## Invariants & gotchas

- **`fit_gam` before any test.** Missing `adata.uns['tradeseq']` → run `ts.fit_gam(adata, n_knots=6)` first.
- **Counts must be raw, non-negative.** `family='nb'` rejects negative values; don't pass a log/scaled matrix as the counts layer.
- **Pseudotime and weights must be dense and NaN-free**, identical shape `(cells × lineages)`. Cells off a lineage get weight 0, not `NaN` pseudotime.
- **Between-lineage tests need ≥ 2 lineages.** `diff_end_test` / `pattern_test` are meaningless on a single lineage.
- **`condition_test` needs `fit_gam(conditions_key=...)`.** It reads the with-conditions design that only that call writes.
- **`get_smoother_pvalues` / `get_smoother_test_stats` are list-mode only.** They require `fit_gam(..., return_models=True)` and raise `TypeError` on an AnnData.
- **Pick `n_knots` with `evaluate_k`**, then read the elbow in `plot_evaluatek_results`. The R vignette uses 6.
- **Plots are ggplot2_py / pheatmap objects, not matplotlib.** `plot_smoothers` / `plot_gene_count` → `ggplot2_py.GGPlot`; `plot_evaluatek_results` → `patchwork.Patchwork`; `plot_cascade` → `pheatmap.PHeatmap`. Render/save them through the Bio-Babel graphics stack.

## When NOT to reach for tradeSeq

- **No trajectory.** tradeSeq tests expression *along a trajectory*; if cells aren't ordered, infer a trajectory first (or use ordinary cluster DE).
- **Cluster-level marker genes only.** Use a standard differential-expression test, not a per-gene NB-GAM.
- **Compositional / proportion shifts.** Out of scope.

## Quick reference

```python
import tradeseq as ts

adata = ts.load_paul15()                       # counts + 2-lineage trajectory

aic = ts.evaluate_k(adata, k_range=range(3, 8), n_genes=200, plot=False)
ts.fit_gam(adata, n_knots=6)                   # writes the tradeseq slots

asso  = ts.association_test(adata)             # changes along pseudotime?
sve   = ts.start_vs_end_test(adata)            # progenitor vs differentiated
end   = ts.diff_end_test(adata)                # between-lineage endpoints
pat   = ts.pattern_test(adata)                 # between-lineage patterns

top = sve["waldStat"].idxmax()
ts.plot_smoothers(adata, gene=top)
ts.plot_gene_count(adata, gene=top)
```

For one symbol's full signature: `biobabel.describe_symbol(symbol_id="tradeseq.<name>")`.
For the end-to-end workflow as a contract: `biobabel.describe_workflow(workflow_id="tradeseq.trajectory_de")` (and `tradeseq.list_mode_smoothers` for the `return_models=True` path).
