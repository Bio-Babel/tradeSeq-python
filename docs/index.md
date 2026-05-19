# tradeSeq-python

Python port of the R/Bioconductor package **tradeSeq** (1.13.12) —
*trajectory-based differential expression analysis for single-cell sequencing
data via negative-binomial generalized additive models (NB-GAM)*.

`tradeSeq` fits a per-gene NB-GAM along one or more trajectory lineages and
exposes a battery of Wald-style statistical tests for differential expression
within and between lineages. This port follows the scverse convention: the
container is [`anndata.AnnData`](https://anndata.readthedocs.io); every R
accessor (`slingPseudotime`, `slingCurveWeights`, …) collapses to direct
AnnData slot reads.

## Installation

```bash
pip install tradeSeq-python
```

The package bundles a 1.7 MB AnnData fixture (Paul-2015 myeloid progenitors,
2660 cells × 240 genes, 2 lineages) under
`tradeseq/resources/paul15_tradeseq.h5ad`. No trajectory-inference library is
required at runtime — pseudotimes and cell weights were computed once on the
R side with slingshot 2.8.0 and frozen into the fixture.

## Quick start

```python
import tradeseq as ts

adata = ts.load_paul15()

# Fit the NB-GAM on every gene along each lineage.
ts.fit_gam(adata, n_knots=6)

# Test for association of expression with pseudotime within each lineage.
asso = ts.association_test(adata)
asso.head()

# Discover progenitor → differentiated cell-type markers.
sve = ts.start_vs_end_test(adata)
top_gene = sve.sort_values("waldStat", ascending=False).index[0]
ts.plot_smoothers(adata, gene=top_gene)
ts.plot_gene_count(adata, gene=top_gene)
```

See [Tutorials](tutorials/tradeSeq.ipynb) for the full workflow walkthrough.

## AnnData slot schema

`fit_gam` mutates `adata` in place. The R `colData(sce)$tradeSeq$*`,
`rowData(sce)$tradeSeq$*`, and `metadata(sce)$tradeSeq$*` payloads map to:

| R slot | AnnData slot |
|---|---|
| `counts(sce)` | `adata.layers["counts"]` |
| `rowData(sce)$tradeSeq$beta` | `adata.varm["tradeseq_beta"]` |
| `rowData(sce)$tradeSeq$Sigma` | `adata.varm["tradeseq_Sigma"]` |
| `rowData(sce)$tradeSeq$converged` | `adata.var["tradeseq_converged"]` |
| `colData(sce)$tradeSeq$dm` | `adata.uns["tradeseq"]["design_matrix"]` |
| `colData(sce)$tradeSeq$X` | `adata.uns["tradeseq"]["lpmatrix"]` |
| `colData(sce)$tradeSeq$conditions` | `adata.uns["tradeseq"]["conditions"]` |
| `metadata(sce)$tradeSeq$knots` | `adata.uns["tradeseq"]["knots"]` |

The namespace prefix (default `"tradeseq"`) can be changed via the `key_added`
argument of `fit_gam` for parallel-fit workflows.

## API surface

18 R-mapped working exports, plus Python-only `load_paul15` and
`plot_cascade` helpers:

| Tier | Exports |
|---|---|
| T0 foundation | `fit_gam`, `nknots` |
| T1 diagnostic | `evaluate_k`, `plot_evaluatek_results` |
| T2 Wald battery | `association_test`, `start_vs_end_test`, `diff_end_test`, `pattern_test`, `early_de_test`, `condition_test` |
| T3 prediction | `predict_cells`, `predict_smooth` |
| T4 plotting | `plot_smoothers`, `plot_gene_count` |
| T5 downstream | `cluster_expression_patterns`, `cascade`, `plot_cascade` |
| T6 list-mode | `get_smoother_pvalues`, `get_smoother_test_stats` |
| Python containers | `FittedGam`, `CascadeResult` |

See [API Reference](api.md) for full signatures and docstrings.

## Implementation notes

- The NB-GAM kernel is implemented in pure Python: `statsmodels`-derived
  penalized IRLS plus a custom REML/Laplace λ score and an outer α-MLE loop.
  This mirrors mgcv's `family="nb"` extended-family behavior, where the
  nominal `GCV.Cp` default is coerced to REML while the returned field remains
  named `gcv.ubre`.
- The cubic-regression-spline basis (`bs='cr'`) is a direct port of
  mgcv-1.9.3's `src/mgcv.c:crspl` — bit-exact to within 1e-10.
- TMM normalisation (`edgeR::calcNormFactors`) is reimplemented in NumPy.
- Visualisation uses the Bio-Babel ecosystem (`ggplot2_py`, `patchwork`,
  `pheatmap`, `scales`, `grid_py`, `gtable_py`) — no matplotlib.
- `cluster_expression_patterns` swaps `clusterExperiment::RSEC` for
  `sklearn.cluster.AgglomerativeClustering` (documented Tier-3 deviation).

For full porting notes, see the `port_reports/tradeSeq/` directory in the
source repository.

## License

MIT. Original R package © Koen Van den Berge, Hector Roux de Bézieux, Kelly
Street, Lieven Clement, Sandrine Dudoit.
