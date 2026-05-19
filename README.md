# tradeSeq-python

[![PyPI](https://img.shields.io/pypi/v/tradeSeq-python)](https://pypi.org/project/tradeSeq-python/)

Python [**tradeSeq**](https://github.com/statOmics/tradeSeq) package — trajectory-based differential expression analysis with negative-binomial generalized additive models (NB-GAM).


## Installation

```bash
pip install tradeSeq-python                # from PyPI
```

## Quickstart

```python
import tradeseq as ts

adata = ts.load_paul15()                              # AnnData (2660 × 240)

ts.fit_gam(adata, n_knots=6)                          # NB-GAM per gene

assoc = ts.association_test(adata)                    # expression vs pseudotime
start = ts.start_vs_end_test(adata)                   # progenitor markers
diff  = ts.diff_end_test(adata)                       # between-lineage endpoints
patt  = ts.pattern_test(adata)                        # between-lineage patterns
early = ts.early_de_test(adata, knots=(1, 2))         # early drivers

ts.plot_smoothers(adata, gene=start['waldStat'].idxmax())
ts.plot_gene_count(adata, gene=start['waldStat'].idxmax())
```

Choose the number of knots with `evaluate_k` / `evaluate_k2` and `plot_evaluatek_results`. `evaluate_k2` is a speed-up version of the original `evaluate_k`.

## Tutorials

Runnable notebooks that reproduce the R tradeSeq vignettes live under [`tutorials/`](tutorials/):

| Notebook | Coverage |
|---|---|
| `tradeSeq.ipynb` | Full Wald battery on the Paul-2015 myeloid trajectory — mirrors `vignettes/tradeSeq.Rmd` |
| `fitGAM.ipynb`   | Model-fitting options — covariates, parallelism, list-mode output, convergence diagnostics |

