# API Reference

Public surface for `tradeseq`. Every function reads or writes
[`anndata.AnnData`](https://anndata.readthedocs.io) per the slot schema
documented on the [Home](index.md) page.

## Datasets

::: tradeseq.datasets

## Model fitting

::: tradeseq.fit_gam

## Diagnostic

Includes the R-faithful `evaluate_k`, the optimized `evaluate_k2`, and
`plot_evaluatek_results`.

::: tradeseq.evaluate_k

## Wald tests

### `association_test`
::: tradeseq.association_test

### `start_vs_end_test`
::: tradeseq.start_vs_end_test

### `diff_end_test`
::: tradeseq.diff_end_test

### `pattern_test`
::: tradeseq.pattern_test

### `condition_test`
::: tradeseq.condition_test

## Prediction

### `predict_cells`
::: tradeseq.predict_cells

### `predict_smooth`
::: tradeseq.predict_smooth

## Visualization

### `plot_smoothers`
::: tradeseq.plot_smoothers

### `plot_gene_count`
::: tradeseq.plot_gene_count

## Downstream

### `cluster_expression_patterns`
::: tradeseq.cluster

### `cascade` / `plot_cascade`
::: tradeseq.cascade

## List-mode accessors

### `get_smoother_pvalues` / `get_smoother_test_stats`
::: tradeseq.get_smoother
