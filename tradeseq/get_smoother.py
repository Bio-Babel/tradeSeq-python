"""Public API: :func:`get_smoother_pvalues` and :func:`get_smoother_test_stats`.

R sources:

* ``tradeSeq/R/getSmootherPvalues.R`` — reads ``summary(m)$s.table[, "p-value"]``
  per gene.
* ``tradeSeq/R/getSmootherTestStats.R`` — reads ``summary(m)$s.table[, "Chi.sq"]``
  per gene.

Both functions take the fitted-model output of :func:`tradeseq.fit_gam`
(``return_models=True``) — a ``dict[str, FittedGam]`` — and concatenate the
per-gene summary tables into a single ``(n_genes, n_smoothers)`` matrix.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from .fit_gam import FittedGam

__all__ = ["get_smoother_pvalues", "get_smoother_test_stats"]


_LIST_ONLY_MESSAGE = (
    "{name} only operates on a dict[str, FittedGam] (the list-mode output of "
    "fit_gam). Call fit_gam(adata, ..., return_models=True) to obtain that "
    "dict, then pass it here."
)


def _validate_list_models(
    models: object, *, name: str
) -> dict[str, FittedGam]:
    """Reject AnnData input and ensure we have a ``dict`` of models.

    Mirrors the R type-method dispatch contract of
    ``getSmootherPvalues`` / ``getSmootherTestStats`` which only define a
    function for list inputs.
    """
    if isinstance(models, ad.AnnData):
        raise TypeError(_LIST_ONLY_MESSAGE.format(name=name))
    if not isinstance(models, dict):
        raise TypeError(
            f"{name}: expected dict[str, FittedGam]; got {type(models).__name__!r}."
        )
    return models


def _get_reference(models: dict[str, FittedGam]) -> FittedGam:
    """Return the first non-None FittedGam in ``models`` (R: ``.getModelReference``)."""
    for fg in models.values():
        if fg is not None:
            return fg
    raise ValueError("All models errored")


def _stack_summary_column(
    models: dict[str, FittedGam],
    column: str,
) -> pd.DataFrame:
    """Stack the per-gene ``summary_s_table[column]`` rows into a DataFrame.

    Genes with a missing fit (``FittedGam is None``) get a row of NaNs,
    mirroring the R ``try-error`` branch:

    .. code-block:: R

        if (is(m)[1] == "try-error") return(rep(NA, nCurves))

    Note that R only NA-fills on ``try-error`` — convergence warnings during
    a successful fit do not trigger NA. Our list-mode contract therefore
    matches NA on ``fg is None`` only; the ``converged_`` field is informational
    and does not propagate into the output (the R summary table is computed
    irrespective of convergence-warning state).
    """
    ref = _get_reference(models)
    n_curves = ref.summary_s_table.shape[0]
    col_names = list(ref.summary_s_table.index)
    rows: list[np.ndarray] = []
    gene_names: list[str] = []
    for name, fg in models.items():
        gene_names.append(name)
        if fg is None:
            rows.append(np.full(n_curves, np.nan, dtype=float))
            continue
        stab = fg.summary_s_table
        if column not in stab.columns:
            raise KeyError(
                f"FittedGam.summary_s_table for gene {name!r} is missing "
                f"column {column!r}; expected one of {list(stab.columns)}."
            )
        rows.append(np.asarray(stab[column].values, dtype=float))
    return pd.DataFrame(np.vstack(rows), index=gene_names, columns=col_names)


def get_smoother_pvalues(
    models: dict[str, FittedGam],
) -> pd.DataFrame:
    """Return per-smoother p-values across genes.

    Port of ``tradeSeq::getSmootherPvalues`` (R source:
    ``tradeSeq/R/getSmootherPvalues.R``).

    Parameters
    ----------
    models : dict[str, FittedGam]
        Fitted-model output of :func:`tradeseq.fit_gam` with
        ``return_models=True``.

    Returns
    -------
    pandas.DataFrame
        ``(n_genes, n_smoothers)`` table indexed by gene name. Each cell is
        the smoother-specific Wald-test p-value reported by the
        per-gene GAM summary (R: ``summary(m)$s.table[, "p-value"]``).
        Genes whose fits errored receive a row of NaNs.

    Raises
    ------
    TypeError
        If ``models`` is an :class:`anndata.AnnData`. The R contract is
        list-only.
    """
    models_d = _validate_list_models(models, name="get_smoother_pvalues")
    return _stack_summary_column(models_d, "p-value")


def get_smoother_test_stats(
    models: dict[str, FittedGam],
) -> pd.DataFrame:
    """Return per-smoother Chi-square test statistics across genes.

    Port of ``tradeSeq::getSmootherTestStats`` (R source:
    ``tradeSeq/R/getSmootherTestStats.R``).

    Parameters
    ----------
    models : dict[str, FittedGam]
        Fitted-model output of :func:`tradeseq.fit_gam` with
        ``return_models=True``.

    Returns
    -------
    pandas.DataFrame
        ``(n_genes, n_smoothers)`` table indexed by gene name. Each cell is
        the smoother-specific Wald-test Chi-square statistic from the per-gene
        GAM summary (R: ``summary(m)$s.table[, "Chi.sq"]``). Genes whose fits
        errored receive a row of NaNs.

    Raises
    ------
    TypeError
        If ``models`` is an :class:`anndata.AnnData`. The R contract is
        list-only.
    """
    models_d = _validate_list_models(models, name="get_smoother_test_stats")
    return _stack_summary_column(models_d, "Chi.sq")
