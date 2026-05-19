"""Public API: :func:`predict_cells` (R source: tradeSeq/R/predictCells.R).

Port of the two ``setMethod`` dispatchers for ``predictCells``:

* AnnData dispatcher (mirroring ``predictCells.R:20-43``): pulls ``dm``, ``X``,
  and ``beta`` from the AnnData container previously populated by
  :func:`tradeseq.fit_gam` and returns ``exp(X @ beta.T + offset)``.
* List dispatcher (``predictCells.R:47-63``): unpacks the ``fitted.values``
  field per gene from a ``dict[str, FittedGam]`` produced by
  ``fit_gam(return_models=True)``.
"""

from __future__ import annotations

from typing import Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd

from .fit_gam import FittedGam
from ._slot_io import read_beta, read_design_matrix, read_lpmatrix

__all__ = ["predict_cells"]


def _resolve_gene_ids_adata(
    adata: ad.AnnData, gene: Union[str, int, Sequence[Union[str, int]]]
) -> tuple[np.ndarray, list]:
    """Return ``(id, gene_list)`` where ``id`` is the integer row index in adata.

    Mirrors ``predictCells.R:25-30``:

    .. code-block:: R

        if (is(gene, "character")) {
          if (!all(gene %in% rownames(models))) stop(...)
          id <- match(gene, rownames(models))
        } else id <- gene
    """
    gene_arr = np.atleast_1d(np.asarray(gene))
    if gene_arr.dtype.kind in "UOS":
        # Character input.
        var_names = list(adata.var_names)
        missing = [g for g in gene_arr if g not in var_names]
        if missing:
            raise ValueError("Not all gene IDs are present in the models object.")
        id_arr = np.array([var_names.index(g) for g in gene_arr], dtype=np.int64)
        gene_list = [str(g) for g in gene_arr]
    else:
        # Integer input — R uses 1-based; the Python container is 0-based, but
        # the public Python API contract here is that integer indices are
        # already 0-based positions in adata.var_names (consistent with
        # tradeseq.fit_gam(genes=range(...))). The returned rownames use the
        # gene NAMES rather than the numeric input to match the R behaviour
        # (rownames(yhat) <- gene where gene can be integer — in R this still
        # uses the integer-as-string as label; mirror exactly).
        id_arr = gene_arr.astype(np.int64)
        gene_list = [int(g) for g in gene_arr]
    return id_arr, gene_list


def _resolve_offset_column(dm: pd.DataFrame) -> str:
    """Return the dm column name carrying the per-cell offset.

    In R the column is named ``offset(<name>)`` and accessed via ``dm$offset``
    (partial-match). Mirror that lookup by finding the unique column whose
    name starts with ``offset``.
    """
    candidates = [c for c in dm.columns if c.startswith("offset")]
    if len(candidates) == 1:
        return candidates[0]
    if "offset" in dm.columns:
        return "offset"
    raise KeyError(
        f"design matrix has no offset column; got {list(dm.columns)}."
    )


def _predict_cells_adata(
    adata: ad.AnnData,
    gene: Union[str, int, Sequence[Union[str, int]]],
    key: str,
) -> np.ndarray:
    """AnnData branch of ``predictCells`` (mirrors ``predictCells.R:20-43``).

    Computes ``yhat = exp(X %*% t(beta) + offset)`` where ``offset`` is replicated
    to a ``(n_genes, n_cells)`` matrix (``byrow=TRUE`` in R).
    """
    dm = read_design_matrix(adata, key=key)
    X = read_lpmatrix(adata, key=key)
    beta_full = read_beta(adata, key=key)
    id_arr, gene_list = _resolve_gene_ids_adata(adata, gene)
    beta = beta_full[id_arr, :]
    # R: exp(t(X %*% t(beta)) + matrix(dm$offset, nrow=length(gene), ncol=nrow(X), byrow=TRUE))
    # X is (n_cells, p); beta is (n_genes, p); X @ beta.T is (n_cells, n_genes);
    # transpose -> (n_genes, n_cells); add offset replicated across genes.
    offset_col = _resolve_offset_column(dm)
    offset_vec = np.asarray(dm[offset_col].values, dtype=float)
    eta = (X @ beta.T).T + offset_vec[None, :]
    yhat = np.exp(eta)
    return yhat


def _predict_cells_list(
    models: dict[str, FittedGam],
    gene: Union[str, int, Sequence[Union[str, int]]],
) -> np.ndarray:
    """List branch of ``predictCells`` (``predictCells.R:47-63``).

    R code path:

    .. code-block:: R

        if (is(gene, "character")) {
          if (!all(gene %in% rownames(models))) stop(...)
          id <- which(rownames(models) %in% gene)
        } else id <- gene
        yhat <- t(sapply(models[id], "[[", "fitted.values"))

    Note ``rownames(models)`` is ``NULL`` for a plain list in R, so the character
    branch always errors in vanilla R. We honour that contract.
    """
    gene_arr = np.atleast_1d(np.asarray(gene))
    keys = list(models.keys())
    if gene_arr.dtype.kind in "UOS":
        # Faithful to R buggy path: rownames(list) is NULL, so this always fails.
        # We provide a clearer error pointing to the underlying issue.
        raise ValueError("The gene ID is not present in the models object.")
    # Integer input: R uses 1-based; Python integer indices are 0-based.
    # The R label written by `rownames(yhat) <- gene` is the integer itself
    # coerced to character, so we mirror that.
    id_arr = gene_arr.astype(np.int64)
    yhat_rows = []
    for i in id_arr:
        fg = models[keys[int(i)]]
        # R: ``gam$fitted.values = exp(X·β + offset)`` for log-link NB.
        # The offset lives on each FittedGam at ``dm_['offset(offset)']``
        # (carried so list-mode consumers don't need the wrapping AnnData).
        offset_col = _resolve_offset_column(fg.dm_)
        offset_vec = np.asarray(fg.dm_[offset_col].values, dtype=float)
        fv = np.exp(fg.lpmatrix_.values @ fg.coef_ + offset_vec)
        yhat_rows.append(fv)
    yhat = np.vstack(yhat_rows)
    return yhat


def predict_cells(
    adata: Union[ad.AnnData, dict[str, FittedGam]],
    *,
    gene: Union[str, int, Sequence[Union[str, int]]],
    key: str = "tradeseq",
) -> np.ndarray:
    """Return per-cell fitted expression values for the requested gene(s).

    Port of ``tradeSeq::predictCells`` (R source: ``tradeSeq/R/predictCells.R``).
    Two dispatch paths mirror the two R ``setMethod`` definitions:

    * ``adata`` is an :class:`anndata.AnnData` produced by
      :func:`tradeseq.fit_gam` — returns ``exp(X @ beta.T + offset)``.
    * ``adata`` is a ``dict[str, FittedGam]`` produced by
      ``tradeseq.fit_gam(return_models=True)`` — returns the per-gene
      ``exp(X @ coef_ + offset)`` rows (R's ``fitted.values``).

    Parameters
    ----------
    adata : anndata.AnnData or dict[str, FittedGam]
        Either AnnData populated by :func:`tradeseq.fit_gam` or the list-mode dict.
    gene : str, int, or sequence of either
        Gene identifier(s). Character inputs must match ``adata.var_names``
        (AnnData path) or fail (list path — mirrors the R bug where
        ``rownames(list)`` is ``NULL``). Integer inputs are 0-based positions.
    key : str, default ``"tradeseq"``
        Namespace prefix in ``adata.uns`` / ``adata.varm`` / ``adata.var``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_genes_requested, n_cells)`` matrix of fitted values.

    Raises
    ------
    ValueError
        If a character gene id is not in the container.
    TypeError
        If ``adata`` is neither an AnnData nor a dict.
    """
    if isinstance(adata, ad.AnnData):
        return _predict_cells_adata(adata, gene, key=key)
    if isinstance(adata, dict):
        return _predict_cells_list(adata, gene)
    raise TypeError(
        f"predict_cells: unsupported models type {type(adata).__name__!r}; "
        "expected anndata.AnnData or dict[str, FittedGam]."
    )
