"""Public API: :func:`predict_smooth` (R source: tradeSeq/R/predictSmooth.R).

Port of the AnnData and list-mode dispatchers plus the two private helpers:

* ``.predictSmooth`` (predictSmooth.R:5-55): AnnData branch, no conditions.
* ``.predictSmooth_conditions`` (predictSmooth.R:58-113): AnnData branch, with
  conditions.
* R's fitted-container method (predictSmooth.R:150-198), mapped to AnnData.
* ``setMethod("predictSmooth", "list")`` (predictSmooth.R:202-237).

For the AnnData path we read ``dm``, ``X``, ``beta``,
pseudotime and (optionally) ``conditions`` from the same slots that
:func:`tradeseq.fit_gam` populates.
"""

from __future__ import annotations

import re
from typing import Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd

from .fit_gam import FittedGam
from ._design import get_predict_range_df
from ._predict import predict_gam
from ._slot_io import (
    read_beta,
    read_conditions,
    read_design_matrix,
    read_lpmatrix,
)
from ._test_common import lpmatrix_dataframe, pseudotime_matrix

__all__ = ["predict_smooth"]


_TIME_RE = re.compile(r"t[1-9]")


def _first_valid_fitted_gam(
    models: dict[str, FittedGam | None], *, context: str
) -> FittedGam:
    """Return the first successful list-mode fit, mirroring R's reference scan."""
    if not models:
        raise ValueError(f"{context}: empty models dict.")
    for fg in models.values():
        if fg is not None:
            return fg
    raise ValueError(f"{context}: all models are failed fits.")


def _n_curves_from_dm(dm: pd.DataFrame) -> int:
    """Count distinct ``tK`` columns in the design matrix (R: grep('t[1-9]'))."""
    return sum(1 for c in dm.columns if _TIME_RE.search(c) is not None)


def _gene_index_adata(
    adata: ad.AnnData, gene: Union[str, int, Sequence[Union[str, int]]]
) -> tuple[np.ndarray, list]:
    """Resolve ``gene`` to ``(id_arr, gene_label_list)`` for the AnnData path.

    Mirrors ``predictSmooth.R:157-162``.
    """
    gene_arr = np.atleast_1d(np.asarray(gene))
    if gene_arr.dtype.kind in "UOS":
        var_names = list(adata.var_names)
        missing = [g for g in gene_arr if g not in var_names]
        if missing:
            raise ValueError("Not all gene IDs are present in the models object.")
        id_arr = np.array([var_names.index(g) for g in gene_arr], dtype=np.int64)
        gene_list = [str(g) for g in gene_arr]
    else:
        id_arr = gene_arr.astype(np.int64)
        # In R, when integer indices are passed, beta rows still carry the gene
        # NAME; downstream rownames(beta) <- gene re-labels to the integer.
        # We mirror the integer-as-label behaviour because that is what R does
        # at the matrix level (rownames written into the CSV are the integers).
        gene_list = [int(g) for g in gene_arr]
    return id_arr, gene_list


def _predict_smooth_no_cond(
    dm: pd.DataFrame,
    X: pd.DataFrame,
    beta: np.ndarray,
    pseudotime: np.ndarray,
    gene_labels: Sequence,
    n_points: int,
    tidy: bool,
) -> Union[pd.DataFrame, np.ndarray]:
    """Port of ``.predictSmooth`` (predictSmooth.R:5-55).

    ``beta`` is a ``(n_gene, p)`` matrix already restricted to the requested
    genes. ``gene_labels`` provides the rownames the R code carries through.
    """
    n_curves = _n_curves_from_dm(dm)
    # Build the stacked prediction grid Xall and the tidy index frame outAll.
    x_blocks = []
    tidy_blocks: list[pd.DataFrame] = []
    last_df: pd.DataFrame | None = None
    for jj in range(1, n_curves + 1):
        df = get_predict_range_df(dm, jj, n_points=n_points)
        xdf = predict_gam(lpmatrix=X, df=df, pseudotime=pseudotime)
        x_blocks.append(xdf.to_numpy())
        last_df = df
        if tidy:
            tidy_blocks.append(
                pd.DataFrame(
                    {
                        "lineage": np.full(n_points, jj, dtype=np.int64),
                        "time": df[f"t{jj}"].to_numpy(dtype=float),
                    }
                )
            )
    Xall = np.concatenate(x_blocks, axis=0)  # (n_curves * n_points, p)
    n_gene = len(gene_labels)
    yhat_mat = np.full((n_gene, n_curves * n_points), np.nan, dtype=float)
    # R: c(exp(t(Xall %*% t(beta[gene, , drop=FALSE])) + df$offset[1]))
    # Use the last df's offset (R uses df$offset[1] — only the first row, and
    # because all rows are filled with the same mean offset, any row works).
    offset_scalar = float(last_df["offset"].iloc[0])
    for jj in range(n_gene):
        b = beta[jj]
        # Skip rows where β is all NaN — R returns NaN automatically; mirror that.
        if np.isnan(b).any():
            yhat_mat[jj, :] = np.nan
            continue
        eta = Xall @ b + offset_scalar
        yhat_mat[jj, :] = np.exp(eta)

    # Build R-style column names: paste0("lineage", J, "_", K) per (lineage, point).
    point_names = [f"lineage{jj}_{kk}"
                   for jj in range(1, n_curves + 1)
                   for kk in range(1, n_points + 1)]
    if not tidy:
        return yhat_mat, point_names, gene_labels

    out_all = pd.concat(tidy_blocks, axis=0, ignore_index=True)
    out_rows: list[pd.DataFrame] = []
    for gg, g in enumerate(gene_labels):
        cur = out_all.copy()
        cur["gene"] = g
        cur["yhat"] = yhat_mat[gg, :]
        out_rows.append(cur)
    return pd.concat(out_rows, axis=0, ignore_index=True)


def _predict_smooth_with_cond(
    dm: pd.DataFrame,
    X: pd.DataFrame,
    beta: np.ndarray,
    pseudotime: np.ndarray,
    gene_labels: Sequence,
    n_points: int,
    conditions: pd.Categorical,
    tidy: bool,
) -> Union[pd.DataFrame, np.ndarray]:
    """Port of ``.predictSmooth_conditions`` (predictSmooth.R:58-113)."""
    n_curves = _n_curves_from_dm(dm)
    cond_levels = list(conditions.categories)
    n_conditions = len(cond_levels)
    x_blocks: list[np.ndarray] = []
    tidy_blocks: list[pd.DataFrame] = []
    last_df: pd.DataFrame | None = None
    for jj in range(1, n_curves + 1):
        per_cond: list[np.ndarray] = []
        per_cond_tidy: list[pd.DataFrame] = []
        for kk in range(1, n_conditions + 1):
            df = get_predict_range_df(
                dm, lineage_id=jj, condition_id=kk, n_points=n_points
            )
            xdf = predict_gam(lpmatrix=X, df=df, pseudotime=pseudotime,
                              conditions=conditions)
            per_cond.append(xdf.to_numpy())
            last_df = df
            if tidy:
                per_cond_tidy.append(
                    pd.DataFrame({
                        "lineage": np.full(n_points, jj, dtype=np.int64),
                        "time": df[f"t{jj}"].to_numpy(dtype=float),
                        "condition": [cond_levels[kk - 1]] * n_points,
                    })
                )
        x_blocks.append(np.concatenate(per_cond, axis=0))
        if tidy:
            tidy_blocks.append(pd.concat(per_cond_tidy, axis=0, ignore_index=True))
    Xall = np.concatenate(x_blocks, axis=0)
    n_gene = len(gene_labels)
    yhat_mat = np.full((n_gene, n_curves * n_conditions * n_points), np.nan, dtype=float)
    offset_scalar = float(last_df["offset"].iloc[0])
    for jj in range(n_gene):
        b = beta[jj]
        if np.isnan(b).any():
            yhat_mat[jj, :] = np.nan
            continue
        eta = Xall @ b + offset_scalar
        yhat_mat[jj, :] = np.exp(eta)
    # R column-name construction: for each lineage and condition, n_points point columns.
    col_names = []
    for jj in range(1, n_curves + 1):
        for kk in range(1, n_conditions + 1):
            base = f"lineage{jj}_condition{cond_levels[kk - 1]}"
            for pp in range(1, n_points + 1):
                col_names.append(f"{base}_point{pp}")
    if not tidy:
        return yhat_mat, col_names, gene_labels
    out_all = pd.concat(tidy_blocks, axis=0, ignore_index=True)
    out_rows: list[pd.DataFrame] = []
    for gg, g in enumerate(gene_labels):
        cur = out_all.copy()
        cur["gene"] = g
        cur["yhat"] = yhat_mat[gg, :]
        out_rows.append(cur)
    return pd.concat(out_rows, axis=0, ignore_index=True)


def _predict_smooth_adata(
    adata: ad.AnnData,
    gene: Union[str, int, Sequence[Union[str, int]]],
    n_points: int,
    tidy: bool,
    key: str,
) -> Union[pd.DataFrame, np.ndarray]:
    """AnnData dispatcher mirroring predictSmooth.R:150-198."""
    dm = read_design_matrix(adata, key=key)
    X = lpmatrix_dataframe(adata, key=key)
    beta_full = read_beta(adata, key=key)
    pseudotime = pseudotime_matrix(adata)
    id_arr, gene_labels = _gene_index_adata(adata, gene)
    beta = beta_full[id_arr, :]
    conditions = read_conditions(adata, key=key)
    if conditions is None:
        result = _predict_smooth_no_cond(
            dm=dm, X=X, beta=beta, pseudotime=pseudotime,
            gene_labels=gene_labels, n_points=n_points, tidy=tidy,
        )
    else:
        result = _predict_smooth_with_cond(
            dm=dm, X=X, beta=beta, pseudotime=pseudotime,
            gene_labels=gene_labels, n_points=n_points,
            conditions=conditions, tidy=tidy,
        )
    if tidy:
        return result
    yhat_mat, col_names, gene_labels = result
    return pd.DataFrame(
        yhat_mat, index=[str(g) for g in gene_labels], columns=col_names
    )


def _predict_smooth_list(
    models: dict[str, FittedGam],
    gene: Union[str, int, Sequence[Union[str, int]]],
    n_points: int,
) -> np.ndarray:
    """List dispatcher (predictSmooth.R:202-237).

    R code path: read ``dm <- m$model[, -1]`` from the reference model, build
    a per-lineage prediction range, then evaluate every gene's GLM linear
    predictor on the same grid. The Python equivalent reads ``FittedGam.dm_``
    (added in Slice 3 coordinator pass) and uses the shared ``lpmatrix_`` as
    the per-cell basis for spline reconstruction.
    """
    # All FittedGam entries share the same design matrix and lpmatrix (they
    # come from the same fit_gam call). Pick the first non-None to use.
    ref = _first_valid_fitted_gam(models, context="predict_smooth")
    dm = ref.dm_
    lpmatrix = ref.lpmatrix_
    n_curves = sum(1 for c in dm.columns if c.startswith("t") and c[1:].isdigit())

    # Resolve gene IDs to FittedGam keys.
    keys = list(models.keys())
    if isinstance(gene, (str, int)):
        gene_seq: list = [gene]
    else:
        gene_seq = list(gene)
    selected_keys: list[str] = []
    for g in gene_seq:
        if isinstance(g, str):
            if g not in models:
                raise ValueError(f"Not all gene IDs are present in the models object: {g!r}")
            selected_keys.append(g)
        else:
            idx = int(g)
            if not 0 <= idx < len(keys):
                raise IndexError(f"gene index {idx} out of range (n={len(keys)})")
            selected_keys.append(keys[idx])

    # pseudotime grid for predict_gam reconstruction (one column per lineage).
    pseudotime = np.column_stack(
        [dm[f"t{j+1}"].values for j in range(n_curves)]
    )

    # Build the long-form df by stacking get_predict_range_df outputs.
    parts = []
    for jj in range(1, n_curves + 1):
        df_j = get_predict_range_df(dm, lineage_id=jj, n_points=n_points)
        parts.append(df_j)
    dfall = pd.concat(parts, ignore_index=True)

    Xpred = predict_gam(lpmatrix=lpmatrix, df=dfall, pseudotime=pseudotime)

    # Each fitted gene: yhat = exp(Xpred @ beta + offset_mean)
    n_total = n_curves * n_points
    yhat_mat = np.full((len(selected_keys), n_total), np.nan)
    for i, k in enumerate(selected_keys):
        fg = models[k]
        if fg is None or fg.coef_ is None or np.any(np.isnan(fg.coef_)):
            continue
        beta = np.asarray(fg.coef_, dtype=np.float64)
        offset_col = dfall["offset"].values if "offset" in dfall.columns else np.zeros(n_total)
        # offset is constant per design row in the R path; use it directly.
        yhat_mat[i, :] = np.exp(Xpred.values @ beta + offset_col)

    # R returns a (n_genes, n_curves * n_points) matrix with column names
    # ``lineage{J}_{K}`` (predictSmooth.R:228-230 — no ``_point`` infix).
    col_names = [
        f"lineage{j+1}_{p+1}" for j in range(n_curves) for p in range(n_points)
    ]
    out = pd.DataFrame(yhat_mat, index=selected_keys, columns=col_names)
    return out


def predict_smooth(
    adata: Union[ad.AnnData, dict[str, FittedGam]],
    *,
    gene: Union[str, int, Sequence[Union[str, int]]],
    n_points: int = 100,
    tidy: bool = True,
    key: str = "tradeseq",
) -> Union[pd.DataFrame, np.ndarray]:
    """Predict smoother values on a uniform pseudotime grid.

    Port of ``tradeSeq::predictSmooth`` (R source: ``tradeSeq/R/predictSmooth.R``).

    Parameters
    ----------
    adata : anndata.AnnData or dict[str, FittedGam]
        AnnData populated by :func:`tradeseq.fit_gam` or the list-mode dict
        from ``fit_gam(return_models=True)``.
    gene : str, int, or sequence of either
        Gene identifier(s) to predict.
    n_points : int, default 100
        Number of evaluation points per lineage.
    tidy : bool, default True
        If True, returns a long-form DataFrame with columns
        ``(lineage, time, gene, yhat)``. If False, returns the wide-form
        ``(n_genes, n_curves * n_points)`` numpy matrix. ``tidy`` is ignored
        in list-mode (which always returns the wide matrix, per R).
    key : str, default ``"tradeseq"``
        Namespace prefix used by :func:`tradeseq.fit_gam`.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Long-form DataFrame when ``tidy=True`` (AnnData path), wide-form matrix
        otherwise.
    """
    if isinstance(adata, ad.AnnData):
        return _predict_smooth_adata(
            adata, gene, n_points=n_points, tidy=tidy, key=key
        )
    if isinstance(adata, dict):
        return _predict_smooth_list(adata, gene, n_points=n_points)
    raise TypeError(
        f"predict_smooth: unsupported models type {type(adata).__name__!r}; "
        "expected anndata.AnnData or dict[str, FittedGam]."
    )
