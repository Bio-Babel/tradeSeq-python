"""Public API: :func:`fit_gam` and :func:`nknots` (R source: tradeSeq/R/fitGAM.R + nknots.R)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2 as _chi2
from tqdm.auto import tqdm

from ._assign_cells import assign_cells
from ._design_builder import (
    DesignMatrices,
    build_smooth_design,
    build_smooth_design_with_conditions,
)
from ._gam import GamFit, fit_nb_gam_block
from ._knots import find_knots
from ._offset import compute_offset
from ._slot_io import write_fit_results

__all__ = ["fit_gam", "nknots", "FittedGam"]

_MGCV_TESTSTAT_EIG_TOL = np.finfo(float).eps ** 0.9


@dataclass
class FittedGam:
    """Thin per-gene container returned when ``return_models=True``.

    Mirrors only the mgcv ``gam`` attributes that the tradeSeq exports
    actually consume — see
    ``tradeseq_porting_essential_suggestions.md`` §1 last paragraph.
    """

    coef_: np.ndarray
    """Coefficient vector β, shape ``(n_coefs,)``."""

    cov_: np.ndarray
    """Bayesian posterior covariance ``Vp``, shape ``(n_coefs, n_coefs)``."""

    lpmatrix_: pd.DataFrame
    """Linear-predictor model matrix at the training cells."""

    dm_: pd.DataFrame
    """Per-cell long-form design matrix (R parity: ``m$model[, -1]``).

    Columns ``U, t1..tL, l1..lL, offset(offset)`` mirror mgcv's
    ``gam$model``. Carried so that list-mode :func:`predict_smooth` and
    other list-mode consumers can build prediction grids without needing
    the wrapping AnnData.
    """

    summary_s_table: pd.DataFrame
    """One-row-per-smoother summary with columns ``edf, Ref.df, Chi.sq, p-value``."""

    alpha_: float
    """Estimated NB dispersion."""

    converged_: bool
    """True iff outer α loop converged."""


def _resolve_input_arrays(
    adata: ad.AnnData,
    layer: str,
    pseudotime_key: str,
    weights_key: str,
    conditions_key: Optional[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[pd.Categorical]]:
    """Extract counts/pseudotime/cell_weights/conditions from the AnnData."""
    if layer not in adata.layers:
        raise KeyError(
            f"adata.layers has no key {layer!r}; expected the raw count layer "
            f"used by fit_gam"
        )
    counts = adata.layers[layer]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = np.asarray(counts).astype(np.float64)  # n_cells × n_genes
    if pseudotime_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm has no key {pseudotime_key!r}; populate it before fit_gam"
        )
    pseudotime = np.asarray(adata.obsm[pseudotime_key], dtype=np.float64)
    if pseudotime.ndim == 1:
        pseudotime = pseudotime[:, None]
    if weights_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm has no key {weights_key!r}; populate it before fit_gam"
        )
    cell_weights = np.asarray(adata.obsm[weights_key], dtype=np.float64)
    if cell_weights.ndim == 1:
        cell_weights = cell_weights[:, None]
    conditions: Optional[pd.Categorical] = None
    if conditions_key is not None:
        if conditions_key not in adata.obs.columns:
            raise KeyError(f"adata.obs has no column {conditions_key!r}")
        conditions = pd.Categorical(adata.obs[conditions_key].values)
        if len(conditions.categories) == 1:
            warnings.warn(
                "Only one condition level provided; running fit_gam without conditions",
                stacklevel=3,
            )
            conditions = None
    return counts, pseudotime, cell_weights, conditions


def _gene_indices(
    adata: ad.AnnData, genes: Optional[Union[np.ndarray, list, range]]
) -> np.ndarray:
    if genes is None:
        return np.arange(adata.n_vars, dtype=np.int64)
    arr = np.asarray(list(genes)) if isinstance(genes, range) else np.asarray(genes)
    if arr.dtype.kind in "UOS":
        if len(set(arr)) != len(arr):
            raise ValueError("The genes vector contains duplicates.")
        missing = [g for g in arr if g not in adata.var_names]
        if missing:
            raise ValueError(
                f"The genes ID is not present in the models object: {missing[:5]}"
            )
        idx = np.array([adata.var_names.get_loc(g) for g in arr], dtype=np.int64)
        return idx
    return arr.astype(np.int64)


def _validate_pseudotime(pseudotime: np.ndarray, cell_weights: np.ndarray) -> None:
    if np.isnan(pseudotime).any():
        raise ValueError("The pseudotimes contain NA values, and these cannot be used for GAM fitting.")
    if np.isnan(cell_weights).any():
        raise ValueError("The cellWeights contain NA values, and these cannot be used for GAM fitting.")
    if pseudotime.shape != cell_weights.shape:
        raise ValueError("pseudotime and cellWeights must have identical dimensions.")


def _smoother_labels(design: DesignMatrices) -> list[str]:
    """Return R-faithful smoother labels (``s(t1):l1``, ``s(t1):l1_1``, etc).

    Mirrors mgcv's ``object$smooth[[i]]$label``. Each block in
    ``design.smooth_blocks`` corresponds to one smoother — its column names
    are e.g. ``s(t1):l1.1, s(t1):l1.2, ...`` (no conditions) or
    ``s(t1):l1_1.1, s(t1):l1_1.2, ...`` (with conditions). The label is the
    column-name prefix (everything before the final ``.k``).
    """
    cols = list(design.lpmatrix.columns)
    labels: list[str] = []
    for start, _stop in design.smooth_blocks:
        col = cols[start]
        # Strip the trailing ``.k`` (basis index, 1-based) off the column name.
        # All columns in a block share the same prefix.
        dot = col.rfind(".")
        labels.append(col[:dot] if dot != -1 else col)
    return labels


def _mgcv_test_stat(
    p: np.ndarray,
    X: np.ndarray,
    V: np.ndarray,
    rank: float,
) -> tuple[float, float, float]:
    """Port the type-0 branch of ``mgcv:::testStat`` used by ``summary.gam``.

    mgcv uses a fractional reference rank (`edf1`) and rotates the smoother
    coefficient covariance through the QR factor of the smooth-specific design
    block before computing the Wald statistic. The exact fractional-rank tail
    probability in mgcv calls Davies' algorithm for weighted chi-square sums;
    Python uses the same fallback branch that mgcv itself uses when the
    weighted-sum calculation is unavailable: a chi-square upper tail at the
    fractional rank. This keeps the statistic and rank construction source
    faithful while avoiding a compiled Davies dependency.
    """
    p = np.asarray(p, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if p.size == 0 or X.size == 0 or V.size == 0:
        return np.nan, np.nan, np.nan

    # R: qrx <- qr(X, tol = 0); R <- qr.R(qrx). For the full-rank CR basis
    # blocks used by tradeSeq this leaves the pivot as identity, so the
    # unpivoted reduced QR is the faithful path.
    _Q, R = np.linalg.qr(X, mode="reduced")
    VR = R @ V @ R.T
    VR = 0.5 * (VR + VR.T)

    eigvals, eigvecs = np.linalg.eigh(VR)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if eigvals.size == 0 or eigvals[0] <= 0.0 or not np.isfinite(eigvals[0]):
        return np.nan, np.nan, np.nan

    # Match mgcv's deterministic eigenvector sign convention.
    signs = np.sign(eigvecs[0, :])
    signs[signs == 0.0] = 1.0
    eigvecs = eigvecs * signs

    k = max(0, int(np.floor(rank)))
    nu = abs(float(rank) - k)
    k1 = k + 1 if nu > 0.0 else k
    r_est = int(np.sum(eigvals > eigvals[0] * _MGCV_TESTSTAT_EIG_TOL))
    if r_est < k1:
        k1 = k = r_est
        nu = 0.0
        rank = float(r_est)
    if k1 <= 0:
        return np.nan, np.nan, np.nan

    vec = eigvecs[:, :k1].copy()
    vec1 = vec.copy()
    if nu > 0.0 and k > 0:
        if k > 1:
            vec[:, : k - 1] = vec[:, : k - 1] / np.sqrt(eigvals[: k - 1])
        b12 = float(np.sqrt(max(0.0, 0.5 * nu * (1.0 - nu))))
        B = np.array([[1.0, b12], [b12, nu]], dtype=np.float64)
        ev = np.diag(eigvals[k - 1 : k1] ** -0.5)
        B = ev @ B @ ev
        eb_vals, eb_vecs = np.linalg.eigh(0.5 * (B + B.T))
        if np.any(eb_vals < -np.finfo(float).eps):
            return np.nan, np.nan, np.nan
        eb_vals = np.maximum(eb_vals, 0.0)
        rB = (eb_vecs * np.sqrt(eb_vals)) @ eb_vecs.T
        vec1 = vec.copy()
        block = vec[:, k - 1 : k1]
        vec1[:, k - 1 : k1] = (rB @ np.diag([-1.0, 1.0]) @ block.T).T
        vec[:, k - 1 : k1] = (rB @ block.T).T
    else:
        vec = eigvecs[:, :k1].copy()
        if k == 0:
            vec = vec * np.sqrt(1.0 / eigvals[0])
        else:
            vec = vec / np.sqrt(eigvals[:k])
        vec1 = vec
        if k == 1:
            rank = 1.0

    Rp = R @ p
    d = float(np.sum((vec.T @ Rp) ** 2))
    d1 = float(np.sum((vec1.T @ Rp) ** 2))
    rank1 = max(float(rank), np.finfo(float).eps)
    pval = float(
        0.5
        * (
            _chi2.sf(max(d, 0.0), df=rank1)
            + _chi2.sf(max(d1, 0.0), df=rank1)
        )
    )
    return d, min(1.0, pval), float(rank)


def _drop_smoother_constant_null(
    beta_s: np.ndarray, V_s: np.ndarray, X_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove the CR-basis constant direction before a per-smoother test.

    With tradeSeq's formula ``-1 + U + s(t1, by=l1) + s(t2, by=l2)``, the
    global intercept and the constant null-space columns of the lineage
    smooths share one unidentifiable direction. mgcv's setup absorbs side
    constraints before storing the per-smooth coefficients; the Python fit
    keeps the original lpmatrix for AnnData/predictGAM parity. Projecting out
    the per-block constant basis direction gives the same estimable smooth
    subspace to ``testStat`` without changing fitted values.
    """
    k = beta_s.size
    if k == 0:
        return beta_s, V_s, X_s
    u = np.ones(k, dtype=np.float64)
    u /= np.linalg.norm(u)
    P = np.eye(k, dtype=np.float64) - np.outer(u, u)
    return P @ beta_s, P @ V_s @ P.T, X_s @ P


def _per_smoother_wald(
    beta: np.ndarray,
    Vp: np.ndarray,
    F: np.ndarray,
    X: np.ndarray,
    smooth_blocks: list[tuple[int, int]],
    labels: list[str],
) -> pd.DataFrame:
    """Per-smoother Wald table matching ``summary(m)$s.table`` columns.

    Mirrors ``mgcv:::summary.gam`` for smooth terms: per-block ``edf`` and
    ``edf1`` are summed from the global influence matrix, then the smooth
    coefficient block is tested with the type-0 ``mgcv:::testStat`` procedure.
    """
    # Pre-compute the diagonal of F·F once: (F·F)[k,k] = sum_j F[k,j]·F[j,k]
    F_arr = np.asarray(F, dtype=np.float64)
    FF_diag = np.einsum("ij,ji->i", F_arr, F_arr)
    F_diag = np.diag(F_arr)
    rows: list[dict] = []
    for (start, stop), label in zip(smooth_blocks, labels):
        # Per-coefficient edf/edf1 summed over the block — mgcv parity.
        edf_s = float(np.sum(F_diag[start:stop]))
        edf1_s = float(np.sum(2.0 * F_diag[start:stop] - FF_diag[start:stop]))
        beta_s = np.asarray(beta[start:stop], dtype=np.float64)
        V_s = np.asarray(Vp[start:stop, start:stop], dtype=np.float64)
        X_s = np.asarray(X[:, start:stop], dtype=np.float64)
        beta_s, V_s, X_s = _drop_smoother_constant_null(beta_s, V_s, X_s)
        rank = min(float(X_s.shape[1]), edf1_s)
        stat, p_val, _rank = _mgcv_test_stat(beta_s, X_s, V_s, rank)
        rows.append(
            {"edf": edf_s, "Ref.df": edf1_s, "Chi.sq": stat, "p-value": p_val}
        )
    return pd.DataFrame(rows, index=labels, columns=["edf", "Ref.df", "Chi.sq", "p-value"])


def _fit_one_gene(
    y: np.ndarray,
    design: DesignMatrices,
    offset: np.ndarray,
    weights: Optional[np.ndarray],
    family: str,
) -> tuple[GamFit | None, bool]:
    """Run :func:`fit_nb_gam_block` on one gene with R-faithful error/warning capture.

    Returns a (fit, converged) tuple; ``fit is None`` on hard failure.
    Captures BOTH errors and warnings — matches R's
    ``try(withCallingHandlers(..., warning=function(w) converged=FALSE))`` at
    ``fitGAM.R:297-307``.
    """
    converged = True
    fit: GamFit | None = None
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fit = fit_nb_gam_block(
                y=y,
                X=design.lpmatrix.values,
                S=design.S,
                offset=offset,
                weights=weights,
                family=family,
            )
            if captured:
                converged = False
            if fit is not None and not fit.converged:
                converged = False
    except Exception:
        fit = None
        converged = False
    return fit, converged


def fit_gam(
    adata: ad.AnnData,
    *,
    layer: str = "counts",
    pseudotime_key: str = "pseudotime",
    weights_key: str = "cell_weights",
    U: Optional[np.ndarray] = None,
    conditions_key: Optional[str] = None,
    genes: Optional[Union[np.ndarray, list, range]] = None,
    sample_weights: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    n_knots: int = 6,
    family: str = "nb",
    parallel: bool = False,
    n_jobs: int = 1,
    verbose: bool = True,
    return_models: bool = False,
    aic: bool = False,
    gcv: bool = False,
    key_added: str = "tradeseq",
    copy: bool = False,
    _w_samp: Optional[np.ndarray] = None,
) -> Optional[ad.AnnData] | dict[str, Optional[FittedGam]] | np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Fit a negative-binomial GAM to every gene along each trajectory lineage.

    Python port of ``tradeSeq::fitGAM`` (R source: ``tradeSeq/R/fitGAM.R:183-310``
    and the three ``setMethod`` dispatchers at lines 488-744). The container
    is AnnData; the equivalents of R's ``rowData`` / ``colData`` /
    ``metadata`` tradeSeq slots are
    populated under ``adata.varm/var/uns[key_added]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Container with raw counts in ``adata.layers[layer]`` and the trajectory
        artefacts in ``adata.obsm[pseudotime_key]`` /
        ``adata.obsm[weights_key]``. ``cells × genes`` AnnData convention is
        assumed; the function transposes once internally.
    layer : str, default "counts"
        Layer holding the raw counts.
    pseudotime_key, weights_key : str
        Keys into ``adata.obsm`` for the trajectory pseudotimes and per-cell
        lineage weights.
    U : numpy.ndarray, optional
        Fixed-effect design, shape ``(n_cells, k)``. Defaults to a single
        intercept column of ones.
    conditions_key : str or None
        If not None, name of an ``adata.obs`` column containing a categorical
        condition vector. Single-level conditions trigger a warning and are
        ignored.
    genes : list/array/range or None
        Subset of genes to fit. ``None`` fits all genes.
    sample_weights : numpy.ndarray or None
        Optional per-gene-per-cell observation weight matrix (e.g. ZINB).
        Shape ``(n_genes, n_cells)`` matching the gene-major R convention.
    offset : numpy.ndarray or None
        Per-cell log-offset. ``None`` triggers TMM-normalized library size
        offsetting via :func:`tradeseq._offset.compute_offset`.
    n_knots : int, default 6
        Number of knots used for the cr-spline basis.
    family : {"nb", "gaussian"}, default "nb"
        Response family. ``"nb"`` runs the joint α-loop.
    parallel : bool, default False
        Toggle parallel per-gene fitting via joblib.
    n_jobs : int, default 1
        Number of joblib workers when ``parallel=True``.
    verbose : bool, default True
        Show a tqdm progress bar over genes.
    return_models : bool, default False
        If False, mutate ``adata`` in place (or return a copy when
        ``copy=True``). If True, return a ``dict[gene, FittedGam]`` for the
        per-gene model contract used by :func:`get_smoother_pvalues` and
        :func:`get_smoother_test_stats`.
    aic : bool, default False
        Return a per-gene AIC vector (R: ``fitGAM.R:332-340``) and skip the
        AnnData / dict assembly entirely. mgcv-faithful NB AIC is
        ``-2 · log_lik + 2 · edf1`` with ``edf1 = 2·tr(F) - tr(F·F)``
        (Wood §6.11.1).
    gcv : bool, default False
        When ``aic=True``, return ``(aic_array, gcv_array)`` instead of just
        the AIC array. R: ``fitGAM.R:341-347`` reads ``m$gcv.ubre``; for
        ``family="nb"`` mgcv stores its REML outer-optimizer score there.
    key_added : str, default "tradeseq"
        Namespace prefix in ``adata.uns`` / ``adata.varm`` / ``adata.var``.
    copy : bool, default False
        If True and ``return_models=False``, mutate a copy and return it.
    _w_samp : numpy.ndarray, optional
        Private validation kwarg — pre-computed multinomial assignment matrix
        for cross-language seed parity. See :func:`assign_cells`.

    Returns
    -------
    anndata.AnnData or None or dict[str, FittedGam] or numpy.ndarray or tuple
        * ``aic=True, gcv=False``: ``np.ndarray`` of shape ``(n_genes,)``.
        * ``aic=True, gcv=True``: ``(aic_array, gcv_array)`` tuple.
        * ``return_models=False, copy=True``: mutated AnnData copy.
        * ``return_models=False, copy=False``: ``None`` (in-place).
        * ``return_models=True``: ``dict[str, Optional[FittedGam]]`` (failed fits are
          recorded as ``None`` so all requested genes appear as keys).
    """
    if conditions_key is not None and return_models:
        warnings.warn(
            "If conditions are provided, AnnData output is usually preferable; "
            "a fitted-model dict is still returned because return_models=True "
            "was requested.",
            stacklevel=2,
        )

    counts_cn, pseudotime, cell_weights, conditions = _resolve_input_arrays(
        adata,
        layer=layer,
        pseudotime_key=pseudotime_key,
        weights_key=weights_key,
        conditions_key=conditions_key,
    )
    _validate_pseudotime(pseudotime, cell_weights)
    n_cells, n_total_genes = counts_cn.shape
    counts_gc = counts_cn.T  # R-shape: genes × cells

    if family == "nb" and (counts_gc < 0).any():
        raise ValueError("All values of the count matrix should be non-negative")

    if U is None:
        U_mat = np.ones((n_cells, 1), dtype=np.float64)
    else:
        U_mat = np.asarray(U, dtype=np.float64)
        if U_mat.ndim == 1:
            U_mat = U_mat[:, None]
        if U_mat.shape[0] != n_cells:
            raise ValueError("The dimensions of U do not match those of counts.")

    gene_idx = _gene_indices(adata, genes)
    gene_names = list(adata.var_names[gene_idx])

    # --- offset
    if offset is None:
        offset_arr = compute_offset(None, counts_gc)
    else:
        offset_arr = np.asarray(offset, dtype=np.float64)

    # --- assign cells, knots
    w_samp = assign_cells(cell_weights, _w_samp=_w_samp)
    knots_list = find_knots(n_knots, pseudotime, w_samp)

    if conditions is None:
        design = build_smooth_design(
            pseudotime=pseudotime,
            w_samp=w_samp,
            U=U_mat,
            knots_list=knots_list,
            offset=offset_arr,
        )
    else:
        design = build_smooth_design_with_conditions(
            pseudotime=pseudotime,
            w_samp=w_samp,
            U=U_mat,
            knots_list=knots_list,
            offset=offset_arr,
            conditions=conditions,
        )

    # --- per-gene fit
    def _do_gene(g_local: int) -> tuple[GamFit | None, bool]:
        g_global = int(gene_idx[g_local])
        y = counts_gc[g_global, :]
        w_obs = None
        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=np.float64)
            w_obs = sw[g_local, :] if sw.shape[0] > 1 else sw[0, :]
        return _fit_one_gene(y, design, offset_arr, w_obs, family)

    iterable = range(len(gene_idx))
    if verbose:
        iterable = tqdm(iterable, total=len(gene_idx), desc="fit_gam")
    if parallel:
        results = Parallel(n_jobs=n_jobs)(delayed(_do_gene)(g) for g in iterable)
    else:
        results = [_do_gene(g) for g in iterable]

    # --- aic-only branch (R: fitGAM.R:332-347): skip AnnData/dict assembly.
    if aic:
        aic_vals = np.full(len(gene_idx), np.nan, dtype=np.float64)
        gcv_vals = np.full(len(gene_idx), np.nan, dtype=np.float64)
        for i, (fit, _ok) in enumerate(results):
            if fit is None:
                continue
            aic_vals[i] = fit.aic
            gcv_vals[i] = fit.gcv
        if gcv:
            return aic_vals, gcv_vals
        return aic_vals

    # --- assemble outputs
    n_coefs = design.lpmatrix.shape[1]
    beta = np.full((len(gene_idx), n_coefs), np.nan, dtype=np.float64)
    sigma_list: list[np.ndarray] = []
    converged = np.zeros(len(gene_idx), dtype=bool)
    fits: dict[str, Optional[FittedGam]] = {}
    smoother_labels = _smoother_labels(design)
    for i, (fit, ok) in enumerate(results):
        if fit is None:
            sigma_list.append(np.full((n_coefs, n_coefs), np.nan))
            if return_models:
                # R: ``rep(NA, nCurves)`` rows for try-error genes. Mirror by
                # inserting a None placeholder so downstream consumers see a
                # full keyset (`get_smoother._stack_summary_column` handles
                # ``fg is None`` already).
                fits[gene_names[i]] = None
            continue
        beta[i, :] = fit.beta
        sigma_list.append(fit.Vp)
        converged[i] = bool(ok)
        if return_models:
            stab = _per_smoother_wald(
                beta=fit.beta,
                Vp=fit.Vp,
                F=fit.F,
                X=design.lpmatrix.values,
                smooth_blocks=design.smooth_blocks,
                labels=smoother_labels,
            )
            fits[gene_names[i]] = FittedGam(
                coef_=fit.beta,
                cov_=fit.Vp,
                lpmatrix_=design.lpmatrix,
                dm_=design.dm,
                summary_s_table=stab,
                alpha_=fit.alpha,
                converged_=bool(ok),
            )

    if return_models:
        return fits

    target = adata.copy() if copy else adata

    # AnnData's `varm` keys must have full n_vars rows. If we fit only a
    # subset of genes, pad with NaN rows so the shape is right; record the
    # subset in `uns[key_added]["genes_fit"]`.
    if len(gene_idx) != target.n_vars:
        beta_full = np.full((target.n_vars, n_coefs), np.nan, dtype=np.float64)
        beta_full[gene_idx, :] = beta
        sigma_full: list[np.ndarray] = [
            np.full((n_coefs, n_coefs), np.nan) for _ in range(target.n_vars)
        ]
        for i, g in enumerate(gene_idx):
            sigma_full[int(g)] = sigma_list[i]
        converged_full = np.zeros(target.n_vars, dtype=bool)
        converged_full[gene_idx] = converged
        beta = beta_full
        sigma_list = sigma_full
        converged = converged_full

    write_fit_results(
        target,
        beta=beta,
        sigma_list=sigma_list,
        converged=converged,
        design_matrix=design.dm,
        lpmatrix=design.lpmatrix.values,
        knots=design.knots,
        family=family,
        conditions=conditions,
        slingshot_coldata=None,
        key=key_added,
    )
    # Record extra metadata that callers (e.g. evaluate_k, get_smoother_*) read.
    uns = target.uns[key_added]
    uns["lpmatrix_columns"] = list(design.lpmatrix.columns)
    uns["genes_fit"] = list(gene_names)
    uns["smooth_blocks"] = [tuple(int(x) for x in pair) for pair in design.smooth_blocks]

    if copy:
        return target
    return None


def nknots(
    models: Union[ad.AnnData, dict],
    *,
    key: str = "tradeseq",
) -> int:
    """Return the number of knots used by ``fit_gam`` for ``models``.

    Port of ``tradeSeq::nknots`` (R source: ``tradeSeq/R/nknots.R:14-29``).

    Parameters
    ----------
    models : anndata.AnnData or dict[str, FittedGam]
        Either an AnnData previously populated by :func:`fit_gam`, or the
        list-mode dict.
    key : str, default "tradeseq"
        Namespace prefix used by :func:`fit_gam`.

    Returns
    -------
    int
    """
    if isinstance(models, ad.AnnData):
        if key not in models.uns or "knots" not in models.uns[key]:
            raise KeyError(
                f"adata.uns[{key!r}] missing 'knots' — call fit_gam first"
            )
        return int(np.asarray(models.uns[key]["knots"]).shape[0])
    if isinstance(models, dict):
        # List-mode: recover from any successful FittedGam by inspecting
        # the lpmatrix column names — the smoother is named ``s(t1):l1.k`` and
        # ``k`` runs from 1 to ``n_knots``. Take the max ``k`` across columns.
        # Skip ``None`` placeholders for failed fits (Gap #24).
        for g, fg in models.items():
            if fg is None:
                continue
            cols = fg.lpmatrix_.columns
            ks = [int(c.split(".")[-1]) for c in cols if c.startswith("s(t1)")]
            if ks:
                return max(ks)
        raise ValueError("Could not derive nknots from an empty model list.")
    raise TypeError(f"Unsupported models type: {type(models).__name__}")
