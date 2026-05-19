"""evaluate_k diagnostic (R source: tradeSeq/R/evaluateK.R)."""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import gammaln
from tqdm.auto import tqdm

import ggplot2_py as g
import patchwork

from ._assign_cells import assign_cells
from ._design_builder import (
    build_smooth_design,
    build_smooth_design_with_conditions,
)
from ._gam import GamFit, fit_nb_gam_block
from ._knots import find_knots
from ._offset import compute_offset

__all__ = ["evaluate_k", "plot_evaluatek_results"]


def _compute_edf1(
    fit: GamFit,
    X: np.ndarray,
    S: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray,
    family: str,
) -> tuple[float, float]:
    """Compute mgcv's AIC-effective EDF and trace EDF from a converged fit.

    Mirrors Wood §6.11.1: ``F = (X'WX + λS)^{-1} X'WX`` (the converged hat
    matrix), so ``trace(F) = edf`` and ``edf1 = 2·trace(F) - trace(F·F)`` is
    the second-order correction used by mgcv's ``m$aic``. Reconstructs ``W``
    from the converged ``β, α`` via the NB working-weights formula
    ``W_i = weights_i · μ_i / (1 + α·μ_i)`` (Poisson limit when family is
    Gaussian/Poisson, i.e. ``W = weights · μ``).

    Uses :func:`numpy.linalg.pinv` on ``H = X'WX + λS`` (rather than
    :attr:`GamFit.Vp`, which is computed via :func:`scipy.linalg.inv` and
    may have wildly negative eigenvalues when ``H`` is ill-conditioned —
    a known limitation of the slice-4 ``_gam.py``). The pseudoinverse is
    rank-truncated by SVD threshold so it stays well-defined even when
    ``H`` is exactly singular at the unpenalised intercept column.

    Parameters
    ----------
    fit
        A converged :class:`GamFit`. Only ``fit.beta``, ``fit.alpha`` and
        ``fit.lam`` are used (NOT ``fit.Vp`` / ``fit.edf``, which can be
        unreliable when the inner IRLS landed on an ill-conditioned λ).
    X, S, offset, weights
        The design matrix, penalty, offset and per-observation weights
        passed into the IRLS that produced ``fit``.
    family
        ``"nb"`` (NB working weights) or ``"gaussian"`` (Poisson-limit
        working weights).

    Returns
    -------
    tuple of (edf, edf1) : float, float
        ``edf = trace(F)`` (mgcv's ``m$edf`` — the GCV-relevant EDF) and
        ``edf1 = 2·edf - trace(F·F)`` (mgcv's AIC-effective EDF).
    """
    eta = X @ fit.beta + offset
    eta = np.clip(eta, -50.0, 50.0)
    mu = np.exp(eta)
    if family == "nb":
        w = weights * mu / (1.0 + fit.alpha * mu)
    else:
        # Mirror _irls_with_offset's Gaussian/Poisson branch.
        w = weights * mu
    XtWX = (X.T * w) @ X
    H = XtWX + float(fit.lam) * S
    # Use pinv so a rank-deficient H (e.g. when the unpenalised intercept
    # column drives the smallest eigenvalue of H to zero) does not blow up
    # the inverse with a 1e13-scale spurious eigenvalue, which would
    # corrupt downstream trace computations.
    H_inv = np.linalg.pinv(H)
    F = H_inv @ XtWX
    edf = float(np.trace(F))
    trace_FF = float(np.sum(F * F.T))
    edf1 = 2.0 * edf - trace_FF
    return edf, edf1


def _nb_aic(
    y: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    edf1: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Mgcv-style negative-binomial AIC for a fitted model.

    Mirrors ``mgcv:::nb()$aic`` and the gam.fit3 assembly
    ``m$aic = -2·logLik + 2·sum(edf1)`` (Wood §6.11.1):

    For ``Theta = 1/alpha`` (NB inverse-dispersion) and per-observation prior
    weights ``wt``,

        AIC = 2 * sum(wt * [ (y + Theta) * log(mu + Theta) - y * log(mu)
                             + lgamma(y + 1) - Theta * log(Theta)
                             + lgamma(Theta) - lgamma(Theta + y) ])
              + 2 * edf1

    where ``edf1 = 2·trace(F) - trace(F·F)`` is the second-order-corrected
    effective degrees of freedom (NOT the bare ``trace(F)`` that
    ``GamFit.aic`` uses, which is suitable for IRLS-internal model
    comparison but NOT mgcv's ``m$aic`` returned to R's ``evaluateK``).

    Parameters
    ----------
    y : ndarray, shape (n,)
        Observed counts.
    mu : ndarray, shape (n,)
        Fitted means from the converged GAM.
    alpha : float
        NB dispersion (``Var(y) = mu + alpha * mu^2``).
    edf1 : float
        AIC-effective EDF (``2·trace(F) - trace(F·F)``).
    weights : ndarray or None
        Per-observation prior weights (``m$prior.weights`` in mgcv).
        Defaults to ones.
    """
    if weights is None:
        weights = np.ones_like(y, dtype=np.float64)
    theta = 1.0 / max(float(alpha), 1e-300)
    mu_safe = np.maximum(mu, 1e-300)
    term = (
        (y + theta) * np.log(mu_safe + theta)
        - y * np.log(mu_safe)
        + gammaln(y + 1.0)
        - theta * np.log(theta)
        + gammaln(theta)
        - gammaln(theta + y)
    )
    return float(2.0 * np.sum(weights * term) + 2.0 * float(edf1))


def _resolve_inputs(
    adata: ad.AnnData,
    layer: str,
    pseudotime_key: str,
    weights_key: str,
    conditions: Optional[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[pd.Categorical]]:
    """Pull counts/pseudotime/cell_weights/conditions out of the AnnData.

    Mirrors the resolution logic in ``fit_gam`` but trimmed to the slots
    ``evaluate_k`` needs.
    """
    if layer not in adata.layers:
        raise KeyError(
            f"adata.layers has no key {layer!r}; expected the raw count layer "
            f"used by fit_gam"
        )
    counts = adata.layers[layer]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = np.asarray(counts).astype(np.float64)
    if pseudotime_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm has no key {pseudotime_key!r}; populate it before evaluate_k"
        )
    pseudotime = np.asarray(adata.obsm[pseudotime_key], dtype=np.float64)
    if pseudotime.ndim == 1:
        pseudotime = pseudotime[:, None]
    if weights_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm has no key {weights_key!r}; populate it before evaluate_k"
        )
    cell_weights = np.asarray(adata.obsm[weights_key], dtype=np.float64)
    if cell_weights.ndim == 1:
        cell_weights = cell_weights[:, None]
    if np.isnan(pseudotime).any():
        raise ValueError(
            "The pseudotimes contain NA values, and these cannot be used for GAM fitting."
        )
    if np.isnan(cell_weights).any():
        raise ValueError(
            "The cellWeights contain NA values, and these cannot be used for GAM fitting."
        )
    if pseudotime.shape != cell_weights.shape:
        raise ValueError(
            "pseudotime and cellWeights must have identical dimensions."
        )

    conditions_cat: Optional[pd.Categorical] = None
    if conditions is not None:
        if conditions not in adata.obs.columns:
            raise KeyError(
                f"adata.obs has no column {conditions!r}; cannot resolve "
                f"conditions"
            )
        conditions_cat = pd.Categorical(adata.obs[conditions].values)

    return counts, pseudotime, cell_weights, conditions_cat


def evaluate_k(
    adata: ad.AnnData,
    *,
    layer: str = "counts",
    pseudotime_key: str = "pseudotime",
    weights_key: str = "cell_weights",
    k_range: Iterable[int] = range(3, 11),
    n_genes: int = 500,
    plot: bool = True,
    aic_diff: float = 2.0,
    random_state: int = 176201,
    verbose: bool = True,
    parallel: bool = False,
    n_jobs: int = 1,
    U: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    family: str = "nb",
    gcv: bool = False,
    conditions: Optional[str] = None,
    _w_samp: Optional[np.ndarray] = None,
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Diagnostic loop over candidate knot counts for the NB-GAM.

    Python port of ``tradeSeq::evaluateK`` (R source:
    ``tradeSeq/R/evaluateK.R:1-186``). For each ``k`` in ``k_range``, fits an
    NB-GAM to a random subset of ``n_genes`` genes and records the per-gene
    AIC. Returns a ``(n_genes, len(k_range))`` DataFrame indexed by the
    sampled gene names with columns ``"k: <k>"`` (matching R's ``aicMat``
    row/column names).

    Parameters
    ----------
    adata : anndata.AnnData
        Container with raw counts in ``adata.layers[layer]`` and the
        trajectory artefacts in ``adata.obsm[pseudotime_key]`` and
        ``adata.obsm[weights_key]``. ``cells × genes`` convention.
    layer : str, default "counts"
        Layer holding the raw counts.
    pseudotime_key, weights_key : str
        Keys into ``adata.obsm`` for the trajectory pseudotimes and per-cell
        lineage weights.
    k_range : iterable of int, default ``range(3, 11)``
        The candidate values of ``n_knots`` to evaluate. Must contain more
        than one entry and every value must be at least 3 (mirrors R
        ``stop("Cannot fit with fewer than 3 knots ...")`` and
        ``stop("There should be more than one k value")``).
    n_genes : int, default 500
        Number of genes drawn at random from ``adata.var_names`` for the
        diagnostic. Mirrors the R default ``nGenes = 500``. Mirroring R's
        ``sample(seq_len(n), nGenes)``, ``n_genes > n_total_genes`` raises
        :class:`ValueError` (R implicitly errors via ``sample``).
    plot : bool, default True
        If True, render the 4-panel diagnostic via
        :func:`plot_evaluatek_results`. The matrix is returned in either
        case.
    aic_diff : float, default 2.0
        Threshold passed straight through to
        :func:`plot_evaluatek_results`. Mirrors R ``aicDiff = 2``.
    random_state : int, default 176201
        Seed for the gene subsample and the per-k ``assign_cells`` draws.
        Mirrors R ``set.seed(...)`` ahead of ``evaluateK``.
    verbose : bool, default True
        Show a progress bar over genes for each candidate ``k``.
    parallel : bool, default False
        Dispatch per-gene fits via joblib within each candidate ``k``.
    n_jobs : int, default 1
        Number of joblib workers when ``parallel=True``.
    U : numpy.ndarray, optional
        Fixed-effect design, shape ``(n_cells, k)``. Defaults to a single
        intercept column of ones.
    sample_weights : numpy.ndarray, optional
        Optional per-gene-per-cell observation-weight matrix. Shape
        ``(n_total_genes, n_cells)`` matching the gene-major R convention;
        the function indexes rows by the random gene subset.
    offset : numpy.ndarray, optional
        Per-cell log-offset. Two shapes are accepted (mirroring R
        ``offset[teller,]`` at ``fitGAM.R:262``):

        - 1-D ``(n_cells,)`` — same offset used for every gene.
        - 2-D ``(n_total_genes, n_cells)`` — per-gene offset; row ``g`` is
          used for gene ``g``.

        ``None`` triggers TMM-normalized library-size offsetting via
        :func:`tradeseq._offset.compute_offset` on the *full* gene matrix
        (mirrors R ``calcNormFactors(counts)`` before the gene subset is
        drawn).
    family : {"nb", "gaussian"}, default "nb"
        Response family passed to :func:`tradeseq._gam.fit_nb_gam_block`.
    gcv : bool, default False
        When True, return a dict ``{"aic": <DataFrame>, "gcv": <DataFrame>}``
        with the per-(gene, k) score stored by R in ``m$gcv.ubre``. For NB
        fits in mgcv 1.9.3 this field contains the REML optimizer score, not
        the literal GCV formula, because ``mgcv::nb()`` is an
        ``extended.family``.
    conditions : str or None, default None
        Name of an ``adata.obs`` column with categorical condition labels.
        When provided, the design switches to
        :func:`build_smooth_design_with_conditions` so each lineage gets a
        condition-specific smoother. Mirrors R ``evaluateK(..., conditions
        = ...)`` (R source ``evaluateK.R:131, 204, 277``).
    _w_samp : numpy.ndarray, optional
        Private validation kwarg — pre-computed multinomial assignment
        matrix for cross-language seed parity. When supplied it is used for
        EVERY ``k`` (deterministic test path); the per-k draw is bypassed.

    Returns
    -------
    pandas.DataFrame or dict[str, pandas.DataFrame]
        When ``gcv=False`` (default) a DataFrame of shape
        ``(n_genes, len(k_range))`` with row index equal to the sampled
        gene names and columns ``"k: 3"``, ``"k: 4"``, ... (mirrors R's
        ``aicMat`` row/col names). When ``gcv=True``, a dict with keys
        ``"aic"`` and ``"gcv"`` carrying two such DataFrames.
    """
    k_list = [int(kk) for kk in k_range]
    if any(kk < 3 for kk in k_list):
        raise ValueError("Cannot fit with fewer than 3 knots, please increase k.")
    if len(k_list) == 1:
        raise ValueError("There should be more than one k value")

    counts_cn, pseudotime, cell_weights, conditions_cat = _resolve_inputs(
        adata,
        layer=layer,
        pseudotime_key=pseudotime_key,
        weights_key=weights_key,
        conditions=conditions,
    )
    n_cells, n_total_genes = counts_cn.shape
    counts_gc = counts_cn.T  # gene × cell

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

    # Offset on the full matrix — mirrors evaluateK.R:9-18.
    if offset is None:
        offset_arr = compute_offset(None, counts_gc)
    else:
        offset_arr = np.asarray(offset, dtype=np.float64)
    if offset_arr.ndim == 2:
        if offset_arr.shape != (n_total_genes, n_cells):
            raise ValueError(
                f"2-D offset must have shape (n_total_genes, n_cells)="
                f"({n_total_genes}, {n_cells}); got {offset_arr.shape}."
            )
    elif offset_arr.ndim != 1:
        raise ValueError(
            f"offset must be 1-D (n_cells,) or 2-D (n_total_genes, n_cells); "
            f"got ndim={offset_arr.ndim}."
        )

    # geneSub <- sample(seq_len(nrow(counts)), nGenes) — evaluateK.R:21.
    # R's `sample(seq_len(n), nGenes)` errors when nGenes > n; mirror that.
    if int(n_genes) > n_total_genes:
        raise ValueError(
            f"n_genes={int(n_genes)} exceeds the number of available genes "
            f"({n_total_genes}); cannot draw a subsample without replacement."
        )
    rng = np.random.default_rng(random_state)
    gene_sub = rng.choice(n_total_genes, size=int(n_genes), replace=False)

    if sample_weights is not None:
        sw = np.asarray(sample_weights, dtype=np.float64)
        if sw.shape != (n_total_genes, n_cells):
            raise ValueError(
                "sample_weights must have shape (n_total_genes, n_cells)."
            )
    else:
        sw = None

    n_pick = int(n_genes)
    aic_mat = np.full((n_pick, len(k_list)), np.nan, dtype=np.float64)
    gcv_mat = np.full((n_pick, len(k_list)), np.nan, dtype=np.float64)

    for kk_idx, kk in enumerate(k_list):
        # R re-draws .assignCells(cellWeights) inside every .fitGAM call
        # (fitGAM.R:230), so each k gets an independent multinomial sample.
        # When `_w_samp` is supplied we replay the same R-side draw for every
        # k (deterministic validation hook).
        w_samp = assign_cells(cell_weights, rng=rng, _w_samp=_w_samp)
        knots_list = find_knots(kk, pseudotime, w_samp)
        if conditions_cat is None:
            design = build_smooth_design(
                pseudotime=pseudotime,
                w_samp=w_samp,
                U=U_mat,
                knots_list=knots_list,
                offset=(
                    offset_arr if offset_arr.ndim == 1
                    else offset_arr[int(gene_sub[0]), :]
                ),
            )
        else:
            design = build_smooth_design_with_conditions(
                pseudotime=pseudotime,
                w_samp=w_samp,
                U=U_mat,
                knots_list=knots_list,
                offset=(
                    offset_arr if offset_arr.ndim == 1
                    else offset_arr[int(gene_sub[0]), :]
                ),
                conditions=conditions_cat,
            )
        X = design.lpmatrix.values
        S = design.S

        def _fit_one_gene_for_k(g_local: int, g_global: int) -> tuple[float, float]:
            y = counts_gc[int(g_global), :]
            w_obs = None
            if sw is not None:
                w_obs = sw[int(g_global), :]
            # Per-gene offset slice when 2-D offset was supplied (R:
            # `offset[teller,]` at fitGAM.R:262).
            if offset_arr.ndim == 1:
                gene_offset = offset_arr
            else:
                gene_offset = offset_arr[int(g_global), :]

            # R wraps each mgcv::gam(...) call in try(withCallingHandlers(...))
            # at fitGAM.R:297-340: failed fits become try-error and AIC = NA.
            # Mirror that here so a single ill-conditioned gene does not abort
            # the diagnostic for the rest. Capture warnings too (R's
            # withCallingHandlers warning branch sets converged=FALSE but the
            # AIC entry is still recorded; we keep that recording semantic).
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    fit = fit_nb_gam_block(
                        y=y,
                        X=X,
                        S=S,
                        offset=gene_offset,
                        weights=w_obs,
                        family=family,
                    )
            except Exception:
                return np.nan, np.nan

            # Reconstruct fitted mean and compute mgcv-faithful AIC. R's
            # ``m$aic`` uses ``edf1 = 2·edf - trace(F·F)`` (Wood §6.11.1), not
            # the simpler ``+ 2·edf`` form ``GamFit.aic`` carries. We also
            # re-derive ``edf = trace(F)`` from a pinv-based hat-matrix
            # reconstruction; ``fit.edf`` from the inner IRLS can be
            # unreliable when the GCV grid lands on an ill-conditioned λ.
            eta = X @ fit.beta + gene_offset
            eta = np.clip(eta, -50.0, 50.0)
            mu = np.exp(eta)
            weights_for_aic = (
                np.ones_like(y, dtype=np.float64) if w_obs is None else w_obs
            )
            edf_trace, edf1 = _compute_edf1(
                fit, X, S, gene_offset, weights_for_aic, family
            )
            aic_val = _nb_aic(
                y, mu, fit.alpha, edf1, weights=w_obs
            )
            # Keep the EDF reconstruction explicit: it is required for the AIC
            # penalty above. The fit kernel's ``gcv`` field carries the
            # mgcv-compatible NB REML score exposed by R as ``m$gcv.ubre``.
            _ = edf_trace
            return aic_val, fit.gcv

        gene_items = list(enumerate(gene_sub))
        if parallel:
            fit_values = Parallel(n_jobs=n_jobs)(
                delayed(_fit_one_gene_for_k)(g_local, int(g_global))
                for g_local, g_global in gene_items
            )
        else:
            iterable = gene_items
            if verbose:
                iterable = tqdm(
                    gene_items,
                    total=len(gene_items),
                    desc=f"evaluate_k k={kk}",
                )
            fit_values = [
                _fit_one_gene_for_k(g_local, int(g_global))
                for g_local, g_global in iterable
            ]

        for g_local, (aic_val, gcv_val) in enumerate(fit_values):
            aic_mat[g_local, kk_idx] = aic_val
            gcv_mat[g_local, kk_idx] = gcv_val

    row_index = pd.Index(
        adata.var_names[np.asarray(gene_sub, dtype=np.int64)],
        name=None,
    )
    cols = [f"k: {k}" for k in k_list]
    aic_df = pd.DataFrame(aic_mat, index=row_index, columns=cols)
    gcv_df = pd.DataFrame(gcv_mat, index=row_index, columns=cols)

    if plot:
        fig = plot_evaluatek_results(aic_df, k_range=k_list, aic_diff=aic_diff)
        aic_df.attrs["plot"] = fig
        try:
            from IPython.display import display
        except ImportError:
            pass
        else:
            display(fig)

    if gcv:
        if plot:
            gcv_df.attrs["plot"] = aic_df.attrs["plot"]
        return {"aic": aic_df, "gcv": gcv_df}
    return aic_df


def plot_evaluatek_results(
    aic_matrix: Union[np.ndarray, pd.DataFrame],
    *,
    k_range: Optional[Iterable[int]] = None,
    aic_diff: float = 2.0,
    fig_width: float | None = 12.0,
    fig_height: float | None = 3.0,
    fig_dpi: int | None = None,
) -> patchwork.Patchwork:
    """Render the 4-panel ``evaluateK`` diagnostic.

    Python port of ``tradeSeq::plot_evalutateK_results`` (R typo preserved;
    Python name reads ``plot_evaluatek_results``). Each of the four R panels
    becomes a separate ggplot2_py geom, composed via
    :func:`patchwork.wrap_plots`.

    The four panels (see R source at
    ``tradeSeq/R/evaluateK.R:342-374``):

    1. Boxplot of per-gene deviations from the gene-wise mean AIC across
       ``k``.
    2. Mean AIC across genes for each ``k``.
    3. Mean relative AIC across genes (each row scaled by its value at the
       first ``k``).
    4. Bar chart of the count of genes whose argmin-k attains each value
       observed in the optimal-k tabulation, restricted to genes with AIC
       range strictly larger than ``aic_diff``. Mirrors R's ``table(...)``
       call which only shows k values that are some gene's optimum.

    Parameters
    ----------
    aic_matrix : numpy.ndarray or pandas.DataFrame
        ``(n_genes, len(k_range))`` AIC matrix from :func:`evaluate_k`. When
        a DataFrame is supplied, ``k_range`` defaults to the column names
        parsed via ``int(col.replace("k: ", ""))`` — mirrors R's
        ``as.numeric(sub("k: ", "", colnames(aicMat)))``.
    k_range : iterable of int, optional
        The ``k`` axis values. Defaults to ``range(3, 11)`` when neither
        the DataFrame columns nor an explicit ``k_range`` provides them.
        Must have the same length as ``aic_matrix.shape[1]``.
    aic_diff : float, default 2.0
        Minimum AIC range a gene must have to enter panel 4. Mirrors R
        ``aicDiff = 2``.
    fig_width, fig_height, fig_dpi : optional
        Notebook display hints on the returned patchwork object. The R source
        draws the four panels as ``par(mfrow = c(1, 4))``; the defaults
        therefore use a wide, low canvas instead of patchwork-python's generic
        square-ish display size. Saving via ``ggsave`` can still override size
        explicitly.

    Returns
    -------
    patchwork.Patchwork
        Composed 4-panel layout.
    """
    if isinstance(aic_matrix, pd.DataFrame):
        if k_range is None:
            # R: k <- as.numeric(sub("k: ", "", colnames(aicMat)))
            k_list = [
                int(str(c).replace("k: ", "").strip())
                for c in aic_matrix.columns
            ]
        else:
            k_list = [int(kk) for kk in k_range]
        aic = aic_matrix.to_numpy(dtype=np.float64)
    else:
        if k_range is None:
            k_list = list(range(3, 11))
        else:
            k_list = [int(kk) for kk in k_range]
        aic = np.asarray(aic_matrix, dtype=np.float64)

    if aic.ndim != 2:
        raise ValueError(
            f"aic_matrix must be 2-D; got ndim={aic.ndim}."
        )
    n_genes, n_k = aic.shape
    if len(k_list) != n_k:
        raise ValueError(
            f"k_range has length {len(k_list)} but aic_matrix has {n_k} columns."
        )

    # --- Panel 1: per-gene deviation from the gene-wise mean AIC.
    row_means = aic.mean(axis=1, keepdims=True)
    devs = aic - row_means
    df1 = pd.DataFrame(
        {
            "k": np.repeat(k_list, n_genes),
            "dev": devs.T.reshape(-1),
        }
    )
    df1["k"] = pd.Categorical(
        df1["k"].astype(int), categories=k_list, ordered=True
    )
    p1 = (
        g.ggplot(df1, g.aes(x="k", y="dev"))
        + g.geom_boxplot()
        + g.labs(
            x="Number of knots",
            y="Deviation from genewise average AIC",
        )
    )

    # --- Panel 2: mean AIC across genes.
    mean_aic = np.nanmean(aic, axis=0)
    df2 = pd.DataFrame({"k": k_list, "mean_aic": mean_aic})
    p2 = (
        g.ggplot(df2, g.aes(x="k", y="mean_aic"))
        + g.geom_point()
        + g.geom_line()
        + g.labs(x="Number of knots", y="Average AIC")
    )

    # --- Panel 3: mean relative AIC (each row divided by its first column).
    first_col = aic[:, [0]]
    rel_aic = aic / first_col
    mean_rel = np.nanmean(rel_aic, axis=0)
    df3 = pd.DataFrame({"k": k_list, "mean_rel_aic": mean_rel})
    p3 = (
        g.ggplot(df3, g.aes(x="k", y="mean_rel_aic"))
        + g.geom_point()
        + g.geom_line()
        + g.labs(x="Number of knots", y="Relative AIC")
    )

    # --- Panel 4: optimal-k counts for genes with sufficient AIC variation.
    # R: aicRange <- apply(apply(aicMat, 1, range), 2, diff)
    # then varID <- which(aicRange > aicDiff), then
    # `table(k[apply(aicMatSub, 1, which.min)])`.
    # ``table`` only shows k values that are some gene's optimum, so we do
    # NOT reindex to the full k_list with zeros — that was a Python-only
    # embellishment now removed (mirror R exactly).
    aic_range_per_gene = aic.max(axis=1) - aic.min(axis=1)
    var_mask = aic_range_per_gene > float(aic_diff)
    if var_mask.any():
        sub = aic[var_mask]
        best_idx = np.argmin(sub, axis=1)
        best_k = np.array([k_list[i] for i in best_idx])
        counts_df = (
            pd.Series(best_k).value_counts().sort_index()
            .rename_axis("k").reset_index(name="n")
        )
    else:
        counts_df = pd.DataFrame({"k": [], "n": []})
    # Categorical so the geom_col axis is sorted; categories come from the
    # observed best-k values (not the full k_list). When no gene varies
    # enough, render an empty panel-4 (still a valid Patchwork).
    if len(counts_df) > 0:
        counts_df["k"] = pd.Categorical(
            counts_df["k"].astype(int),
            categories=sorted(counts_df["k"].astype(int).tolist()),
            ordered=True,
        )
    p4 = (
        g.ggplot(counts_df, g.aes(x="k", y="n"))
        + g.geom_col()
        + g.labs(x="Number of knots", y="# Genes with optimal k")
    )

    fig = patchwork.wrap_plots([p1, p2, p3, p4], nrow=1)
    if fig_width is not None:
        fig.fig_width = float(fig_width)
    if fig_height is not None:
        fig.fig_height = float(fig_height)
    if fig_dpi is not None:
        fig.fig_dpi = int(fig_dpi)
    return fig
