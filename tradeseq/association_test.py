"""Association test (R source: tradeSeq/R/associationTest.R:1-516).

Port of ``associationTest`` for AnnData populated by :func:`fit_gam`.
Dispatches on the presence of the ``conditions`` slot:

* No conditions → ``.associationTest`` (associationTest.R:1-193). Builds a
  per-lineage ``nPoints``-point contrast block and runs a multivariate Wald
  test per gene.
* Conditions present → ``.associationTest_conditions`` (associationTest.R:
  196-413). Builds one ``nPoints``-point contrast block per
  ``(lineage, condition)`` pair, assembled in ``expand.grid`` order, and emits
  per-pair Wald triplets.
"""

from __future__ import annotations

import re
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from ._test_common import (
    INVERSE_DEFAULT_L2FC0,
    INVERSE_DEFAULT_L2FC_NONZERO,
    build_assoc_per_lineage_contrast,
    fetch_test_arrays,
    max_pseudotime_per_lineage,
    mean_log_fc,
    n_curves_from_dm,
    run_wald_table,
)
from .fit_gam import nknots

__all__ = ["association_test"]


def association_test(
    adata: ad.AnnData,
    *,
    global_: bool = True,
    lineages: bool = False,
    l2fc: float = 0.0,
    n_points: Optional[int] = None,
    contrast_type: str = "start",
    inverse: Optional[str] = None,
    eigen_thresh: float = 1e-2,
    key: str = "tradeseq",
) -> pd.DataFrame:
    """Test whether average gene expression changes along pseudotime.

    Python port of ``tradeSeq::associationTest`` for AnnData inputs
    (mirroring associationTest.R:475-516).
    Dispatches on the presence of ``adata.uns[key]["conditions"]``:

    * If ``None`` → ``.associationTest`` (associationTest.R:1-193). Builds an
      ``n_points``-point per-lineage contrast block and runs a multivariate
      Wald test per gene.
    * Otherwise → ``.associationTest_conditions`` (associationTest.R:196-413).
      Builds one block per ``(lineage, condition)`` pair; the global L is
      assembled in ``expand.grid`` order; the per-lineage block emits
      ``waldStat_lineage{jj}_condition{level}`` columns.

    Parameters
    ----------
    adata : anndata.AnnData
        Container previously populated by :func:`tradeseq.fit_gam`.
    global_ : bool, default True
        If True, return the omnibus statistic across all lineages.
    lineages : bool, default False
        If True, return per-lineage statistics.
    l2fc : float, default 0.0
        TREAT-style log2 fold-change threshold. ``0`` recovers the ordinary
        Wald test.
    n_points : int or None, default None
        Number of contrast pseudotime points per lineage. ``None`` resolves to
        ``2 * nknots(adata)`` (associationTest.R:481).
    contrast_type : {"start", "end", "consecutive"}, default "start"
        Direction of the per-lineage contrast pattern.
    inverse : {"Chol", "QR", "generalized", "eigen"} or None, default None
        Matrix-inverse strategy. ``None`` resolves to ``"Chol"`` when
        ``l2fc == 0`` and ``"eigen"`` otherwise — R's
        ``ifelse(l2fc==0, "Chol", "eigen")`` default (associationTest.R:483).
    eigen_thresh : float, default 1e-2
        Relative eigenvalue cutoff used when ``inverse="eigen"``.
    key : str, default "tradeseq"
        Namespace under which :func:`fit_gam` wrote its outputs.

    Returns
    -------
    pandas.DataFrame
        Per-gene test statistics indexed by gene name.

        No-conditions branch (``.associationTest``):

        * ``global_=True, lineages=False`` → columns
          ``(waldStat, df, pvalue, meanLogFC)``.
        * ``global_=False, lineages=True`` → columns
          ``(waldStat_1, df_1, pvalue_1, ..., meanLogFC)``.
        * ``global_=True, lineages=True`` → both blocks concatenated, with
          ``meanLogFC`` at the end.

        Conditions branch (``.associationTest_conditions``): the per-lineage
        block is replaced by per-(lineage, condition) columns named
        ``waldStat_lineage{jj}_condition{level}`` (and df/pvalue), where
        ``level`` is the actual condition factor level (associationTest.R:
        369-380).
    """
    if not (global_ or lineages):
        raise ValueError("at least one of `global_` or `lineages` must be True")

    if l2fc != 0 and contrast_type not in ("start", "end", "consecutive"):
        raise ValueError(
            f"contrast_type must be 'start', 'end' or 'consecutive'; "
            f"got {contrast_type!r}"
        )

    if inverse is None:
        inverse = (
            INVERSE_DEFAULT_L2FC0 if l2fc == 0 else INVERSE_DEFAULT_L2FC_NONZERO
        )

    beta_all, sigma_all, dm, lpmatrix, pseudotime, conditions = fetch_test_arrays(
        adata, key=key
    )

    n_curves = n_curves_from_dm(dm)
    if n_curves == 0:
        raise ValueError(
            "Could not detect any lineage columns 't1, t2, ...' in the design matrix."
        )

    if n_points is None:
        n_points = 2 * nknots(adata, key=key)

    # Conditions branch: dispatch to .associationTest_conditions
    # (associationTest.R:493-502 → 196-413).
    if conditions is not None:
        return _association_test_conditions(
            adata=adata,
            beta_all=beta_all,
            sigma_all=sigma_all,
            dm=dm,
            lpmatrix=lpmatrix,
            pseudotime=pseudotime,
            conditions=conditions,
            global_=global_,
            lineages=lineages,
            l2fc=l2fc,
            n_points=n_points,
            contrast_type=contrast_type,
            inverse=inverse,
            eigen_thresh=eigen_thresh,
            n_curves=n_curves,
        )

    max_t = max_pseudotime_per_lineage(dm, n_curves)

    # Build per-lineage contrast blocks, then concatenate.
    L_blocks: list[np.ndarray] = []
    for jj in range(1, n_curves + 1):
        L_jj = build_assoc_per_lineage_contrast(
            lpmatrix=lpmatrix,
            dm=dm,
            pseudotime=pseudotime,
            conditions=conditions,
            lineage_id=jj,
            max_t=float(max_t[jj - 1]),
            n_points=n_points,
            contrast_type=contrast_type,
        )
        L_blocks.append(L_jj)
    L = np.concatenate(L_blocks, axis=1)

    gene_index = adata.var_names

    # Global test
    if global_:
        wald_global = run_wald_table(
            beta_all,
            sigma_all,
            L,
            l2fc=l2fc,
            inverse=inverse,
            eigen_thresh=eigen_thresh,
        )

    # Per-lineage test
    if lineages:
        per_lineage = np.empty((beta_all.shape[0], 3 * n_curves), dtype=float)
        for jj in range(n_curves):
            block = run_wald_table(
                beta_all,
                sigma_all,
                L_blocks[jj],
                l2fc=l2fc,
                inverse=inverse,
                eigen_thresh=eigen_thresh,
            )
            # Match R's column ordering: waldStat_jj, df_jj, pvalue_jj go to
            # positions jj, n_curves + jj, 2*n_curves + jj before reordering.
            # We populate the final column ordering directly:
            # waldStat_1, df_1, pvalue_1, waldStat_2, df_2, pvalue_2, ...
            per_lineage[:, 3 * jj : 3 * jj + 3] = block

    # Mean log fold-change column (always computed against the combined L).
    fc_mean = mean_log_fc(beta_all, L)

    # Assemble output DataFrame, column order matches R associationTest.R:186-192.
    cols: dict[str, np.ndarray] = {}
    if global_:
        cols["waldStat"] = wald_global[:, 0]
        cols["df"] = wald_global[:, 1]
        cols["pvalue"] = wald_global[:, 2]
    if lineages:
        for jj in range(1, n_curves + 1):
            cols[f"waldStat_{jj}"] = per_lineage[:, 3 * (jj - 1) + 0]
            cols[f"df_{jj}"] = per_lineage[:, 3 * (jj - 1) + 1]
            cols[f"pvalue_{jj}"] = per_lineage[:, 3 * (jj - 1) + 2]
    cols["meanLogFC"] = fc_mean

    return pd.DataFrame(cols, index=gene_index)


def _max_pseudotime_per_lineage_conditions(
    dm: pd.DataFrame, n_curves: int
) -> np.ndarray:
    """Per-lineage max pseudotime in the conditions-aware design (R:289-290).

    R idiom:

    .. code-block:: R

        lID <- rowSums(dm[, grep(paste0("l", jj))])
        tmax <- max(dm[lID == 1, paste0("t", jj)])

    Operates on a design matrix whose lineage columns are named
    ``l{jj}_{kk}`` for each condition ``kk``. The ``rowSums == 1`` mask picks
    the cells assigned to lineage ``jj`` (any condition).
    """
    out = np.zeros(n_curves, dtype=float)
    for jj in range(1, n_curves + 1):
        rx = re.compile(rf"l{jj}(_|$)")
        l_cols = [c for c in dm.columns if rx.match(c) is not None]
        if not l_cols:
            raise ValueError(
                f"design matrix has no lineage columns matching 'l{jj}(_|$)'"
            )
        l_block = dm[l_cols].to_numpy(dtype=float)
        mask = l_block.sum(axis=1) == 1
        t_col = dm[f"t{jj}"].to_numpy(dtype=float)
        out[jj - 1] = float(t_col[mask].max())
    return out


def _association_test_conditions(
    *,
    adata: ad.AnnData,
    beta_all: np.ndarray,
    sigma_all: np.ndarray,
    dm: pd.DataFrame,
    lpmatrix: pd.DataFrame,
    pseudotime: np.ndarray,
    conditions: pd.Categorical,
    global_: bool,
    lineages: bool,
    l2fc: float,
    n_points: int,
    contrast_type: str,
    inverse: str,
    eigen_thresh: float,
    n_curves: int,
) -> pd.DataFrame:
    """Conditions-aware associationTest (R: .associationTest_conditions, 196-413).

    Iterates over ``(lineage, condition)`` pairs to build per-pair contrast
    blocks ``L_{jj,kk}``, then assembles the global L by ``expand.grid`` order
    (associationTest.R:335-337, 384-386):

    .. code-block:: R

        combs <- apply(expand.grid(seq_len(nCurves), seq_len(nlevels(conditions))),
                       1, paste0, collapse="")
        L <- do.call(cbind, mget(paste0("L", combs)))

    ``expand.grid(seq_len(nCurves), seq_len(K))`` yields rows ``(1,1), (2,1),
    ..., (nCurves,1), (1,2), (2,2), ...`` — column-major over (lineage,
    condition). The per-lineage block in the output table is laid out in
    R-row-major order ``(1,A), (1,B), (2,A), (2,B), ...`` (associationTest.R:
    369-380).
    """
    cond_cat = pd.Categorical(conditions)
    cond_levels = list(cond_cat.categories)
    n_conditions = len(cond_levels)

    # Per-lineage max pseudotime — conditions-aware (rowSums over l{jj}_* == 1).
    max_t = _max_pseudotime_per_lineage_conditions(dm, n_curves)

    # Build per-(lineage, condition) blocks. Key by (jj, kk) — 1-based.
    L_blocks: dict[tuple[int, int], np.ndarray] = {}
    for jj in range(1, n_curves + 1):
        for kk in range(1, n_conditions + 1):
            # The conditions-aware contrast block is built by passing
            # ``condition=kk`` into get_predict_custom_point_df via the existing
            # build_assoc_per_lineage_contrast helper. To do that we need a
            # version that forwards the condition kwarg; since the existing
            # helper doesn't accept it, we inline the loop here.
            C = _build_conditions_lineage_block(
                lpmatrix=lpmatrix,
                dm=dm,
                pseudotime=pseudotime,
                conditions=conditions,
                lineage_id=jj,
                condition_id=kk,
                max_t=float(max_t[jj - 1]),
                n_points=n_points,
                contrast_type=contrast_type,
            )
            L_blocks[(jj, kk)] = C

    # Global L: expand.grid(seq_len(nCurves), seq_len(nlevels(conditions)))
    # rows are (1,1), (2,1), ..., (nCurves,1), (1,2), ...
    # i.e. lineage varies fastest, condition slowest.
    L_combined_cols: list[np.ndarray] = []
    for kk in range(1, n_conditions + 1):
        for jj in range(1, n_curves + 1):
            L_combined_cols.append(L_blocks[(jj, kk)])
    L_combined = np.concatenate(L_combined_cols, axis=1)

    gene_index = adata.var_names

    # Global Wald test
    if global_:
        wald_global = run_wald_table(
            beta_all,
            sigma_all,
            L_combined,
            l2fc=l2fc,
            inverse=inverse,
            eigen_thresh=eigen_thresh,
        )

    # Per-(lineage, condition) Wald tests, in row-major (lineage, condition)
    # order matching R:369-380.
    if lineages:
        per_block = np.empty(
            (beta_all.shape[0], 3 * n_curves * n_conditions), dtype=float
        )
        col_idx = 0
        for jj in range(1, n_curves + 1):
            for kk in range(1, n_conditions + 1):
                block = run_wald_table(
                    beta_all,
                    sigma_all,
                    L_blocks[(jj, kk)],
                    l2fc=l2fc,
                    inverse=inverse,
                    eigen_thresh=eigen_thresh,
                )
                per_block[:, col_idx : col_idx + 3] = block
                col_idx += 3

    fc_mean = mean_log_fc(beta_all, L_combined)

    cols: dict[str, np.ndarray] = {}
    if global_:
        cols["waldStat"] = wald_global[:, 0]
        cols["df"] = wald_global[:, 1]
        cols["pvalue"] = wald_global[:, 2]
    if lineages:
        col_idx = 0
        for jj in range(1, n_curves + 1):
            for kk in range(1, n_conditions + 1):
                tag = f"lineage{jj}_condition{cond_levels[kk - 1]}"
                cols[f"waldStat_{tag}"] = per_block[:, col_idx + 0]
                cols[f"df_{tag}"] = per_block[:, col_idx + 1]
                cols[f"pvalue_{tag}"] = per_block[:, col_idx + 2]
                col_idx += 3
    cols["meanLogFC"] = fc_mean

    return pd.DataFrame(cols, index=gene_index)


def _build_conditions_lineage_block(
    *,
    lpmatrix: pd.DataFrame,
    dm: pd.DataFrame,
    pseudotime: np.ndarray,
    conditions: pd.Categorical,
    lineage_id: int,
    condition_id: int,
    max_t: float,
    n_points: int,
    contrast_type: str,
) -> np.ndarray:
    """Build the (lineage, condition) contrast block ``L_{jj,kk}``.

    Mirrors ``associationTest.R:287-326``: builds the per-pair predictor matrix
    by passing ``condition=kk`` into ``.getPredictCustomPointDf``, then fills
    the contrast matrix according to ``contrastType``.
    """
    from ._design import get_predict_custom_point_df
    from ._predict import predict_gam

    n_coefs = lpmatrix.shape[1]
    contrast_points = np.linspace(0.0, max_t, n_points)
    df_points = pd.concat(
        [
            get_predict_custom_point_df(
                dm, lineage_id, float(tp), condition=condition_id
            )
            for tp in contrast_points
        ],
        axis=0,
        ignore_index=True,
    )
    x_points = predict_gam(
        lpmatrix=lpmatrix,
        df=df_points,
        pseudotime=pseudotime,
        conditions=conditions,
    ).to_numpy()  # (n_points, p)

    C = np.zeros((n_coefs, n_points - 1), dtype=float)
    if contrast_type == "start":
        for pp in range(1, n_points):
            C[:, pp - 1] = x_points[pp, :] - x_points[0, :]
    elif contrast_type == "end":
        for pp in range(1, n_points):
            C[:, pp - 1] = x_points[pp - 1, :] - x_points[-1, :]
    elif contrast_type == "consecutive":
        for pp in range(1, n_points):
            C[:, pp - 1] = x_points[pp, :] - x_points[pp - 1, :]
    else:
        raise ValueError(
            f"contrast_type must be one of 'start', 'end', 'consecutive'; "
            f"got {contrast_type!r}"
        )
    return C
