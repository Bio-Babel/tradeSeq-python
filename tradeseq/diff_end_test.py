"""Differential end-point test (R source: tradeSeq/R/diffEndTest.R:1-237).

Port of ``diffEndTest`` for AnnData populated by :func:`fit_gam`
(mirroring diffEndTest.R:43-72). Builds an all-pairs end-point contrast matrix
(``combn(nLineages, 2)``) and runs a multivariate Wald test per gene.
"""

from __future__ import annotations

import re
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from ._test_common import (
    build_diffend_pair_contrasts,
    fetch_test_arrays,
    get_fold_changes,
    n_curves_from_dm,
    run_wald_table,
)

__all__ = ["diff_end_test"]


def diff_end_test(
    adata: ad.AnnData,
    *,
    global_: bool = True,
    pairwise: bool = False,
    l2fc: float = 0.0,
    key: str = "tradeseq",
) -> pd.DataFrame:
    """Test differential expression between the endpoints of all lineage pairs.

    Python port of ``tradeSeq::diffEndTest`` for AnnData inputs
    (mirroring diffEndTest.R:205-219). Builds a per-pair contrast
    ``X_end_i - X_end_j`` for every ``combn(nLineages, 2)`` pair and runs a
    multivariate Wald test per gene.

    Parameters
    ----------
    adata : anndata.AnnData
        Container previously populated by :func:`tradeseq.fit_gam`.
    global_ : bool, default True
        If True, return the omnibus statistic across all pairs.
    pairwise : bool, default False
        If True, return per-pair statistics. When the design matrix has only
        two lineage indicator columns (``l[1-9]`` count = 2; this is the
        no-conditions / single-lineage-pair regime), the per-pair statistic
        equals the global one, so this flag is silently coerced to ``False``
        to mirror the R behaviour (diffEndTest.R:30-33). With conditions
        present and 2 lineages × 2 conditions the L-column count is 4 and
        pairwise is preserved (combn(2, 2) = 1 underlying pair).
    l2fc : float, default 0.0
        TREAT-style log2 fold-change threshold.
    key : str, default "tradeseq"
        Namespace under which :func:`fit_gam` wrote its outputs.

    Returns
    -------
    pandas.DataFrame
        Per-gene test statistics indexed by gene name.

        * ``global_=True, pairwise=False`` → ``(waldStat, df, pvalue,
          logFC1_2, logFC1_3, ...)``.
        * ``global_=False, pairwise=True`` →
          ``(waldStat_1vs2, df_1vs2, pvalue_1vs2, ..., logFC1_2, ...)``.
        * ``global_=True, pairwise=True`` → both blocks concatenated.
    """
    if not (global_ or pairwise):
        raise ValueError("at least one of `global_` or `pairwise` must be True")

    beta_all, sigma_all, dm, lpmatrix, pseudotime, conditions = fetch_test_arrays(
        adata, key=key
    )

    # R uses two distinct greps at diffEndTest.R:26-27:
    #   nCurves   <- length(grep("l[1-9]"))   # counts ALL lineage indicator cols
    #                                         # (= nLineages * nConditions when
    #                                         # conditions are present)
    #   nLineages <- length(grep("t[1-9]"))   # counts pseudotime cols (= true
    #                                         # number of trajectory lineages)
    # The ``nCurves`` count drives the n==1/n==2 early-return guards (R:29-33);
    # the ``nLineages`` count drives the ``combn(nLineages, 2)`` pair loop. The
    # two values agree only in the no-conditions case.
    _l_re = re.compile(r"l[1-9]")
    n_curves = sum(1 for c in dm.columns if _l_re.search(c) is not None)
    n_lineages = n_curves_from_dm(dm)
    if n_lineages == 0:
        raise ValueError(
            "Could not detect any lineage columns 't1, t2, ...' in the design matrix."
        )
    if n_curves == 1:
        raise ValueError("You cannot run this test with only one lineage.")
    if n_curves == 2 and pairwise:
        # R emits ``message()`` here; we use ``warnings.warn`` so the caller can
        # capture/silence it. Mirrors diffEndTest.R:30-33.
        warnings.warn(
            "Only two lineages; skipping pairwise comparison.",
            stacklevel=2,
        )
        pairwise = False

    L, pairs = build_diffend_pair_contrasts(
        lpmatrix=lpmatrix,
        dm=dm,
        pseudotime=pseudotime,
        conditions=conditions,
        n_lineages=n_lineages,
    )
    # ``inverse`` defaults to "QR" (waldTestFC default).
    inverse = "QR"

    # Global Wald (omnibus over all pairs)
    if global_:
        wald_global = run_wald_table(
            beta_all, sigma_all, L, l2fc=l2fc, inverse=inverse
        )

    # Per-pair Wald
    if pairwise:
        n_pairs = L.shape[1]
        per_pair = np.empty((beta_all.shape[0], 3 * n_pairs), dtype=float)
        for jj in range(n_pairs):
            L_jj = L[:, [jj]]
            block = run_wald_table(
                beta_all, sigma_all, L_jj, l2fc=l2fc, inverse=inverse
            )
            per_pair[:, 3 * jj : 3 * jj + 3] = block

    # Pairwise fold changes (signed): one column per pair, named "logFC<i>_<j>".
    n_pairs_all = L.shape[1]
    fc_all = np.empty((beta_all.shape[0], n_pairs_all), dtype=float)
    for ii in range(beta_all.shape[0]):
        if np.isnan(beta_all[ii, :]).any():
            fc_all[ii, :] = np.nan
        else:
            fc_all[ii, :] = get_fold_changes(beta_all[ii, :], L)

    cols: dict[str, np.ndarray] = {}
    if global_:
        cols["waldStat"] = wald_global[:, 0]
        cols["df"] = wald_global[:, 1]
        cols["pvalue"] = wald_global[:, 2]
    if pairwise:
        for jj, (i, j) in enumerate(pairs):
            cols[f"waldStat_{i}vs{j}"] = per_pair[:, 3 * jj + 0]
            cols[f"df_{i}vs{j}"] = per_pair[:, 3 * jj + 1]
            cols[f"pvalue_{i}vs{j}"] = per_pair[:, 3 * jj + 2]
    for jj, (i, j) in enumerate(pairs):
        cols[f"logFC{i}_{j}"] = fc_all[:, jj]

    return pd.DataFrame(cols, index=adata.var_names)
