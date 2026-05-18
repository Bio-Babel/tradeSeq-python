"""Expression-peak cascade stub (R source: tradeSeq/R/cascade.R).

This module exposes the public name :func:`cascade` so downstream code that
imports it gets a clear deferral error rather than ``AttributeError``. The
real algorithm — pointwise first-derivative thresholding plus peak-time
ordering, with a heatmap side-effect — is **not** implemented in this port.

Rationale
---------
``tradeSeq/R/cascade.R`` has a latent bug at line 122 (``dm <- colData(sce)$tradeSeq$dm``
references the unbound symbol ``sce`` instead of the function argument ``models``),
making the function non-functional in the upstream R package. The porting
plan defers the port to v2 — see ``port_reports/tradeSeq/08_validation.md``
and ``tradeseq_porting_essential_suggestions.md`` §8.
"""

from __future__ import annotations

from typing import Any

import anndata as ad

__all__ = ["cascade"]


_DEFERRAL_MESSAGE = (
    "cascade is deferred to v2; see port_reports/tradeSeq/08_validation.md for context"
)


def cascade(adata: ad.AnnData, *args: Any, **kwargs: Any) -> None:
    """Order genes by expression peaks along a lineage (stub, deferred to v2).

    Port of ``tradeSeq::cascade`` (``tradeSeq/R/cascade.R``). The full
    implementation tests pointwise first derivatives against a threshold,
    identifies significantly-increasing genes, and orders them by the first
    pseudotime at which their derivative crosses zero downward (the peak).

    .. warning::

       This is a deferral stub. Calling it always raises
       :class:`NotImplementedError`. See
       ``port_reports/tradeSeq/08_validation.md`` for the v2 plan and
       ``tradeSeq/R/cascade.R`` line 122 for the upstream latent bug that
       motivates the deferral.

    Parameters
    ----------
    adata : anndata.AnnData
        Container previously populated by :func:`tradeseq.fit_gam`. Currently
        unused; declared so the public signature is stable for v2.
    *args
        Reserved for the v2 signature (``lineage``, ``genes``, ``n_points``,
        ``epsilon``, ``derivative_threshold``, ``der_pval_threshold``,
        ``plot_heatmap``, ``cluster_heatmap``).
    **kwargs
        Reserved for the v2 signature.

    Returns
    -------
    None
        The stub never returns; it always raises.

    Raises
    ------
    NotImplementedError
        Always, with the deferral context message.

    Examples
    --------
    >>> import tradeseq
    >>> adata = tradeseq.load_paul15()
    >>> tradeseq.cascade(adata)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    NotImplementedError: cascade is deferred to v2; ...
    """
    raise NotImplementedError(_DEFERRAL_MESSAGE)
