"""Cell-to-lineage assignment (R source: tradeSeq/R/fitGAM.R:1-30).

Port of the internal helper ``tradeSeq:::.assignCells``. Given a non-negative
``cell_weights`` matrix of shape ``(n_cells, n_lineages)``, draw a single
lineage assignment per cell via a multinomial sample whose probability vector
is the row-normalized weights. Returns a 0/1 indicator matrix with exactly one
``1`` per row.

Cross-language seed parity (R ``stats::rmultinom`` vs NumPy
``Generator.multinomial``) is unattainable. To validate downstream consumers
against an R-side draw, callers may pass a precomputed indicator matrix via
the private ``_w_samp`` kwarg; it is shape-checked and returned unchanged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["assign_cells"]


def assign_cells(
    cell_weights: Optional[np.ndarray],
    *,
    rng: Optional[np.random.Generator] = None,
    _w_samp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample one lineage per cell from row-normalized ``cell_weights``.

    Mirrors the R function ``tradeSeq:::.assignCells`` line-for-line.

    Parameters
    ----------
    cell_weights
        Non-negative ``(n_cells, n_lineages)`` matrix (or 1-D ``(n_cells,)``
        vector treated as a single lineage). The R source also tolerates the
        case where ``dim(cell_weights)`` is ``NULL``; we mirror that by
        accepting both 1-D arrays and ``None`` (which yields a single-column
        matrix of ones — to match the R branch that hits no error path).
    rng
        Optional NumPy ``Generator``. Defaults to ``np.random.default_rng()``
        when sampling is actually required. Reproducible sampling is intended
        to be verified by callers that pass ``_w_samp`` directly.
    _w_samp
        Private validation hook: a precomputed ``(n_cells, n_lineages)``
        indicator matrix obtained from an R-side dump. When supplied, the
        function performs the shape check and returns it unchanged so the
        caller can verify downstream behaviour against the R draw without
        attempting impossible cross-language seed parity.

    Returns
    -------
    numpy.ndarray
        ``(n_cells, n_lineages)`` integer indicator matrix with exactly one
        ``1`` per row.

    Raises
    ------
    ValueError
        If any cell has total weight zero (mirrors R ``stop(...)``), or if
        ``_w_samp`` is supplied with a mismatched shape.
    """
    # cell_weights == NULL branch: R never hits this with the package's own
    # callers, but the documented contract is "matrix(1, length, 1)". We
    # mirror it for forward-compatibility with R behaviour. The R code only
    # reaches `matrix(1, ..., 1)` when cell_weights is a vector with no zeros
    # — we preserve that semantics for the 1-D path below.
    if cell_weights is None:
        raise ValueError("Some cells have no positive cell weights.")

    cw = np.asarray(cell_weights)

    # R: `is.null(dim(cellWeights))` — true for vectors.
    if cw.ndim == 1:
        if np.any(cw == 0):
            raise ValueError("Some cells have no positive cell weights.")
        result = np.ones((cw.shape[0], 1), dtype=np.int64)
        if _w_samp is not None:
            _validate_wsamp(_w_samp, result.shape)
            return np.asarray(_w_samp)
        return result

    if cw.ndim != 2:
        raise ValueError(
            f"cell_weights must be 1-D or 2-D, got ndim={cw.ndim}."
        )

    row_sums = cw.sum(axis=1)
    if np.any(row_sums == 0):
        raise ValueError("Some cells have no positive cell weights.")

    n_cells, n_lineages = cw.shape

    if _w_samp is not None:
        _validate_wsamp(_w_samp, (n_cells, n_lineages))
        return np.asarray(_w_samp)

    # R: `sweep(cellWeights, 1, FUN="/", STATS=apply(cellWeights, 1, sum))`
    norm_weights = cw / row_sums[:, None]

    if rng is None:
        rng = np.random.default_rng()

    # R: apply(normWeights, 1, function(prob) rmultinom(n=1, prob=prob, size=1))
    # Produces a (n_lineages, n_cells) matrix in R; transposed below.
    # NumPy's `Generator.multinomial(n=1, pvals=...)` is vectorisable across
    # the leading axis.
    w_samp = rng.multinomial(n=1, pvals=norm_weights, size=n_cells)
    # If only one lineage, R reshapes to (n_cells, 1). NumPy already returns
    # the correct (n_cells, n_lineages) shape, so no further work is needed.
    return w_samp.astype(np.int64)


def _validate_wsamp(w_samp: np.ndarray, expected_shape: tuple[int, int]) -> None:
    """Validate the private ``_w_samp`` hook matches expected shape."""
    arr = np.asarray(w_samp)
    if arr.shape != expected_shape:
        raise ValueError(
            f"_w_samp has shape {arr.shape}; expected {expected_shape}."
        )
