"""Knot placement for tradeSeq smoothers (R source: fitGAM.R:104-181 ``.findKnots``).

The quantile placement algorithm with 3 fallback layers and per-lineage
endpoint snapping. The returned dictionary mirrors R's ``knotList`` which is
``lapply(seq_len(ncol(pseudotime)), function(i) knots)`` — the same knot
vector repeated for every lineage (mgcv shares the basis across lineages via
the ``id=1`` argument inside ``fitGAM``'s ``smoothForm``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["find_knots"]


def _duplicated_mask(a: np.ndarray, from_last: bool = False) -> np.ndarray:
    """Reproduce R's ``duplicated`` semantics.

    ``duplicated(x)`` marks every element that is equal to an *earlier*
    element as ``TRUE`` (first occurrence keeps ``FALSE``).
    ``duplicated(x, fromLast=TRUE)`` flips the direction (last occurrence
    keeps ``FALSE``).
    """
    n = a.shape[0]
    out = np.zeros(n, dtype=bool)
    if from_last:
        seen: set[float] = set()
        for i in range(n - 1, -1, -1):
            v = float(a[i])
            if v in seen:
                out[i] = True
            else:
                seen.add(v)
    else:
        seen = set()
        for i in range(n):
            v = float(a[i])
            if v in seen:
                out[i] = True
            else:
                seen.add(v)
    return out


def _quantile_type7(a: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """R ``stats::quantile`` with ``type=7`` (the default).

    Equivalent to ``numpy.quantile(..., method='linear')`` which is itself
    the numpy default. Wrapped explicitly so the contract is documented.
    """
    return np.quantile(a, probs, method="linear")


def find_knots(
    n_knots: int,
    pseudotime: np.ndarray,
    w_samp: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pick ``n_knots`` quantile-spaced knots on the pseudotime vector.

    Parameters
    ----------
    n_knots
        Number of knots to place (``>= 2``; ``mgcv`` requires ``>= 3`` for
        ``bs='cr'``, but this function does not enforce that — leave to
        caller).
    pseudotime
        Pseudotime matrix, shape ``(n_cells, n_lineages)``. NaN entries are
        not allowed.
    w_samp
        Sampled cell-to-lineage assignment from ``assign_cells``; same shape
        as ``pseudotime``. Each row has exactly one ``1``.

    Returns
    -------
    knot_list
        Dictionary ``{"t1": knots, "t2": knots, ...}`` with one entry per
        lineage; the same ``knots`` array is repeated as values (R returns
        the same vector per lineage column because the GAM formula shares
        the smoother basis across lineages via ``id=1``).

    Notes
    -----
    The algorithm (``fitGAM.R:104-181``) is:

      1. Build ``tAll`` of length ``n_cells`` by picking, for each cell,
         the pseudotime entry of the lineage column where ``w_samp == 1``.
      2. Primary placement::

            knotLocs <- quantile(tAll, probs=(0:(nknots-1))/(nknots-1))
      3. If any duplicate, fallback 1: re-quantile on the longest lineage
         (``t1[l1==1]``).
      4. If still duplicate, fallback 2: replace duplicate(s) with the mean
         of immediate neighbours. The last position uses ``duplicated(...,
         fromLast=TRUE)`` to pick the run-tail.
      5. If still duplicate, fallback 3: ``seq(min(tAll), max(tAll),
         length=nknots)``.
      6. Endpoint snap: for every lineage beyond the first, replace the
         knot nearest to the lineage's max pseudotime by that max — unless
         the max is already a knot.
      7. Final clamps: ``knots[0] = min(tAll)`` and
         ``knots[nknots - 1] = max(tAll)``.
    """
    pseudotime = np.asarray(pseudotime, dtype=np.float64)
    w_samp = np.asarray(w_samp)

    if pseudotime.ndim != 2:
        raise ValueError(
            f"pseudotime must be 2-D, got shape {pseudotime.shape}"
        )
    if w_samp.shape != pseudotime.shape:
        raise ValueError(
            "pseudotime and w_samp must have identical shape; "
            f"got {pseudotime.shape} vs {w_samp.shape}"
        )

    n_cells, n_lineages = pseudotime.shape

    # Build tAll: per-row, take the pseudotime entry of the lineage whose
    # w_samp == 1. R: tAll[ii] <- pseudotime[ii, which(as.logical(wSamp[ii,]))]
    # NB: assign_cells guarantees exactly one '1' per row.
    selected = np.argmax(np.asarray(w_samp, dtype=bool), axis=1)  # (n_cells,)
    t_all = pseudotime[np.arange(n_cells), selected]  # (n_cells,)

    probs = np.arange(n_knots, dtype=np.float64) / (n_knots - 1)

    knot_locs = _quantile_type7(t_all, probs)

    # First lineage column / first weight column.
    t1 = pseudotime[:, 0]
    l1 = (w_samp[:, 0] == 1).astype(np.int64)

    if _duplicated_mask(knot_locs).any():
        # Fallback 1: re-quantile on the longest (first) lineage's assigned
        # cells.
        knot_locs = _quantile_type7(t1[l1 == 1], probs)

        if _duplicated_mask(knot_locs).any():
            # Fallback 2: mean-of-neighbours replacement.
            dup_id = _duplicated_mask(knot_locs)
            if int(np.max(np.where(dup_id)[0])) == knot_locs.shape[0] - 1:
                # The duplicate run extends to the last knot. R re-marks
                # duplicates using fromLast=TRUE so the run-tail is replaced
                # by the mean of its immediate neighbours.
                dup_id = _duplicated_mask(knot_locs, from_last=True)
            dup_positions = np.where(dup_id)[0]
            # R: knotLocs[dupId] <- mean(c(knotLocs[which(dupId) - 1],
            #                              knotLocs[which(dupId) + 1]))
            # R indexing with 0 yields ``numeric(0)`` (silently dropped from
            # c(...)); indexing with ``length + 1`` yields ``NA``. Mirror
            # this by collecting ONLY the in-range neighbour values per
            # duplicate position:
            #   * p == 0       -> drop the left neighbour (R's numeric(0))
            #   * p == n - 1   -> drop the right neighbour (R's NA dropped
            #                     by mean's default behaviour — but R's
            #                     actual behaviour with NA in c() gives
            #                     mean=NA; in practice the algorithm's
            #                     fromLast remap protects p==n-1 from ever
            #                     being a duplicate, so this branch is
            #                     defensive only).
            # The default-duplicated branch never produces p == 0 (the
            # first element is always a "keep"); the fromLast=TRUE branch
            # never produces p == n - 1 (the last element is the "keep").
            n_kl = knot_locs.shape[0]
            neighbour_vals: list[float] = []
            for p in dup_positions:
                if p - 1 >= 0:
                    neighbour_vals.append(float(knot_locs[p - 1]))
                if p + 1 < n_kl:
                    neighbour_vals.append(float(knot_locs[p + 1]))
            # R's ``mean(c(...))`` collapses the concatenated vector into a
            # single scalar — every duplicate position receives the SAME
            # replacement value (the grand mean over all in-range
            # neighbours).
            if len(neighbour_vals) > 0:
                replacement = float(np.mean(neighbour_vals))
                knot_locs[dup_positions] = replacement

        if _duplicated_mask(knot_locs).any():
            # Fallback 3: evenly spaced knots.
            knot_locs = np.linspace(
                float(np.min(t_all)), float(np.max(t_all)), n_knots
            )

    # Endpoint snap step.
    # In R, max(numeric(0)) returns -Inf with a warning. We match that
    # semantic exactly so the rest of the algorithm behaves the same when a
    # lineage column has no assigned cells in w_samp (the final clamps then
    # overwrite any -Inf knots).
    if n_lineages == 1:
        max_t = np.array([np.max(pseudotime[:, 0])], dtype=np.float64)
    else:
        max_t_list: list[float] = []
        for jj in range(1, n_lineages):
            picks = pseudotime[:, jj][w_samp[:, jj] == 1]
            if picks.size == 0:
                max_t_list.append(-np.inf)  # mirror R's max(numeric(0))
            else:
                max_t_list.append(float(np.max(picks)))
        max_t = np.array(max_t_list, dtype=np.float64)

    # Replace the nearest knot for each lineage-end NOT already at a knot.
    if not np.all(np.isin(max_t, knot_locs)):
        max_t_to_place = max_t[~np.isin(max_t, knot_locs)]
        # R: replaceId <- vapply(maxT, function(ll){which.min(abs(ll-knotLocs))})
        # Note: R's which.min returns the first index of the minimum.
        for ll in max_t_to_place:
            diffs = np.abs(ll - knot_locs)
            replace_id = int(np.argmin(diffs))
            knot_locs[replace_id] = ll
        # The R code emits a warning if not all endpoints could be placed;
        # since we replace in-place over a possibly shorter knot vector,
        # this can still leave some maxT off-grid (e.g. when two maxT values
        # snap to the same knot). We do not raise — match R's silent
        # 'warning' which is suppressed by the caller. Caller documents.

    # Final clamps.
    knot_locs[0] = float(np.min(t_all))
    knot_locs[n_knots - 1] = float(np.max(t_all))

    out: dict[str, np.ndarray] = {
        f"t{ii + 1}": knot_locs.copy() for ii in range(n_lineages)
    }
    return out
