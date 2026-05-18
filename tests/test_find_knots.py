"""Bit-exact validation of ``find_knots`` against ``tradeSeq:::.findKnots``.

We dump reference outputs from R into /tmp via ``Rscript``, then check that
``find_knots`` reproduces them given the same inputs (specifically the same
``wSamp`` matrix loaded from R's seed=7 call to ``.assignCells``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradeseq._knots import find_knots, _duplicated_mask

RSCRIPT = "/home/groups/xiaojie/nianping/Conda_Files/envs/test_tradeSeq/bin/Rscript"
TMP = Path("/tmp")


def _have_rscript() -> bool:
    return Path(RSCRIPT).exists() or shutil.which("Rscript") is not None


@pytest.fixture(scope="module", autouse=True)
def _dump_r_fixtures():
    """Run the R reference script once, writing CSVs to /tmp."""
    if not _have_rscript():
        pytest.skip("Rscript not available; cannot validate find_knots")
    script = r"""
suppressPackageStartupMessages({
  library(tradeSeq)
  library(slingshot)
})
data(crv, package = "tradeSeq")
ps <- slingPseudotime(crv, na = FALSE)
cw <- slingCurveWeights(crv)

set.seed(7)
wSamp <- tradeSeq:::.assignCells(cw)

# Primary case: nknots = 6, 2 lineages.
kl <- tradeSeq:::.findKnots(6, ps, wSamp)
write.csv(data.frame(t1 = kl$t1, t2 = kl$t2),
          "/tmp/r_knots_n6.csv", row.names = FALSE)
write.csv(wSamp, "/tmp/r_wsamp_seed7.csv", row.names = FALSE)
write.csv(ps,    "/tmp/r_ps_seed7.csv",    row.names = FALSE)

# n = 4 knots.
kl4 <- tradeSeq:::.findKnots(4, ps, wSamp)
write.csv(data.frame(t1 = kl4$t1, t2 = kl4$t2),
          "/tmp/r_knots_n4.csv", row.names = FALSE)

# n = 10 knots.
kl10 <- tradeSeq:::.findKnots(10, ps, wSamp)
write.csv(data.frame(t1 = kl10$t1, t2 = kl10$t2),
          "/tmp/r_knots_n10.csv", row.names = FALSE)

# Single-lineage case.
ps1     <- ps[, 1, drop = FALSE]
wSamp1  <- matrix(1, nrow = nrow(ps1), ncol = 1)
kl1 <- tradeSeq:::.findKnots(6, ps1, wSamp1)
write.csv(data.frame(t1 = kl1$t1),  "/tmp/r_knots_one_lineage.csv",
          row.names = FALSE)
write.csv(ps1, "/tmp/r_ps_one_lineage.csv",    row.names = FALSE)
write.csv(wSamp1, "/tmp/r_wsamp_one_lineage.csv", row.names = FALSE)

# Pathological fallback 1 trigger: most pseudotime values squeezed at 0.5,
# distinct values only on a tiny minority. Re-quantile on the longest
# lineage covers the issue.
n <- 30
ps_squeeze <- matrix(0.5, nrow = n, ncol = 2)
ps_squeeze[1, 1] <- 0
ps_squeeze[2, 1] <- 1
ps_squeeze[3, 2] <- 0.6
wSamp_sq <- matrix(0, nrow = n, ncol = 2)
wSamp_sq[, 1] <- c(1, 1, 0, rep(1, n - 3))
wSamp_sq[, 2] <- 1 - wSamp_sq[, 1]
kl_s <- tradeSeq:::.findKnots(6, ps_squeeze, wSamp_sq)
write.csv(data.frame(t1 = kl_s$t1, t2 = kl_s$t2),
          "/tmp/r_knots_squeeze.csv", row.names = FALSE)
write.csv(ps_squeeze, "/tmp/r_ps_squeeze.csv", row.names = FALSE)
write.csv(wSamp_sq,   "/tmp/r_wsamp_squeeze.csv", row.names = FALSE)

# Fallback 2 trigger: nearly all pseudotime at one value; only one cell at a
# higher value to provide a different 100% quantile. After fallback 1, knots
# 2..5 still duplicate; mean-of-neighbours then resolves it.
n2 <- 100
ps_dup <- matrix(0, nrow = n2, ncol = 2)
ps_dup[, 1] <- c(rep(0.5, n2 - 1), 1.0)
ps_dup[, 2] <- 0.5
wSamp_dup <- matrix(0, nrow = n2, ncol = 2)
wSamp_dup[, 1] <- 1
wSamp_dup[, 2] <- 0
kl_d <- suppressWarnings(tradeSeq:::.findKnots(6, ps_dup, wSamp_dup))
write.csv(data.frame(t1 = kl_d$t1, t2 = kl_d$t2),
          "/tmp/r_knots_dup.csv", row.names = FALSE)
write.csv(ps_dup,   "/tmp/r_ps_dup.csv",    row.names = FALSE)
write.csv(wSamp_dup,"/tmp/r_wsamp_dup.csv", row.names = FALSE)

# Fallback 3 trigger: lineage 1 entirely at 0.5 (and one cell at 0/1
# but assigned to a row that w_samp doesn't pick). When the longest lineage
# is fully degenerate, the only escape is the evenly-spaced linspace.
ps_uni <- matrix(0.5, nrow = 10, ncol = 2)
ps_uni[1, 1] <- 0; ps_uni[10, 1] <- 1
wSamp_uni <- cbind(c(1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                   c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1))
kl_u <- suppressWarnings(tradeSeq:::.findKnots(6, ps_uni, wSamp_uni))
write.csv(data.frame(t1 = kl_u$t1, t2 = kl_u$t2),
          "/tmp/r_knots_uni.csv", row.names = FALSE)
write.csv(ps_uni,    "/tmp/r_ps_uni.csv",    row.names = FALSE)
write.csv(wSamp_uni, "/tmp/r_wsamp_uni.csv", row.names = FALSE)

# Boundary case 1 (gap #5 — fromLast=TRUE marks p == 0): pseudotime such that
# the quantile-based knotLocs reduce to (0.5, 0.5, 0.5, 0.5, 1.0, 1.0) i.e.
# a duplicate run that touches BOTH the first and last positions. After R's
# fromLast remap the dup mask becomes (T, T, T, F, T, F), so the very first
# position is a duplicate. R's ``knotLocs[which(dupId) - 1]`` would be
# ``knotLocs[0]`` (numeric(0)) and is silently dropped from ``c(...)`` before
# ``mean()``. Python's naive ``knot_locs[p - 1]`` would wrap to ``knot_locs[-1]``,
# producing a different replacement value — this fixture exercises the guard.
ps_bd <- matrix(0.0, nrow = 6, ncol = 2)
ps_bd[, 1] <- c(0.5, 0.5, 0.5, 0.5, 1.0, 1.0)
ps_bd[, 2] <- 0.5
wSamp_bd <- matrix(0, nrow = 6, ncol = 2)
wSamp_bd[, 1] <- 1
kl_bd <- suppressWarnings(tradeSeq:::.findKnots(6, ps_bd, wSamp_bd))
write.csv(ps_bd,    "/tmp/r_ps_bd_p0.csv", row.names = FALSE)
write.csv(wSamp_bd, "/tmp/r_wsamp_bd_p0.csv", row.names = FALSE)
write.csv(data.frame(t1 = kl_bd$t1, t2 = kl_bd$t2),
          "/tmp/r_knots_bd_p0.csv", row.names = FALSE)

# Boundary case 2 (gap #5 — single duplicate at the end). Designed so that
# fallback 2 RESOLVES the duplicate (no fallback 3) and the resolved value
# survives into the final knot vector — this is the cleanest end-to-end
# observable test of fallback 2's mean-of-neighbours arithmetic.
ps_se <- matrix(0.0, nrow = 6, ncol = 2)
ps_se[, 1] <- c(0.1, 0.3, 0.5, 0.7, 0.9, 0.9)
ps_se[, 2] <- 0.5
wSamp_se <- matrix(0, nrow = 6, ncol = 2)
wSamp_se[, 1] <- 1
kl_se <- suppressWarnings(tradeSeq:::.findKnots(6, ps_se, wSamp_se))
write.csv(ps_se,    "/tmp/r_ps_bd_single_dup_end.csv", row.names = FALSE)
write.csv(wSamp_se, "/tmp/r_wsamp_bd_single_dup_end.csv", row.names = FALSE)
write.csv(data.frame(t1 = kl_se$t1, t2 = kl_se$t2),
          "/tmp/r_knots_bd_single_dup_end.csv", row.names = FALSE)
"""
    proc = subprocess.run(
        [RSCRIPT, "-e", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        pytest.skip(f"R reference dump failed: {proc.stderr[:500]}")


def _load_csv(name: str) -> np.ndarray:
    p = TMP / name
    return np.loadtxt(p, delimiter=",", skiprows=1)


def test_find_knots_primary_n6() -> None:
    """Primary quantile path on the crv dataset (nknots=6, 2 lineages)."""
    ps = _load_csv("r_ps_seed7.csv")
    w_samp = _load_csv("r_wsamp_seed7.csv")
    ref = pd.read_csv(TMP / "r_knots_n6.csv")
    out = find_knots(6, ps, w_samp)
    assert set(out.keys()) == {"t1", "t2"}
    np.testing.assert_array_almost_equal(out["t1"], ref["t1"].to_numpy(),
                                         decimal=12)
    np.testing.assert_array_almost_equal(out["t2"], ref["t2"].to_numpy(),
                                         decimal=12)


def test_find_knots_n4() -> None:
    ps = _load_csv("r_ps_seed7.csv")
    w_samp = _load_csv("r_wsamp_seed7.csv")
    ref = pd.read_csv(TMP / "r_knots_n4.csv")
    out = find_knots(4, ps, w_samp)
    np.testing.assert_array_almost_equal(out["t1"], ref["t1"].to_numpy(),
                                         decimal=12)
    np.testing.assert_array_almost_equal(out["t2"], ref["t2"].to_numpy(),
                                         decimal=12)


def test_find_knots_n10() -> None:
    ps = _load_csv("r_ps_seed7.csv")
    w_samp = _load_csv("r_wsamp_seed7.csv")
    ref = pd.read_csv(TMP / "r_knots_n10.csv")
    out = find_knots(10, ps, w_samp)
    np.testing.assert_array_almost_equal(out["t1"], ref["t1"].to_numpy(),
                                         decimal=12)


def test_find_knots_single_lineage() -> None:
    ps = pd.read_csv(TMP / "r_ps_one_lineage.csv").to_numpy()
    w_samp = pd.read_csv(TMP / "r_wsamp_one_lineage.csv").to_numpy()
    ref = pd.read_csv(TMP / "r_knots_one_lineage.csv")
    out = find_knots(6, ps, w_samp)
    assert list(out.keys()) == ["t1"]
    np.testing.assert_array_almost_equal(out["t1"], ref["t1"].to_numpy(),
                                         decimal=12)


def test_find_knots_fallback1_squeeze() -> None:
    """Pathological squeezed pseudotime triggers fallback 1 (re-quantile on l1)."""
    ps = pd.read_csv(TMP / "r_ps_squeeze.csv").to_numpy()
    w_samp = pd.read_csv(TMP / "r_wsamp_squeeze.csv").to_numpy()
    ref = pd.read_csv(TMP / "r_knots_squeeze.csv")
    out = find_knots(6, ps, w_samp)
    np.testing.assert_array_almost_equal(out["t1"], ref["t1"].to_numpy(),
                                         decimal=12)
    np.testing.assert_array_almost_equal(out["t2"], ref["t2"].to_numpy(),
                                         decimal=12)


def test_find_knots_fallback2_meanofneighbours() -> None:
    """nearly-singular pseudotime triggers fallback 2 (mean-of-neighbours)."""
    ps = pd.read_csv(TMP / "r_ps_dup.csv").to_numpy()
    w_samp = pd.read_csv(TMP / "r_wsamp_dup.csv").to_numpy()
    ref = pd.read_csv(TMP / "r_knots_dup.csv")
    out = find_knots(6, ps, w_samp)
    np.testing.assert_array_almost_equal(out["t1"], ref["t1"].to_numpy(),
                                         decimal=12)
    np.testing.assert_array_almost_equal(out["t2"], ref["t2"].to_numpy(),
                                         decimal=12)


def test_find_knots_fallback3_evenly_spaced() -> None:
    """Degenerate longest lineage triggers fallback 3 (linspace)."""
    ps = pd.read_csv(TMP / "r_ps_uni.csv").to_numpy()
    w_samp = pd.read_csv(TMP / "r_wsamp_uni.csv").to_numpy()
    ref = pd.read_csv(TMP / "r_knots_uni.csv")
    out = find_knots(6, ps, w_samp)
    np.testing.assert_array_almost_equal(out["t1"], ref["t1"].to_numpy(),
                                         decimal=12)
    np.testing.assert_array_almost_equal(out["t2"], ref["t2"].to_numpy(),
                                         decimal=12)


def test_find_knots_endpoint_snap_clamps() -> None:
    """First/last knots equal min/max of tAll."""
    ps = _load_csv("r_ps_seed7.csv")
    w_samp = _load_csv("r_wsamp_seed7.csv")
    out = find_knots(6, ps, w_samp)
    # Build tAll the way R does
    sel = np.argmax(w_samp.astype(bool), axis=1)
    t_all = ps[np.arange(ps.shape[0]), sel]
    assert out["t1"][0] == pytest.approx(float(np.min(t_all)), abs=1e-15)
    assert out["t1"][-1] == pytest.approx(float(np.max(t_all)), abs=1e-15)


def test_duplicated_mask_default() -> None:
    """``_duplicated_mask`` matches R's ``duplicated`` (first occurrence kept)."""
    a = np.array([1.0, 2.0, 2.0, 3.0, 2.0])
    mask = _duplicated_mask(a, from_last=False)
    np.testing.assert_array_equal(mask, [False, False, True, False, True])


def test_duplicated_mask_from_last() -> None:
    """``_duplicated_mask(..., fromLast=TRUE)`` keeps the last occurrence."""
    a = np.array([1.0, 2.0, 2.0, 3.0, 2.0])
    mask = _duplicated_mask(a, from_last=True)
    np.testing.assert_array_equal(mask, [False, True, True, False, False])


def test_find_knots_rejects_dim_mismatch() -> None:
    ps = np.zeros((5, 2))
    w_samp = np.zeros((5, 1))
    with pytest.raises(ValueError, match="identical shape"):
        find_knots(4, ps, w_samp)


def test_find_knots_rejects_1d_pseudotime() -> None:
    with pytest.raises(ValueError, match="2-D"):
        find_knots(4, np.zeros(5), np.zeros(5))


def test_find_knots_returns_independent_arrays_per_lineage() -> None:
    """The dict returns one knot array per lineage and they must be independent
    so callers can mutate them safely."""
    ps = _load_csv("r_ps_seed7.csv")
    w_samp = _load_csv("r_wsamp_seed7.csv")
    out = find_knots(6, ps, w_samp)
    assert out["t1"] is not out["t2"]
    out["t1"][0] = -99.0
    assert out["t2"][0] != -99.0


# -----------------------------------------------------------------------------
# Gap #5: fallback-2 mean-of-neighbours boundary regression tests
# -----------------------------------------------------------------------------


def test_find_knots_boundary_fromlast_p0() -> None:
    """Fallback 2 with fromLast=TRUE and a duplicate at Python position 0.

    R's ``mean(c(knotLocs[which(dupId)-1], knotLocs[which(dupId)+1]))`` silently
    drops the empty-vector left neighbour at the very first position; Python's
    naive ``knot_locs[p-1]`` wraps to the last element. The fix collects only
    in-range neighbours so the grand mean matches R's. End-to-end this case
    still falls through to fallback 3 (linspace), but the result must agree
    with R bit-by-bit.
    """
    ps = pd.read_csv(TMP / "r_ps_bd_p0.csv").to_numpy()
    w_samp = pd.read_csv(TMP / "r_wsamp_bd_p0.csv").to_numpy()
    ref = pd.read_csv(TMP / "r_knots_bd_p0.csv")
    out = find_knots(6, ps, w_samp)
    np.testing.assert_array_almost_equal(out["t1"], ref["t1"].to_numpy(),
                                         decimal=12)
    np.testing.assert_array_almost_equal(out["t2"], ref["t2"].to_numpy(),
                                         decimal=12)


def test_find_knots_boundary_fromlast_p0_indexerror_free() -> None:
    """``find_knots`` must not raise even when the dup mask includes p == 0.

    Prior to the fix the function relied on Python's negative-index wrap to
    avoid an IndexError; with the explicit guard ``p - 1 >= 0`` we no longer
    accidentally consume ``knot_locs[-1]``, but the function must still
    return without raising on the same input.
    """
    ps = pd.read_csv(TMP / "r_ps_bd_p0.csv").to_numpy()
    w_samp = pd.read_csv(TMP / "r_wsamp_bd_p0.csv").to_numpy()
    # Just run; failure mode would be either IndexError or a NaN in the output.
    out = find_knots(6, ps, w_samp)
    assert not np.any(np.isnan(out["t1"]))


def test_find_knots_fallback2_single_dup_end() -> None:
    """Fallback 2 resolves a single dup at the last position (no fallback 3).

    With ``knotLocs = (0.1, 0.3, 0.5, 0.7, 0.9, 0.9)`` after fallback 1, R
    triggers ``fromLast=TRUE`` (the trailing position is the only duplicate),
    remarks the second-to-last position as the duplicate, then replaces it
    with the mean of its two neighbours ``(0.7 + 0.9) / 2 = 0.8``. The final
    knots ``(0.1, 0.3, 0.5, 0.7, 0.8, 0.9)`` exercise the mean-of-neighbours
    arithmetic end-to-end (fallback 3 does NOT fire).
    """
    ps = pd.read_csv(TMP / "r_ps_bd_single_dup_end.csv").to_numpy()
    w_samp = pd.read_csv(TMP / "r_wsamp_bd_single_dup_end.csv").to_numpy()
    ref = pd.read_csv(TMP / "r_knots_bd_single_dup_end.csv")
    out = find_knots(6, ps, w_samp)
    np.testing.assert_array_almost_equal(out["t1"], ref["t1"].to_numpy(),
                                         decimal=12)
    # The smoking-gun assertion: the intermediate 0.8 from mean(0.7, 0.9)
    # survives. This is the actual mean-of-neighbours value.
    assert out["t1"][4] == pytest.approx(0.8, abs=1e-12)


def test_find_knots_boundary_no_indexerror_p_eq_n_minus_1() -> None:
    """Defensive: the symmetric guard ``p + 1 < n`` prevents an IndexError if
    a duplicate at the very last index ever reached the mean-of-neighbours
    branch (the actual algorithm flow always remaps this case via
    ``fromLast=TRUE``, but we want the boundary code to be safe nonetheless).
    """
    # The all-degenerate case exercises duplicate handling at the boundary.
    n = 8
    ps = np.full((n, 1), 0.5)
    w_samp = np.ones((n, 1))
    # Should not raise (the fix removed the implicit dependency on Python's
    # negative-index wrap behaviour).
    out = find_knots(6, ps, w_samp)
    assert out["t1"].shape == (6,)
