"""Bit-exact validation of ``cr_basis`` against ``mgcv::smoothCon(s(t, bs='cr'))``.

Reference matrices are dumped from R into CSV files in /tmp by the smoke
generator inside this test (so the harness is self-contained and the
fixtures live alongside the assertion code). On systems without ``Rscript``
available, the test is skipped.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tradeseq._basis import cr_basis, _get_FS, _crspl

RSCRIPT = "/home/groups/xiaojie/nianping/Conda_Files/envs/test_tradeSeq/bin/Rscript"
TMP = Path("/tmp")


def _have_rscript() -> bool:
    return Path(RSCRIPT).exists() or shutil.which("Rscript") is not None


@pytest.fixture(scope="module", autouse=True)
def _dump_r_fixtures():
    """Run the R reference script once, writing CSVs to /tmp."""
    if not _have_rscript():
        pytest.skip("Rscript not available; cannot validate cr_basis")
    script = r"""
suppressPackageStartupMessages({library(mgcv)})

# Test 1: typical case — 50 random points in [0, 1], 6 equally spaced knots.
set.seed(7)
t  <- runif(50, 0, 1)
knots <- seq(0, 1, length.out = 6)
sm <- smoothCon(s(t, bs='cr', k = 6), data = data.frame(t = t),
                knots = list(t = knots), absorb.cons = FALSE)[[1]]
write.csv(sm$X,        "/tmp/cr_X_typical.csv",    row.names = FALSE)
write.csv(sm$S[[1]],   "/tmp/cr_S_typical.csv",    row.names = FALSE)
write.csv(data.frame(t = t),        "/tmp/cr_t_typical.csv",     row.names = FALSE)
write.csv(data.frame(knots = knots),"/tmp/cr_knots_typical.csv", row.names = FALSE)

# Test 2: with extrapolation — t with values outside [knots_min, knots_max].
set.seed(13)
t2 <- c(runif(20, 0, 1), -0.5, -0.2, 1.2, 1.5)
knots2 <- seq(0, 1, length.out = 6)
sm2 <- smoothCon(s(t, bs='cr', k = 6), data = data.frame(t = t2),
                 knots = list(t = knots2), absorb.cons = FALSE)[[1]]
write.csv(sm2$X,        "/tmp/cr_X_extrap.csv",    row.names = FALSE)
write.csv(sm2$S[[1]],   "/tmp/cr_S_extrap.csv",    row.names = FALSE)
write.csv(data.frame(t = t2),        "/tmp/cr_t_extrap.csv",     row.names = FALSE)
write.csv(data.frame(knots = knots2),"/tmp/cr_knots_extrap.csv", row.names = FALSE)

# Test 3: knots at exact data values.
t3 <- c(0, 0.2, 0.4, 0.6, 0.8, 1, 0.5, 0.15)
knots3 <- seq(0, 1, length.out = 6)
sm3 <- smoothCon(s(t, bs='cr', k = 6), data = data.frame(t = t3),
                 knots = list(t = knots3), absorb.cons = FALSE)[[1]]
write.csv(sm3$X,        "/tmp/cr_X_onknots.csv",    row.names = FALSE)
write.csv(sm3$S[[1]],   "/tmp/cr_S_onknots.csv",    row.names = FALSE)
write.csv(data.frame(t = t3),        "/tmp/cr_t_onknots.csv",     row.names = FALSE)
write.csv(data.frame(knots = knots3),"/tmp/cr_knots_onknots.csv", row.names = FALSE)

# Test 4: minimum (k = 3) basis.
set.seed(101)
t4 <- runif(20, 0, 1)
knots4 <- seq(0, 1, length.out = 3)
sm4 <- smoothCon(s(t, bs='cr', k = 3), data = data.frame(t = t4),
                 knots = list(t = knots4), absorb.cons = FALSE)[[1]]
write.csv(sm4$X,        "/tmp/cr_X_min.csv",        row.names = FALSE)
write.csv(sm4$S[[1]],   "/tmp/cr_S_min.csv",        row.names = FALSE)
write.csv(data.frame(t = t4),        "/tmp/cr_t_min.csv",         row.names = FALSE)
write.csv(data.frame(knots = knots4),"/tmp/cr_knots_min.csv",     row.names = FALSE)

# Test 5: non-uniform knot spacing.
set.seed(202)
t5 <- runif(40, 0, 1)
knots5 <- c(0, 0.1, 0.3, 0.5, 0.8, 1.0)
sm5 <- smoothCon(s(t, bs='cr', k = 6), data = data.frame(t = t5),
                 knots = list(t = knots5), absorb.cons = FALSE)[[1]]
write.csv(sm5$X,        "/tmp/cr_X_nonunif.csv",     row.names = FALSE)
write.csv(sm5$S[[1]],   "/tmp/cr_S_nonunif.csv",     row.names = FALSE)
write.csv(data.frame(t = t5),        "/tmp/cr_t_nonunif.csv",      row.names = FALSE)
write.csv(data.frame(knots = knots5),"/tmp/cr_knots_nonunif.csv",  row.names = FALSE)

# Test 6: raw (scale.penalty=FALSE) for direct getFS check.
nk6 <- 6
xk6 <- seq(0, 1, length.out = nk6)
nx6 <- 5
x6  <- c(0.1, 0.25, 0.5, 0.75, 0.9)
X6 <- rep(0, nx6 * nk6); F6 <- S6 <- rep(0, nk6 * nk6)
oo <- .C(mgcv:::C_crspl, x = as.double(x6), n = as.integer(nx6),
         xk = as.double(xk6), nk = as.integer(nk6),
         X = as.double(X6), S = as.double(S6),
         F = as.double(F6), Fsupplied = as.integer(0L))
write.csv(matrix(oo$X, nx6, nk6), "/tmp/cr_raw_X.csv", row.names = FALSE)
write.csv(matrix(oo$S, nk6, nk6), "/tmp/cr_raw_S.csv", row.names = FALSE)
write.csv(matrix(oo$F, nk6, nk6), "/tmp/cr_raw_F.csv", row.names = FALSE)
write.csv(data.frame(t = x6),     "/tmp/cr_raw_t.csv", row.names = FALSE)
write.csv(data.frame(knots = xk6),"/tmp/cr_raw_knots.csv", row.names = FALSE)
"""
    proc = subprocess.run(
        [RSCRIPT, "-e", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(f"R reference dump failed: {proc.stderr[:500]}")


def _load_csv(name: str) -> np.ndarray:
    p = TMP / name
    return np.loadtxt(p, delimiter=",", skiprows=1)


CASES = ["typical", "extrap", "onknots", "min", "nonunif"]


@pytest.mark.parametrize("case", CASES)
def test_cr_basis_matches_mgcv(case: str) -> None:
    t = _load_csv(f"cr_t_{case}.csv")
    knots = _load_csv(f"cr_knots_{case}.csv")
    X_ref = _load_csv(f"cr_X_{case}.csv")
    S_ref = _load_csv(f"cr_S_{case}.csv")

    X, S = cr_basis(t, knots)

    assert X.shape == X_ref.shape
    assert S.shape == S_ref.shape
    np.testing.assert_allclose(X, X_ref, atol=1e-10, rtol=0)
    np.testing.assert_allclose(S, S_ref, atol=1e-10, rtol=0)


def test_raw_getFS_matches_C_crspl() -> None:
    """Verify _get_FS reproduces mgcv's raw F and S (before symmetrization)."""
    knots = _load_csv("cr_raw_knots.csv")
    t = _load_csv("cr_raw_t.csv")
    F_ref = _load_csv("cr_raw_F.csv")
    S_ref = _load_csv("cr_raw_S.csv")
    X_ref = _load_csv("cr_raw_X.csv")

    F, S = _get_FS(knots)
    X = _crspl(t, knots, F)

    np.testing.assert_allclose(F, F_ref, atol=1e-10, rtol=0)
    np.testing.assert_allclose(S, S_ref, atol=1e-10, rtol=0)
    np.testing.assert_allclose(X, X_ref, atol=1e-10, rtol=0)


def test_scale_penalty_false_returns_raw_S() -> None:
    """With ``scale_penalty=False`` only symmetrization is applied to S."""
    knots = _load_csv("cr_raw_knots.csv")
    t = _load_csv("cr_raw_t.csv")
    S_raw_ref = _load_csv("cr_raw_S.csv")

    _, S = cr_basis(t, knots, scale_penalty=False)
    S_sym = (S_raw_ref + S_raw_ref.T) / 2
    np.testing.assert_allclose(S, S_sym, atol=1e-10, rtol=0)


def test_design_row_sums_to_one_inside_range() -> None:
    """Cubic-regression spline is a partition of unity inside the knot range."""
    knots = np.linspace(0.0, 1.0, 6)
    t = np.linspace(0.01, 0.99, 20)
    X, _ = cr_basis(t, knots)
    np.testing.assert_allclose(X.sum(axis=1), 1.0, atol=1e-10)


def test_design_at_knot_is_unit_vector() -> None:
    """Evaluating at a knot returns the corresponding standard basis vector."""
    knots = np.linspace(0.0, 1.0, 6)
    t = knots.copy()
    X, _ = cr_basis(t, knots)
    np.testing.assert_allclose(X, np.eye(6), atol=1e-10)


def test_penalty_is_symmetric_and_psd() -> None:
    """Penalty matrix is symmetric and positive semi-definite."""
    knots = np.linspace(0.0, 1.0, 6)
    rng = np.random.default_rng(123)
    t = rng.uniform(0.0, 1.0, size=40)
    _, S = cr_basis(t, knots)
    np.testing.assert_allclose(S, S.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(S)
    # Allow tiny negative noise from floating point.
    assert eigvals.min() > -1e-10
    # rank = nk - 2 (null space dim 2 from natural-spline boundaries).
    n_positive = int((eigvals > 1e-8).sum())
    assert n_positive == 4


def test_cr_basis_rejects_decreasing_knots() -> None:
    """Strictly-increasing knot constraint is enforced."""
    knots = np.array([0.0, 0.3, 0.2, 0.6, 0.8, 1.0])
    t = np.array([0.5])
    with pytest.raises(ValueError, match="strictly increasing"):
        cr_basis(t, knots)


def test_cr_basis_rejects_less_than_three_knots() -> None:
    knots = np.array([0.0, 1.0])
    t = np.array([0.5])
    with pytest.raises(ValueError, match="at least 3 knots"):
        cr_basis(t, knots)
