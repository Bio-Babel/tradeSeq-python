"""Cubic-regression spline basis (R source: mgcv-1.9.3/R/smooth.r:smooth.construct.cr.smooth.spec).

The natural cubic regression spline (`bs='cr'` in mgcv) is built directly from
the C routine ``mgcv-1.9.3/src/mgcv.c:crspl`` (called via
``.C(C_crspl, ...)``) and rescaled per ``smoothCon``'s ``scale.penalty=TRUE``
branch, which is the default ``mgcv::gam`` uses.

Algorithm (Wood, "Generalized Additive Models", 2nd ed §5.3.1):
    * Given strictly increasing knot sequence ``xk`` of length ``nk``:
      build the ``nk-2`` x ``nk`` second-difference matrix ``D`` and the
      ``nk-2`` x ``nk-2`` tridiagonal ``B`` matrix.
    * Solve ``B y = D`` for ``y`` (LAPACK ``dptsv``).
    * Pad ``y`` with a zero row at top and bottom -> ``F`` (the matrix mapping
      knot values to second derivatives).
    * The penalty matrix is ``S = D' F`` (equivalently ``D' B^{-1} D``).
    * For each query point ``x[i]``, locate the interval ``[xk[j], xk[j+1]]``
      and evaluate the cubic-spline basis via Wood's eq 5.7.

After construction ``mgcv::smoothCon`` symmetrizes ``S <- (S + S') / 2`` then
rescales ``S <- S / (||S||_1 / ||X||_inf^2)`` (``scale.penalty=TRUE``). We
reproduce both steps so the returned ``S`` matches ``mgcv::gam``'s internal
penalty exactly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cr_basis"]


def _get_FS(xk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute F (knot-to-second-derivative map) and raw penalty S = D' F.

    Mirror of ``mgcv-1.9.3/src/mgcv.c:getFS``.

    Parameters
    ----------
    xk
        Strictly increasing knot sequence, length ``n >= 3``.

    Returns
    -------
    F
        ``(n, n)`` array. ``F[1:n-1, :]`` is ``B^{-1} D`` (each column);
        top and bottom rows are zero (natural-spline boundary).
    S
        ``(n, n)`` raw penalty matrix ``D' B^{-1} D``.
    """
    n = xk.shape[0]
    if n < 3:
        raise ValueError(f"cr_basis requires at least 3 knots, got {n}")
    n1 = n - 1
    n2 = n - 2

    h = np.diff(xk)  # length n-1
    if np.any(h <= 0):
        raise ValueError("Knots must be strictly increasing.")

    # D is (n-2) x n: D[i, i] = 1/h[i], D[i, i+1] = -1/h[i]-1/h[i+1],
    # D[i, i+2] = 1/h[i+1].
    D = np.zeros((n2, n), dtype=np.float64)
    for i in range(n2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -1.0 / h[i] - 1.0 / h[i + 1]
        D[i, i + 2] = 1.0 / h[i + 1]

    # B is (n-2) x (n-2) symmetric tridiagonal:
    # diag B[i,i] = (h[i] + h[i+1])/3, super-diag B[i, i+1] = h[i+1]/6.
    ldB = (h[:n2] + h[1:n2 + 1]) / 3.0  # length n-2
    sdB = h[1:n2] / 6.0  # length n-3

    # Solve B Y = D, columns of D as RHS. We build the dense tridiag and use
    # numpy.linalg.solve so behaviour matches LAPACK ``dptsv`` numerically.
    B = np.diag(ldB) + np.diag(sdB, k=1) + np.diag(sdB, k=-1)
    # D is (n-2, n). After dptsv, the same buffer holds Y = B^{-1} D.
    Y = np.linalg.solve(B, D)  # shape (n-2, n)

    # Assemble F (shape (n, n)): zero first and last columns; the interior
    # columns are Y transposed (Y.T -> shape (n, n-2)).
    # Per the C source (mgcv.c:185-192): Di walks column-major through the
    # solved D buffer, writing into F[i, j+1] where i is the data-knot
    # index and j is the column index in Y. That gives F[i, j+1] = Y[j, i],
    # i.e. F[:, 1:n-1] = Y.T.
    F = np.zeros((n, n), dtype=np.float64)
    F[:, 1:n1] = Y.T

    # S = D' Y in math; the C code (mgcv.c:194-214) builds this row-by-row.
    S = D.T @ Y  # shape (n, n)
    return F, S


def _crspl(
    x: np.ndarray, xk: np.ndarray, F: np.ndarray
) -> np.ndarray:
    """Evaluate cubic-regression spline basis at points ``x``.

    Mirror of the ``crspl`` evaluation loop (no ``Fsupplied`` branch — we
    always supply ``F`` from ``_get_FS``).

    Parameters
    ----------
    x
        Query points, shape ``(nx,)``.
    xk
        Knot positions (strictly increasing), shape ``(nk,)``.
    F
        ``(nk, nk)`` knot-to-second-derivative map from ``_get_FS``.

    Returns
    -------
    X
        Design matrix, shape ``(nx, nk)``.
    """
    nx = x.shape[0]
    nk = xk.shape[0]
    kmin = xk[0]
    kmax = xk[-1]
    X = np.zeros((nx, nk), dtype=np.float64)

    # mgcv keeps a sliding ``j`` index across rows for direct neighbour
    # search; we always use bisection since the per-loop optimisation is
    # numerical (not algebraic). Result is bit-identical given the floating
    # point arithmetic order matches the C arithmetic (it does — we use the
    # same closed-form per-point formulas).
    for i in range(nx):
        xi = float(x[i])
        if xi < kmin:
            # extrapolate below
            h = xk[1] - kmin
            xik = xi - kmin
            cjm = -xik * h / 3.0
            cjp = -xik * h / 6.0
            # X[i, k] = cjm * F[k, 0] + cjp * F[k, 1]
            for k in range(nk):
                X[i, k] = cjm * F[k, 0] + cjp * F[k, 1]
            X[i, 0] += 1.0 - xik / h
            X[i, 1] += xik / h
        elif xi > kmax:
            # extrapolate above
            j = nk - 1
            h = kmax - xk[j - 1]
            xik = xi - kmax
            cjm = xik * h / 6.0
            cjp = xik * h / 3.0
            # X[i, k] = cjm * F[k, j-1] + cjp * F[k, j]
            for k in range(nk):
                X[i, k] = cjm * F[k, j - 1] + cjp * F[k, j]
            X[i, nk - 2] += -xik / h
            X[i, nk - 1] += 1.0 + xik / h
        else:
            # routine evaluation: locate interval [xk[j], xk[j+1]] s.t.
            # xk[j] <= xi <= xk[j+1].
            j = int(np.searchsorted(xk, xi, side="right") - 1)
            if j < 0:
                j = 0
            if j > nk - 2:
                j = nk - 2
            xj = xk[j]
            xj1 = xk[j + 1]
            h = xj1 - xj
            ajm = xj1 - xi
            ajp = xi - xj
            cjm = (ajm * (ajm * ajm / h - h)) / 6.0
            cjp = (ajp * (ajp * ajp / h - h)) / 6.0
            ajm /= h
            ajp /= h
            # X[i, k] = cjm * F[k, j] + cjp * F[k, j+1]
            for k in range(nk):
                X[i, k] = cjm * F[k, j] + cjp * F[k, j + 1]
            X[i, j] += ajm
            X[i, j + 1] += ajp
    return X


def cr_basis(
    t: np.ndarray,
    knots: np.ndarray,
    *,
    scale_penalty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mgcv ``bs='cr'`` natural cubic-regression spline basis.

    Parameters
    ----------
    t
        Query / data points, shape ``(n,)``. Need not be sorted.
    knots
        Strictly increasing knot positions, shape ``(k,)``, ``k >= 3``.
    scale_penalty
        If True (default), apply ``mgcv::smoothCon``'s ``scale.penalty=TRUE``
        rescaling so the returned ``S`` matches ``mgcv::gam``'s internal
        penalty. If False, return the raw ``D' B^{-1} D`` penalty from
        ``mgcv-1.9.3/src/mgcv.c:getFS``.

    Returns
    -------
    design_matrix
        ``(n, k)`` float64 design matrix (Wood eq 5.7).
    penalty_matrix
        ``(k, k)`` float64 symmetric penalty matrix.

    Notes
    -----
    The reference R implementation is::

        mgcv::smoothCon(mgcv::s(t, bs='cr', k=k), data=..., knots=...,
                        absorb.cons=FALSE)[[1]]$X / $S[[1]]

    which calls the C routine ``crspl`` (mgcv-1.9.3/src/mgcv.c:220) for the
    design and ``getFS`` (line 157) for the penalty, then symmetrizes ``S``
    and applies the ``scale.penalty`` rescaling described in
    ``R/smooth.r:smoothCon`` (lines 88-95 of the deparsed source).
    """
    t = np.asarray(t, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {t.shape}")
    if knots.ndim != 1:
        raise ValueError(f"knots must be 1-D, got shape {knots.shape}")
    F, S = _get_FS(knots)
    X = _crspl(t, knots, F)

    # mgcv applies S <- (S + S') / 2 after raw construction (smooth.r:
    # smooth.construct.cr.smooth.spec, see object$S[[1]] line).
    S = (S + S.T) / 2.0

    if scale_penalty:
        # smoothCon scale.penalty branch:
        #   maXX <- norm(X, type='I')^2     # infinity-norm = max row |sum|
        #   maS  <- norm(S, type='O') / maXX  # default norm = 1-norm
        #   S    <- S / maS
        maXX = float(np.linalg.norm(X, ord=np.inf)) ** 2
        maS = float(np.linalg.norm(S, ord=1)) / maXX
        S = S / maS

    return X, S
