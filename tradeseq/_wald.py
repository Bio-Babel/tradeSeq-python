"""Wald-test machinery (R source: tradeSeq/R/utils.R:350-424).

Implements two internal helpers used by the Tier-T2 Wald battery:

* :func:`wald_test_fc` - port of ``waldTestFC`` (utils.R:350-392). Computes a
  multivariate Wald statistic with a TREAT-style log-fold-change tube, with
  four interchangeable matrix-inverse strategies (``"QR"``, ``"Chol"``,
  ``"generalized"``, ``"eigen"``).
* :func:`get_eigen_stat_gam_fc` - port of ``getEigenStatGAMFC`` (utils.R:409-
  424). Returns the eigen-decomposition Wald statistic and its effective
  rank; returns ``(NaN, NaN)`` when the effective rank collapses to 1 (R sets
  this NaN sentinel explicitly).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

__all__ = ["wald_test_fc", "get_eigen_stat_gam_fc"]


def _treat_estimate(est_fc: np.ndarray, l2fc: float) -> np.ndarray:
    """TREAT-style shrinkage of a log fold-change vector toward zero.

    Implements ``sign(estFC) * pmax(0, abs(estFC) - log(2^l2fc))``.
    """
    log_fc_cutoff = np.log(2.0**l2fc)
    return np.sign(est_fc) * np.maximum(0.0, np.abs(est_fc) - log_fc_cutoff)


def _qr_pivot_keep(L: np.ndarray) -> np.ndarray:
    """Return the column indices of ``L`` corresponding to ``qr(L)$pivot[1:rank]``.

    Matches R's QR-based rank-deficient-column drop:
    ``L[, qr(L)$pivot[seq_len(qr(L)$rank)], drop = FALSE]``.

    R's default ``qr()`` path uses LINPACK ``dqrdc2`` rather than LAPACK
    ``geqp3``. ``dqrdc2`` preserves the current column order and moves columns
    whose residual norm falls below ``tol * original_norm`` to the right edge.
    That detail matters for tradeSeq's default ``inverse="Chol"`` branch:
    different full-rank subsets of a rank-deficient contrast matrix can produce
    different finite Cholesky Wald statistics.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L.reshape(-1, 1)
    if L.size == 0 or L.shape[1] == 0:
        return np.empty(0, dtype=int)

    tol = 1e-7
    piv = list(range(L.shape[1]))
    original_norm = np.linalg.norm(L, axis=0)
    q_basis: list[np.ndarray] = []
    keep: list[int] = []
    active_end = len(piv)
    pos = 0
    while pos < active_end:
        col = piv[pos]
        v = L[:, col].astype(float, copy=True)
        for q in q_basis:
            v -= q * float(np.dot(q, v))
        residual_norm = float(np.linalg.norm(v))
        threshold = tol * float(original_norm[col])
        if residual_norm <= threshold:
            piv.append(piv.pop(pos))
            active_end -= 1
            continue
        q_basis.append(v / residual_norm)
        keep.append(col)
        pos += 1
    return np.asarray(keep, dtype=int)


def get_eigen_stat_gam_fc(
    beta: np.ndarray,
    Sigma: np.ndarray,
    L: np.ndarray,
    l2fc: float,
    eigen_thresh: float = 1e-2,
) -> tuple[float, float]:
    """Eigen-decomposition Wald statistic with TREAT-style l2fc tube.

    Port of ``getEigenStatGAMFC`` (utils.R:409-424). Eigendecomposes
    ``Lᵀ Σ L`` (symmetric), retains eigenvalues exceeding
    ``eigen_thresh * lambda_max`` to define an effective rank ``r``, then
    builds a pseudoinverse half-covariance ``V D^{-1/2}`` and computes
    ``||V' est||²``.

    Parameters
    ----------
    beta : np.ndarray
        Coefficient vector of length ``p`` (or column matrix).
    Sigma : np.ndarray
        ``(p, p)`` coefficient covariance.
    L : np.ndarray
        ``(p, k)`` contrast matrix.
    l2fc : float
        Log-2 fold-change threshold for TREAT-style shrinkage. ``0`` reduces
        to an ordinary Wald test.
    eigen_thresh : float, default 1e-2
        Relative eigenvalue cutoff (fraction of the largest eigenvalue).

    Returns
    -------
    stat : float
        Wald statistic, or ``np.nan`` when effective rank is 1.
    rank : float
        Effective rank used, or ``np.nan`` when effective rank is 1.
    """
    beta = np.asarray(beta, dtype=float).reshape(-1, 1)
    Sigma = np.asarray(Sigma, dtype=float)
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L.reshape(-1, 1)

    est_fc = L.T @ beta  # (k, 1)
    est = _treat_estimate(est_fc, l2fc)
    sigma = L.T @ Sigma @ L  # (k, k)
    # symmetrize for numerical hygiene; R's eigen(..., symmetric=TRUE) does this implicitly
    sigma_sym = 0.5 * (sigma + sigma.T)

    # eigh returns ASCENDING eigenvalues; reverse to DESCENDING to match R's
    # eigen(..., symmetric = TRUE).
    eig_vals_asc, eig_vecs_asc = np.linalg.eigh(sigma_sym)
    eig_vals = eig_vals_asc[::-1]
    eig_vecs = eig_vecs_asc[:, ::-1]

    if eig_vals.size == 0 or eig_vals[0] == 0:
        return (np.nan, np.nan)

    ratio = eig_vals / eig_vals[0]
    r = int(np.sum(ratio > eigen_thresh))
    if r == 1:
        return (np.nan, np.nan)

    V = eig_vecs[:, :r]
    D_inv_half = np.diag(1.0 / np.sqrt(eig_vals[:r]))
    half_cov_inv = V @ D_inv_half  # (k, r)
    half_stat = est.T @ half_cov_inv  # (1, r)
    stat_arr = half_stat @ half_stat.T  # (1, 1)
    stat = float(stat_arr.reshape(()))
    return (stat, float(r))


def wald_test_fc(
    beta: np.ndarray,
    Sigma: np.ndarray,
    L: np.ndarray,
    l2fc: float = 0.0,
    inverse: str = "QR",
    eigen_thresh: float = 1e-2,
) -> tuple[float, float, float]:
    """Multivariate Wald test on ``Lᵀβ = 0`` with optional TREAT-style l2fc tube.

    Port of ``waldTestFC`` (utils.R:350-392). Four inverse strategies are
    available:

    * ``"QR"``: ``np.linalg.solve`` on ``LᵀΣL``.
    * ``"Chol"``: Cholesky-based inverse of ``LᵀΣL``.
    * ``"generalized"``: Moore-Penrose pseudoinverse.
    * ``"eigen"``: delegates to :func:`get_eigen_stat_gam_fc`.

    Rank-deficient columns of ``L`` are dropped via ``qr(L)$pivot[1:rank]``
    before the inverse is computed (except in the eigen branch, which uses
    the full ``L`` and relies on the eigen-value tail cutoff).

    Parameters
    ----------
    beta : np.ndarray
        Coefficient vector of length ``p``.
    Sigma : np.ndarray
        ``(p, p)`` coefficient covariance.
    L : np.ndarray
        ``(p, k)`` contrast matrix.
    l2fc : float, default 0.0
        log-2 fold-change threshold for TREAT-style shrinkage.
    inverse : {"QR", "Chol", "generalized", "eigen"}, default "QR"
        Matrix-inverse strategy.
    eigen_thresh : float, default 1e-2
        Relative eigenvalue cutoff used only by ``inverse="eigen"``.

    Returns
    -------
    stat : float
        Wald statistic (or ``np.nan`` on failure).
    df : float
        Degrees of freedom (rank of the contrast) or ``np.nan`` on failure.
    pval : float
        Chi-squared upper-tail p-value (or ``np.nan`` on failure).
    """
    beta = np.asarray(beta, dtype=float).reshape(-1, 1)
    Sigma = np.asarray(Sigma, dtype=float)
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L.reshape(-1, 1)

    allowed = {"QR", "Chol", "generalized", "eigen"}
    if inverse not in allowed:
        raise ValueError(
            f"Unknown inverse strategy: {inverse!r}; expected one of "
            f"{sorted(allowed)}."
        )

    if inverse == "eigen":
        try:
            stat, df = get_eigen_stat_gam_fc(
                beta, Sigma, L, l2fc, eigen_thresh=eigen_thresh
            )
        except Exception:
            return (np.nan, np.nan, np.nan)
        if np.isnan(stat):
            return (np.nan, np.nan, np.nan)
        pval = float(chi2.sf(stat, df=df))
        return (float(stat), float(df), pval)

    # Drop rank-deficient L columns via column-pivoted QR (matches R qr(L)).
    keep = _qr_pivot_keep(L)
    LQR = L[:, keep]
    if LQR.shape[1] == 0:
        return (np.nan, np.nan, np.nan)

    M = LQR.T @ Sigma @ LQR  # (r, r)

    try:
        if inverse == "Chol":
            # R: chol2inv(chol(M)) — inverse via upper-triangular Cholesky factor.
            chol = np.linalg.cholesky(M)  # lower
            inv_chol = np.linalg.solve(chol, np.eye(chol.shape[0]))
            sigma_inv = inv_chol.T @ inv_chol
        elif inverse == "QR":
            # R: qr.solve(M) — solve(M, I) via QR.
            sigma_inv = np.linalg.solve(M, np.eye(M.shape[0]))
        elif inverse == "generalized":
            # R: MASS::ginv(M).
            sigma_inv = np.linalg.pinv(M)
    except Exception:
        return (np.nan, np.nan, np.nan)

    est_fc = LQR.T @ beta  # (r, 1)
    est = _treat_estimate(est_fc, l2fc).reshape(-1, 1)
    wald_arr = est.T @ sigma_inv @ est  # (1, 1)
    wald = float(wald_arr.reshape(()))
    if wald < 0:
        wald = 0.0
    df = float(LQR.shape[1])
    pval = float(chi2.sf(wald, df=df))
    return (wald, df, pval)
