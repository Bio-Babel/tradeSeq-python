"""NB-GAM fit kernel (replaces ``mgcv::gam(family="nb")``).

Implements the negative-binomial generalized additive model with a fixed
cubic-regression-spline basis (built by :mod:`tradeseq._basis`). The outer α
(NB dispersion) loop iterates the maximum-likelihood estimate of α to
convergence, mirroring ``mgcv::nb()``.

Five source-verified mgcv facts the implementation matches
(see ``tradeseq_porting_essential_suggestions.md`` §2):

1. ``family = "nb"`` → joint estimation of θ alongside smoothing (NOT
   ``MASS::negative.binomial(fixed_θ)``).
2. ``mgcv`` coerces extended-family NB fits from the nominal default
   ``GCV.Cp`` method to ``REML`` inside ``estimate.gam``. The object field is
   still named ``gcv.ubre`` by ``gam()``, but its value is the outer REML score.
3. ``Sigma`` stored downstream is ``Vp = (X'WX + S_λ)^{-1} φ`` — the Bayesian
   posterior covariance — NOT the frequentist ``V``.
4. ``weights`` is the per-observation observation weight (e.g. zero-inflation),
   passed straight to the GLM. NOT the lineage-assignment ``cell_weights``.
5. Offset enters the design matrix; not the ``offset=`` GLM argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import digamma, gammaln, polygamma

__all__ = ["GamFit", "fit_nb_gam_block"]


def _symmetrize_near_psd(A: np.ndarray, *, context: str) -> tuple[np.ndarray, float]:
    """Return the symmetric part of a matrix that should be positive semidefinite.

    Penalized GAM normal matrices have the form ``X'WX + λS`` and are
    mathematically symmetric positive semidefinite. In the tradeSeq basis they
    can be numerically rank deficient by one column, so treating tiny negative
    eigenvalues from roundoff as real curvature corrupts ``Vp``. Large negative
    eigenvalues are not expected and indicate an upstream construction bug.
    """
    sym = 0.5 * (A + A.T)
    eigvals = np.linalg.eigvalsh(sym)
    max_eval = float(max(eigvals[-1], 0.0))
    tol = np.finfo(float).eps * max(sym.shape) * max(max_eval, 1.0)
    if eigvals[0] < -100.0 * tol:
        raise np.linalg.LinAlgError(
            f"{context} is not positive semidefinite: "
            f"min eigenvalue {eigvals[0]:.6g}, tolerance {tol:.6g}."
        )
    return sym, tol


def _solve_penalized_system(H: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a penalized normal equation without adding artificial ridge terms.

    The tradeSeq design has an intercept plus lineage-specific CR bases whose
    null spaces can be nearly collinear. Cholesky can technically succeed on
    those matrices while amplifying the unidentified direction into huge
    cancelling coefficients. mgcv handles the same situation through
    rank-aware reparameterization; the equivalent Python-native solve is the
    symmetric Moore-Penrose minimum-norm solution.
    """
    H_sym, tol = _symmetrize_near_psd(H, context="penalized normal matrix")
    eigvals, eigvecs = np.linalg.eigh(H_sym)
    if eigvals[-1] <= tol:
        return np.zeros_like(rhs, dtype=np.float64)
    keep = eigvals > tol
    rhs_proj = eigvecs[:, keep].T @ rhs
    return eigvecs[:, keep] @ (rhs_proj / eigvals[keep])


def _penalized_hessian_pinv(H: np.ndarray) -> np.ndarray:
    """Rank-aware symmetric inverse for ``Vp = (X'WX + λS)^-``.

    mgcv's stored ``Vp`` is the Bayesian posterior covariance based on the
    penalized Hessian. The CR-spline basis used here can leave one numerical
    null direction; a plain dense inverse turns that near-zero direction into a
    huge signed eigenvalue. Using the symmetric Moore-Penrose inverse preserves
    the estimable coefficient subspace and keeps downstream Wald matrices
    positive semidefinite.
    """
    H_sym, tol = _symmetrize_near_psd(H, context="penalized Hessian")
    eigvals, eigvecs = np.linalg.eigh(H_sym)
    if eigvals[-1] <= tol:
        return np.zeros_like(H_sym)
    keep = eigvals > tol
    inv = (eigvecs[:, keep] / eigvals[keep]) @ eigvecs[:, keep].T
    return 0.5 * (inv + inv.T)


@dataclass
class GamFit:
    """Result of a single penalized NB-GAM fit."""

    beta: np.ndarray
    """Coefficient vector, shape ``(n_coefs,)``."""

    Vp: np.ndarray
    """Bayesian posterior covariance, shape ``(n_coefs, n_coefs)``."""

    alpha: float
    """Estimated NB dispersion (``Var(y) = mu + alpha * mu^2``)."""

    lam: float
    """Selected smoothness parameter."""

    deviance: float
    """Final NB deviance."""

    aic: float
    """mgcv-faithful NB AIC: ``-2 · log_lik + 2 · edf1`` (Wood §6.11.1)."""

    edf: float
    """Effective degrees of freedom = trace(hat matrix)."""

    converged: bool
    """``True`` iff outer α loop converged within tolerance."""

    smoothing_parameters: np.ndarray
    """Length-1 ndarray holding the selected ``lam`` (parallel to mgcv's `gam$sp`)."""

    F: np.ndarray
    """Hat-like matrix ``F = (X'WX + λS)^{-1} · X'WX`` at converged β/α/λ."""

    edf1: float
    """Second-order effective degrees of freedom: ``edf1 = 2*tr(F) - tr(F·F)``.

    R parity: mgcv's per-coefficient ``edf1`` (Wood §6.11.1). Used by ``mgcv::AIC``
    as the penalty: ``AIC = -2·log_lik + 2·edf1``.
    """

    log_lik: float
    """Negative-binomial log-likelihood at the converged fit (sum over cells)."""

    gcv: float
    """Smoothness-selection score at the converged fit.

    For NB fits this mirrors mgcv's ``m$gcv.ubre`` REML/outer-optimizer score.
    For Gaussian fits it remains Wood's literal GCV score
    ``n·D / (n - edf)^2``.
    """


def _safe_exp(eta: np.ndarray) -> np.ndarray:
    """Clip ``exp`` to avoid float overflow on extreme linear predictors."""
    return np.exp(np.clip(eta, -50.0, 50.0))


def _nb_deviance(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """Negative-binomial deviance (mgcv parameterisation).

    ``D = 2 * sum( y * log(y/mu) - (y + 1/alpha) * log((y + 1/alpha) / (mu + 1/alpha)) )``
    with the y=0 limit handled by treating ``0 * log(0/mu)`` as ``0``.
    """
    # NumPy evaluates both branches of np.where before masking, so guard the
    # log against y==0 (R's deviance uses the 0*log(0/mu) limit = 0).
    with np.errstate(divide="ignore", invalid="ignore"):
        if alpha <= 0:
            term1 = np.where(y > 0, y * np.log(y / np.maximum(mu, 1e-300)), 0.0)
            return float(2.0 * np.sum(term1 - (y - mu)))
        theta = 1.0 / alpha
        term1 = np.where(y > 0, y * np.log(y / np.maximum(mu, 1e-300)), 0.0)
        term2 = (y + theta) * np.log((y + theta) / (mu + theta))
        return float(2.0 * np.sum(term1 - term2))


def _penalized_irls(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    lam: float,
    alpha: float,
    weights: np.ndarray,
    beta_init: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float, float, float]:
    """One penalized IRLS fit at fixed ``alpha`` and ``lam``.

    The model is ``y ~ NB(mu, alpha)`` with ``log(mu) = X β`` (offset already
    inside ``X`` as a column of values × 1 coefficient — see
    :func:`fit_nb_gam_block` for the trick: we *do not* pass offset separately;
    instead the design includes the offset column and an additional fixed
    1-valued coefficient. **Actually** the simpler approach: split offset out
    of ``X`` and pass it here as part of the linear predictor. The caller
    decides.)

    Returns
    -------
    beta : ndarray
    deviance : float
    edf : float
        Effective degrees of freedom = trace((X'WX + λS)^{-1} X'WX).
    aic : float
        ``deviance + 2 * edf``.
    """
    n, p = X.shape
    beta = beta_init.copy()
    dev_old = np.inf

    for _ in range(max_iter):
        eta = X @ beta
        mu = _safe_exp(eta)
        mu = np.maximum(mu, 1e-12)

        # NB variance with log link: Var(y|mu) = mu * (1 + alpha * mu).
        # Working weights w = (dmu/deta)^2 / V(mu) = mu^2 / (mu * (1 + alpha*mu))
        #                  = mu / (1 + alpha*mu)
        w = weights * mu / (1.0 + alpha * mu)
        z = eta + (y - mu) / mu  # working response

        # Solve (X' W X + λ S) β = X' W z with a rank-aware symmetric solve.
        XtW = X.T * w
        H = XtW @ X + lam * S
        rhs = XtW @ z

        beta = _solve_penalized_system(H, rhs)

        dev = _nb_deviance(y, mu, alpha)
        if abs(dev_old - dev) / (abs(dev_old) + 0.1) < tol:
            break
        dev_old = dev

    # Final updated quantities at converged β
    eta = X @ beta
    mu = _safe_exp(eta)
    w = weights * mu / (1.0 + alpha * mu)
    H = (X.T * w) @ X + lam * S
    A_xtwx = (X.T * w) @ X
    H_inv = _penalized_hessian_pinv(H)
    edf = float(np.trace(H_inv @ A_xtwx))
    dev = _nb_deviance(y, mu, alpha)
    aic = dev + 2.0 * edf
    return beta, dev, edf, aic


def _gcv_score(deviance: float, edf: float, n: int) -> float:
    """Wood's GCV score for non-Gaussian families (eq 6.16):

    ``GCV = n * D / (n - edf)^2``
    """
    denom = max(n - edf, 1e-6)
    return n * deviance / (denom * denom)


def _positive_logdet(A: np.ndarray) -> tuple[float, int]:
    """Log pseudo-determinant and numerical rank of a near-PSD matrix."""
    A_sym, tol = _symmetrize_near_psd(A, context="log-determinant matrix")
    eigvals = np.linalg.eigvalsh(A_sym)
    if eigvals[-1] <= tol:
        return 0.0, 0
    keep = eigvals > tol
    return float(np.sum(np.log(eigvals[keep]))), int(np.sum(keep))


def _nb_reml_score(
    *,
    beta: np.ndarray,
    S: np.ndarray,
    lam: float,
    H: np.ndarray,
    log_lik: float,
    S_logdet: float,
    S_rank: int,
) -> float:
    """Laplace REML score matching mgcv's NB smoothness-selection shape.

    ``tradeSeq`` passes ``family="nb"``. In mgcv 1.9.3 that family is an
    ``extended.family`` and ``estimate.gam`` changes the requested default
    method to REML before calling ``gam.outer``. For one shared smoothing
    parameter, the λ-dependent part of mgcv's objective is the penalized
    negative log-likelihood plus the log-determinant adjustment between the
    penalized Hessian and the positive-rank penalty block:

    ``-logLik + 0.5 * λ β' S β + 0.5 log|X'WX + λS| - 0.5 log|λS|_+``.

    Additive constants independent of λ are intentionally omitted; they do not
    affect selection and leave per-gene scores affinely aligned with
    ``m$gcv.ubre``.
    """
    lam = float(lam)
    if lam <= 0.0 or not np.isfinite(lam):
        raise ValueError("lam must be a positive finite scalar.")
    H_logdet, _H_rank = _positive_logdet(H)
    penalty = 0.5 * lam * float(beta @ (S @ beta))
    lamS_logdet = S_logdet + S_rank * np.log(lam)
    return float(-log_lik + penalty + 0.5 * H_logdet - 0.5 * lamS_logdet)


def _nb_log_likelihood(
    y: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    weights: np.ndarray,
    *,
    family: str = "nb",
) -> float:
    """Negative-binomial (or Gaussian/Poisson) log-likelihood at converged fit.

    For ``family == "nb"`` with mean ``mu`` and dispersion ``alpha`` (so
    ``Var(y) = mu + alpha*mu^2``, ``size = theta = 1/alpha``):

    ``logL = sum_i w_i * [ lgamma(y_i+theta) - lgamma(theta) - lgamma(y_i+1)
            + theta*log(theta/(theta+mu_i)) + y_i*log(mu_i/(theta+mu_i)) ]``

    For ``family == "gaussian"`` this is undefined (returns 0); the caller
    should not use ``log_lik`` for AIC in that case.
    """
    if family == "gaussian":
        # Penalty-free GLS does not have a natural log-likelihood here.
        return 0.0
    theta = 1.0 / max(alpha, 1e-12)
    mu_safe = np.maximum(mu, 1e-300)
    log_terms = (
        gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
        + theta * np.log(theta / (theta + mu_safe))
        + y * np.log(mu_safe / (theta + mu_safe))
    )
    return float(np.sum(weights * log_terms))


def _nb_alpha_mle(
    y: np.ndarray, mu: np.ndarray, weights: np.ndarray, alpha_init: float
) -> float:
    """Maximum-likelihood estimate of ``alpha`` given ``mu``.

    Solves the NB score equation for theta = 1/alpha by Newton's method.
    Returns ``alpha`` in ``[1e-6, 1e6]``.
    """
    theta = 1.0 / max(alpha_init, 1e-6)
    for _ in range(40):
        # NB log-likelihood derivative w.r.t. theta:
        # dL/dtheta = sum_i w_i * ( digamma(y_i+theta) - digamma(theta)
        #             + log(theta/(theta+mu_i)) + 1 - (y_i+theta)/(theta+mu_i) )
        tym = theta + mu
        d_term = digamma(y + theta) - digamma(theta) + np.log(theta / tym) + 1.0 - (y + theta) / tym
        dL = float(np.sum(weights * d_term))
        # second derivative:
        # d2L/dtheta2 = sum_i w_i * ( trigamma(y_i+theta) - trigamma(theta)
        #             + 1/theta - 2/(theta+mu) + (y_i+theta)/(theta+mu)^2 )
        d2_term = polygamma(1, y + theta) - polygamma(1, theta) + 1.0 / theta - 2.0 / tym + (y + theta) / (tym * tym)
        d2L = float(np.sum(weights * d2_term))
        if d2L >= 0 or not np.isfinite(dL) or not np.isfinite(d2L):
            break
        step = dL / d2L
        new_theta = theta - step
        # backtracking to keep theta positive
        n_bt = 0
        while new_theta <= 0 and n_bt < 20:
            step *= 0.5
            new_theta = theta - step
            n_bt += 1
        if new_theta <= 0:
            break
        if abs(new_theta - theta) / (abs(theta) + 1e-8) < 1e-7:
            theta = new_theta
            break
        theta = new_theta
    theta = float(np.clip(theta, 1e-6, 1e6))
    return float(1.0 / theta)


def fit_nb_gam_block(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    offset: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    n_lam: int = 31,
    log_lam_range: tuple[float, float] = (-8.0, 18.0),
    max_outer: int = 25,
    tol: float = 1e-5,
    initial_alpha: float = 1.0,
    family: str = "nb",
) -> GamFit:
    """Fit a penalized NB-GAM with mgcv-style smoothness selection and α-loop.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Non-negative response.
    X : ndarray, shape (n, p)
        Design matrix (smoother basis × by-variable + fixed effects). The offset
        is NOT included here.
    S : ndarray, shape (p, p)
        Symmetric positive-semidefinite penalty matrix.
    offset : ndarray, shape (n,)
        Per-observation log-offset (added to ``X @ beta``).
    weights : ndarray, shape (n,) or None
        Per-observation weights. Defaults to ones.
    n_lam : int
        Number of log-spaced smoothness values to scan.
    log_lam_range : tuple (low, high)
        Range of ``log(lam)`` for the smoothness grid.
    max_outer : int
        Maximum number of α-updating outer iterations.
    tol : float
        Outer α-convergence tolerance on relative change.
    initial_alpha : float
        Starting NB dispersion.
    family : {"nb", "gaussian"}
        ``"nb"`` (default) iterates α and selects λ by the REML score used by
        mgcv's extended-family NB path. ``"gaussian"`` is a single penalized
        fit with ``alpha = 0`` and literal GCV selection.

    Returns
    -------
    GamFit
    """
    n, p = X.shape
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)

    # ABSORB OFFSET: extend design with an offset column whose coefficient is
    # constrained to 1 by augmenting both X and S accordingly. The simpler
    # and equivalent approach used here: subtract offset from working response
    # at every IRLS step. Net effect: linear predictor = X @ beta + offset.
    # We implement that by replacing X by [X, offset[:,None]] and pinning the
    # last coefficient to 1 via a hard prior. Cleanest: keep X separate and
    # subtract offset inside the IRLS loop.

    # We re-implement IRLS here so we can include the offset cleanly.
    log_lams = np.linspace(log_lam_range[0], log_lam_range[1], n_lam)
    S_logdet, S_rank = _positive_logdet(S)

    # Initial β: OLS on (log(y+1) - offset) for a stable start.
    log_y = np.log(y + 1.0) - offset
    try:
        beta0, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    except np.linalg.LinAlgError:
        beta0 = np.zeros(p)

    alpha = float(initial_alpha) if family == "nb" else 0.0
    beta = beta0.copy()
    best_lam = float(np.exp(np.mean(log_lams)))
    best_dev = np.inf
    best_edf = float(p)
    converged = False

    for outer in range(max_outer):
        alpha_prev = alpha
        best_score = np.inf
        best_beta_this = beta
        best_dev_this = np.inf
        best_lam_this = best_lam
        best_edf_this = best_edf
        for log_lam in log_lams:
            lam = float(np.exp(log_lam))
            b_lam, dev_lam, edf_lam, _aic_lam = _irls_with_offset(
                y, X, offset, S, lam, alpha, weights, beta
            )
            if family == "nb":
                mu_lam = _safe_exp(X @ b_lam + offset)
                w_lam = weights * mu_lam / (1.0 + alpha * mu_lam)
                H_lam = (X.T * w_lam) @ X + lam * S
                log_lik_lam = _nb_log_likelihood(
                    y, mu_lam, alpha, weights, family=family
                )
                score = _nb_reml_score(
                    beta=b_lam,
                    S=S,
                    lam=lam,
                    H=H_lam,
                    log_lik=log_lik_lam,
                    S_logdet=S_logdet,
                    S_rank=S_rank,
                )
            else:
                score = _gcv_score(dev_lam, edf_lam, n)
            if score < best_score:
                best_score = score
                best_beta_this = b_lam
                best_dev_this = dev_lam
                best_lam_this = lam
                best_edf_this = edf_lam
        beta = best_beta_this
        best_lam = best_lam_this
        best_dev = best_dev_this
        best_edf = best_edf_this

        if family == "gaussian":
            converged = True
            break

        # Update α via MLE on residuals
        mu = _safe_exp(X @ beta + offset)
        alpha_new = _nb_alpha_mle(y, mu, weights, alpha)
        if abs(alpha_new - alpha_prev) / (abs(alpha_prev) + 1e-8) < tol:
            alpha = alpha_new
            converged = True
            break
        alpha = alpha_new

    # Final Vp at converged (β, α, λ)
    mu = _safe_exp(X @ beta + offset)
    w = weights * mu / (1.0 + alpha * mu) if family == "nb" else weights
    XtWX = (X.T * w) @ X
    H = XtWX + best_lam * S
    Vp = _penalized_hessian_pinv(H)
    # NB GLM: dispersion φ = 1 (scale fixed). Vp already in correct units.

    # Hat-like matrix F = (X'WX + λS)^{-1} · X'WX  — Wood §6.11.1.
    F_mat = Vp @ XtWX
    # edf1 = 2·tr(F) - tr(F·F)
    trF = float(np.trace(F_mat))
    trFF = float(np.trace(F_mat @ F_mat))
    edf1 = 2.0 * trF - trFF

    # Log-likelihood at converged fit (mgcv NB: AIC = -2·log_lik + 2·edf1).
    log_lik = _nb_log_likelihood(y, mu, alpha, weights, family=family)
    aic = -2.0 * log_lik + 2.0 * edf1
    if family == "nb":
        gcv = _nb_reml_score(
            beta=beta,
            S=S,
            lam=best_lam,
            H=H,
            log_lik=log_lik,
            S_logdet=S_logdet,
            S_rank=S_rank,
        )
    else:
        gcv = _gcv_score(best_dev, best_edf, n)

    return GamFit(
        beta=beta,
        Vp=Vp,
        alpha=alpha,
        lam=best_lam,
        deviance=best_dev,
        aic=aic,
        edf=best_edf,
        converged=converged,
        smoothing_parameters=np.asarray([best_lam], dtype=np.float64),
        F=F_mat,
        edf1=edf1,
        log_lik=log_lik,
        gcv=gcv,
    )


def _irls_with_offset(
    y: np.ndarray,
    X: np.ndarray,
    offset: np.ndarray,
    S: np.ndarray,
    lam: float,
    alpha: float,
    weights: np.ndarray,
    beta_init: np.ndarray,
    *,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float, float, float]:
    """Penalized IRLS for log-link NB/Gaussian GLM with an offset.

    The linear predictor is ``eta = X @ beta + offset``. Returns
    ``(beta, deviance, edf, aic)``.
    """
    n, p = X.shape
    beta = beta_init.copy()
    dev_old = np.inf
    for _ in range(max_iter):
        eta = X @ beta + offset
        mu = _safe_exp(eta)
        mu = np.maximum(mu, 1e-12)
        if alpha > 0:
            w = weights * mu / (1.0 + alpha * mu)
        else:
            # Poisson limit (alpha == 0): w = weights * mu
            w = weights * mu
        z = eta + (y - mu) / mu - offset  # working response on (X β) scale
        XtW = X.T * w
        H = XtW @ X + lam * S
        rhs = XtW @ z
        beta = _solve_penalized_system(H, rhs)
        dev = _nb_deviance(y, mu, alpha)
        if abs(dev_old - dev) / (abs(dev_old) + 0.1) < tol:
            break
        dev_old = dev
    eta = X @ beta + offset
    mu = _safe_exp(eta)
    w = weights * mu / (1.0 + alpha * mu) if alpha > 0 else weights * mu
    XtWX = (X.T * w) @ X
    H = XtWX + lam * S
    H_inv = _penalized_hessian_pinv(H)
    edf = float(np.trace(H_inv @ XtWX))
    dev = _nb_deviance(y, mu, alpha)
    return beta, dev, edf, dev + 2.0 * edf
