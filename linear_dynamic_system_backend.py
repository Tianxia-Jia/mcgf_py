from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Literal

import jax
import jax.numpy as jnp

from correlation import DTYPE, _as_dtype, cor_stat
from estimate_transform import Transform, make_transforms
from estimate_backend import (
    _assemble_full_uncon,
    _split_free_fixed,
    _unconstrained_to_constrained_dict,
    _build_par_base,
    _build_par_lagr,
    _run_optimizer,
    _wls_loss,
    _tree_to_py,
)

Array = jax.Array
RIDGE = 1e-10


@jax.jit
def _symmetrize(mat: Array, ridge: float = RIDGE) -> Array:
    return 0.5 * (mat + mat.T) + ridge * jnp.eye(mat.shape[0], dtype=mat.dtype)


# ---------------------------------------------------------------------
# 1) Build block covariance/correlation matrices directly from lag-corrs
# ---------------------------------------------------------------------
def _block_cov(corrs: Array, offs_r: Array, offs_c: Array) -> Array:
    """
    corrs: (n, n, L) where corrs[:, :, ell] is lag-ell covariance/correlation (ell >= 0).
    offs_r, offs_c: integer time offsets (e.g., [0, -1, -2, ...])
    """
    d = offs_r[:, None] - offs_c[None, :]
    lag = jnp.abs(d).astype(jnp.int32)

    # gather blocks: (n, n, i, j)
    blocks = corrs[:, :, lag]
    blocks_T = jnp.swapaxes(blocks, 0, 1)
    blocks = jnp.where((d < 0)[None, None, :, :], blocks_T, blocks)

    lag_r, lag_c = offs_r.shape[0], offs_c.shape[0]
    n = corrs.shape[0]
    blocks = jnp.transpose(blocks, (2, 0, 3, 1)).reshape(lag_r * n, lag_c * n)
    return blocks


# ---------------------------------------------------------------------
# 2) JAX version of mcgf:::cov_par (no explicit inverse)
# ---------------------------------------------------------------------
def _cov_par(A: Array, B: Array, D: Array, ridge: float = RIDGE) -> Tuple[Array, Array]:
    """
    One-step LDS form (update 1 time step at a time).

    Let lag = p. Define state as:
        z_t = [W_t, W_{t-1}, ..., W_{t-p}]   (p+1 blocks)

    The conditional for the Markov chain:
        W_t | (W_{t-1},...,W_{t-p})  ~  N(B D^{-1} * [W_{t-1};...;W_{t-p}],  A - B D^{-1} B^T)

    Shapes:
        A: (n, n)
        B: (n, n*lag)
        D: (n*lag, n*lag)

    Returns:
        W = B D^{-1}  (n, n*lag)
        Sigma_e = A - B D^{-1} B^T  (n, n)
    """
    # fix floating-point errors that break symmetry
    D = _symmetrize(D, ridge=ridge)
    WT = jax.scipy.linalg.solve(D, B.T, assume_a="sym")  # (n*lag, n)
    W = WT.T  # (n, n*lag)

    Sigma_e = A - W @ B.T
    Sigma_e = _symmetrize(Sigma_e)
    return W, Sigma_e


# ============================================================
# 3) Build companion F and process-noise Q from W, Sigma_e
# ============================================================
@jax.jit
def _F_onestep(W: Array, n: int, lag: int) -> Array:
    """
    State z_t = [W_t, W_{t-1}, ..., W_{t-lag}]  (lag+1 blocks)

    Dynamics:
        W_{t+1} = W * [W_t; ...; W_{t-lag+1}] + e_{t+1}
        shift: [W_t; ...; W_{t-lag+1}] becomes the lower blocks

    F has shape (n*(lag+1), n*(lag+1)):
        top row blocks: [W_{n×(n*lag)}, 0_{n×n}]
        bottom blocks:  [I_{n*lag}, 0_{np×n}]
    """
    dim = n * (lag + 1)
    F = jnp.zeros((dim, dim), dtype=DTYPE)

    # top block row: [W, 0]
    F = F.at[:n, : n * lag].set(W)

    # shift: z_{t+1}[1:] = z_t[:lag]
    I = jnp.eye(n * lag, dtype=DTYPE)
    F = F.at[n:, : n * lag].set(I)
    return F


@jax.jit
def _Q_onestep(Sigma_e: Array, n: int, lag: int) -> Array:
    """
    Process noise enters only the first block (the W_t / W_{t+1} equation).
    """
    dim = n * (lag + 1)
    Q = jnp.zeros((dim, dim), dtype=DTYPE)
    Q = Q.at[:n, :n].set(Sigma_e)
    return Q


# ---------------------------------------------------------------------
# 4) Build (A,B,D) from a lag-correlation array corrs[lag] (lag>=0)
# ---------------------------------------------------------------------
def _ABD_from_corrs(corrs: Array, lag: int) -> Tuple[Array, Array, Array]:
    """
    One-step partition consistent with state z_t = [W_t, W_{t-1}, ..., W_{t-lag}]:

        A = Cov(W_t, W_t)                          -> (n, n)
        B = Cov(W_t, [W_{t-1},...,W_{t-lag}])      -> (n, n*lag)
        D = Cov([W_{t-1},...,W_{t-lag}], itself)   -> (n*lag, n*lag)

    Requires corrs to contain lags 0..lag (so corrs.shape[2] >= lag+1).
    """
    # W_t
    offs_y = jnp.array([0], dtype=jnp.int32)

    # W_{t-1}, ..., W_{t-lag}
    offs_x = -jnp.arange(1, lag + 1, dtype=jnp.int32)

    A = _block_cov(corrs, offs_y, offs_y)
    B = _block_cov(corrs, offs_y, offs_x)
    D = _block_cov(corrs, offs_x, offs_x)
    return A, B, D


# ---------------------------------------------------------------------
# 5) Fixed-point stationary covariance: P = F P F^T + Q
# ---------------------------------------------------------------------
@jax.jit
def _stationary_cov_fixed_point(
    F: Array,
    Q: Array,
    tol: float = 1e-2,
    maxiter: int = 100,
    ridge: float = RIDGE,
) -> Array:

    tol = jnp.asarray(tol, dtype=DTYPE)
    ridge = jnp.asarray(ridge, dtype=DTYPE)

    def body(state):
        i, cov, _ = state
        Pn = F @ cov @ F.T + Q
        Pn = _symmetrize(Pn, ridg=ridge)
        errn = jnp.max(jnp.abs(Pn - cov))
        return (i + 1, Pn, errn)

    def cond(state):
        i, _, err = state
        return jnp.logical_and(i < maxiter, err > tol)

    init = (jnp.asarray(0, dtype=jnp.int32), Q, jnp.asarray(jnp.inf, dtype=DTYPE))
    _, cov, _ = jax.lax.while_loop(cond, body, init)
    return cov


# ---------------------------------------------------------------------
# 6) Extract lag-cov / lag-corr of W_t from state covariance P
# ---------------------------------------------------------------------
def _covs_from_stat_cov(cov: Array, n: int, lag: int) -> Array:
    """
    cov: stationary covariance of z_t = [W_t, W_{t-1}, ..., W_{t-lag}] (dim = n*(lag+1))

    Returns covs: (n, n, lag+1) where covs[:,:,ell] = Cov(W_t, W_{t-ell})
    """
    covs = cov[:n, :].reshape(n, lag + 1, n).transpose(0, 2, 1)
    covs = covs.at[:, :, 0].set(_symmetrize(covs[:, :, 0]))
    return covs


@jax.jit
def _covs_to_corrs(covs: Array, eps: float = 1e-10) -> Array:
    """
    covs: (n, n, L)
    Converts each lag-cov matrix to correlation using sd from covs[:,:,0].
    """
    cov0 = covs[:, :, 0]
    sd = jnp.sqrt(jnp.clip(jnp.diag(cov0), a_min=eps))
    denom = sd[:, None] * sd[None, :]

    corrs = covs / denom[None, :, :]
    corrs = jnp.clip(corrs, -1.0, 1.0)

    # enforce exact 1 on diag at lag 0
    n = cov0.shape[0]
    corr0 = jnp.where(jnp.eye(n, dtype=bool), 1.0, corrs[:, :, 0])
    corrs = corrs.at[:, :, 0].set(corr0)
    return corrs


# ---------------------------------------------------------------------
# 7) One call: model corr -> build F,Q -> stationary corr (fixed point)
# ---------------------------------------------------------------------
def stationary_corr_from_model_corrs(
    corrs_model: Array,
    n: int,
    lag: int,
    tol: float = 1e-6,
    maxiter: int = 100,
    ridge: float = RIDGE,
) -> Array:
    """
    One-step LDS (no horizon).

    Inputs:
        corrs_model: (n, n, lag+1)  lag slices 0..lag

    Steps:
        (A,B,D) from corrs_model
        W, Sigma_e from (A,B,D)
        build one-step companion (F,Q)
        solve stationary P = F P F^T + Q
        extract covs for lags 0..lag and normalize to corrs
    """
    A, B, D = _ABD_from_corrs(corrs_model, lag=lag)
    W, Sigma_e = _cov_par(A, B, D)

    F = _F_onestep(W, n=n, lag=lag)
    Q = _Q_onestep(Sigma_e, n=n, lag=lag)

    P = _stationary_cov_fixed_point(F, Q, tol=tol, maxiter=maxiter, ridge=ridge)
    covs = _covs_from_stat_cov(P, n=n, lag=lag)
    return _covs_to_corrs(covs)


# ---------------------------------------------------------------------
# 8) Fit ALL parameters at once (base + lagr), with stationary-corr correction
# ---------------------------------------------------------------------
def jax_fit_all(
    *,
    base: Literal["sep", "fs"],
    lagrangian: Literal["none", "lagr_tri", "lagr_exp", "lagr_askey"],
    # model inputs (for correlation model evaluation):
    lag: int,
    h: Any,
    u: Any,
    cor_emp: Any,
    h1: Any = None,
    h2: Any = None,
    # init/fixed (constrained space):
    par_init: Dict[str, Any],
    par_fixed: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Transform]] = None,
    # fixed-point controls:
    fp_tol: float = 1e-2,
    fp_maxiter: int = 100,
    ridge: float = RIDGE,
    # optimizer
    method: str = "auto",
    maxiter: Optional[int] = 10000,
    control: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    - Evaluates the (possibly non-stationary / not MCGF-consistent) model correlations.
    - Builds a one-step LDS (update 1 time step at a time), then computes the implied
      stationary lag-corrs via fixed-point Lyapunov iteration.
    - Minimizes WLS between stationary lag-corrs and cor_emp.

    Requirement:
        cor_emp.shape == (n, n, lag+1)
    """
    control = control or {}

    h = _as_dtype(h)
    u = _as_dtype(u)
    cor_emp = _as_dtype(cor_emp)

    n, n2, n_block = cor_emp.shape
    if n != n2:
        raise ValueError(f"cor_emp must be (n,n,lag+1). Got {cor_emp.shape}.")
    if n_block != lag + 1:
        raise ValueError(
            f"Expected cor_emp last axis = lag+1 = {lag+1}, got {n_block}."
        )

    tf = transforms if transforms is not None else make_transforms()

    # Parameter list depends on whether lagrangian is used
    base_param_names = ("nugget", "c", "gamma", "a", "alpha", "beta")
    lagr_param_names = ("lam", "v1", "v2", "k")

    if lagrangian == "none":
        param_names = base_param_names
    else:
        if h1 is None or h2 is None:
            raise ValueError("Need h1/h2 when lagrangian != 'none'")
        h1 = _as_dtype(h1)
        h2 = _as_dtype(h2)
        param_names = base_param_names + lagr_param_names

    free_names, x0_free, fixed_uncon = _split_free_fixed(
        param_names, par_init, par_fixed, tf
    )

    def _model_corrs_from_params(par_all_con: Dict[str, Array]) -> Array:
        par_base = _build_par_base(base, par_all_con)

        if lagrangian == "none":
            return cor_stat(
                base=base,
                lagrangian="none",
                par_base=par_base,
                par_lagr=None,
                lam=0.0,
                h=h,
                u=u,
                base_fixed=False,
            )
        else:
            par_lagr, lam = _build_par_lagr(lagrangian, par_all_con)
            return cor_stat(
                base=base,
                lagrangian=lagrangian,
                par_base=par_base,
                par_lagr=par_lagr,
                lam=lam,
                h=h,
                h1=h1,
                h2=h2,
                u=u,
                base_fixed=False,
            )

    def f(x_free: Array) -> Array:
        x_full = _assemble_full_uncon(x_free, free_names, fixed_uncon, param_names)
        par_all_con = _unconstrained_to_constrained_dict(x_full, param_names, tf)

        corrs_model = _model_corrs_from_params(par_all_con)
        corrs_stat = stationary_corr_from_model_corrs(
            corrs_model, n=n, lag=lag, tol=fp_tol, maxiter=fp_maxiter, ridge=ridge
        )
        return _wls_loss(corrs_stat, cor_emp)

    value_and_grad = jax.value_and_grad(f)
    opt_res = _run_optimizer(
        value_and_grad, x0_free, method=method, control=control, maxiter=maxiter
    )

    x_star_free = opt_res["x_star"]
    x_star_full = _assemble_full_uncon(
        x_star_free, free_names, fixed_uncon, param_names
    )
    par_all_hat = _unconstrained_to_constrained_dict(x_star_full, param_names, tf)

    # unpack for reporting
    par_base_hat = _build_par_base(base, par_all_hat)

    if lagrangian == "none":
        par_lagr_hat, lam_hat = None, 0.0
    else:
        par_lagr_hat, lam_hat = _build_par_lagr(lagrangian, par_all_hat)

    obj_val = float(f(x_star_free))

    return _tree_to_py(
        {
            "par_base": par_base_hat,
            "par_lagr": par_lagr_hat,
            "lam": lam_hat,
            "objective": obj_val,
            "converged": opt_res["converged"],
            "n_iter": opt_res["n_iter"],
            "message": opt_res.get("message", ""),
            "backend": "jax",
            "notes": {
                "stationary_corr": True,
                "fp_tol": float(fp_tol),
                "fp_maxiter": int(fp_maxiter),
                "lag": int(lag),
                "one_step_lds": True,
            },
        }
    )
