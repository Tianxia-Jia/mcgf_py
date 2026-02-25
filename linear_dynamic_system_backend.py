from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Literal

import jax
from jax import lax
from jaxopt import FixedPointIteration, AndersonAcceleration
import jax.numpy as jnp

from correlation import DTYPE, _as_dtype, cor_base_stat, cor_lagr_lds
from estimate_transform import Transform, make_transforms
from estimate_backend import (
    _assemble_full_uncon,
    _split_free_fixed,
    _unconstrained_to_constrained_dict,
    _build_par_base,
    _build_par_lds,
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


def _cov_joint(covs: Array) -> Array:

    offs = -jnp.arange(0, covs.shape[2], dtype=jnp.int32)
    return _block_cov(covs, offs, offs)


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
@jax.jit(static_argnames=("n", "lag"))
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


@jax.jit(static_argnames=("n", "lag"))
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
def _ABD_from_corrs_joint(corrs: Array, lag: int) -> Tuple[Array, Array, Array]:
    """
    One-step partition consistent with state z_t = [W_t, W_{t-1}, ..., W_{t-lag}]:

        A = Cov(W_t, W_t)                          -> (n, n)
        B = Cov(W_t, [W_{t-1},...,W_{t-lag}])      -> (n, n*lag)
        D = Cov([W_{t-1},...,W_{t-lag}], itself)   -> (n*lag, n*lag)
    """
    dim = corrs.shape[0]
    n_loc = dim // (lag + 1)
    if n_loc * (lag + 1) != dim:
        raise ValueError(f"corrs has incompatible size {dim} for lag={lag}.")

    A = corrs[:n_loc, :n_loc]
    B = corrs[:n_loc, n_loc:]
    D = corrs[n_loc:, n_loc:]
    return A, B, D


# ---------------------------------------------------------------------
# 5) Fixed-point stationary covariance: P = F P F^T + Q
# ---------------------------------------------------------------------
FpMethod = Literal[
    "scan_fp",  # Scan-based fixed-point iteration
    "jaxopt_fpi",  # jaxopt.FixedPointIteration
    "jaxopt_anderson",  # jaxopt.AndersonAcceleration
]


@jax.jit(static_argnames=("maxiter"))
def _stationary_cov_scan_fp(
    F: Array, Q: Array, *, tol: float = 1e-1, maxiter: int = 100, ridge: float = RIDGE
) -> Array:
    """
    Stationary covariance solve for Σ = F Σ Fᵀ + Q via scan-based fixed-point iteration.
    - maxiter is static for scan (required by JAX).
    - ridge is added once into Q.
    - symmetrize once at the end (no per-iter symmetrize).
    - uses lax.cond to avoid matmuls after convergence.
    """
    tol = jnp.asarray(tol, dtype=DTYPE)
    ridge = jnp.asarray(ridge, dtype=DTYPE)

    # Initial covariance guess (Σ₀)
    dim = F.shape[0]
    P0 = Q + ridge * jnp.eye(dim, dtype=F.dtype)

    # Initial error: set to +∞ so first update always executes
    err0 = jnp.asarray(jnp.inf, dtype=DTYPE)

    # Convergence flag: False initially
    done0 = jnp.asarray(False)

    # ------------------------------------------------------------
    # One iteration step of the fixed-point method
    # ------------------------------------------------------------
    def step(state, _):
        cov, err, done = state

        def update_fn(_):
            Pn = F @ cov @ F.T + Q
            errn = jnp.max(jnp.abs(Pn - cov))
            done_n = errn <= tol
            return Pn, errn, done_n

        Pn, errn, new_done = lax.cond(
            done,
            lambda _: (cov, err, True),
            update_fn,
            operand=None,
        )

        done_n = jnp.logical_or(done, new_done)

        return (Pn, errn, done_n), None

    (P, _, _), _ = lax.scan(step, (P0, err0, done0), xs=None, length=maxiter)
    P = 0.5 * (P + P.T)
    return P


def _stationary_cov_jaxopt_fpi(
    F: Array,
    Q: Array,
    *,
    ridge: float = RIDGE,
    tol: float = 1e-1,
    maxiter: int = 200,
    implicit_diff: bool = True,
) -> Array:
    """
    Stationary covariance via jaxopt.FixedPointIteration on T(P)=F P Fᵀ + Qr.
    """
    ridge = jnp.asarray(ridge, dtype=DTYPE)

    dim = F.shape[0]
    P0 = Q + ridge * jnp.eye(dim, dtype=F.dtype)
    FT = F.T

    def T(P, F, FT, Q):
        return F @ P @ FT + Q

    solver = FixedPointIteration(
        fixed_point_fun=T,
        maxiter=int(maxiter),
        tol=float(tol),
        implicit_diff=bool(implicit_diff),
    )

    @jax.jit
    def _run(P0, F, FT, Q):
        P = solver.run(P0, F=F, FT=FT, Q=Q).params
        return 0.5 * (P + P.T)

    return _run(P0, F, FT, Q)


def _stationary_cov_jaxopt_anderson(
    F: jnp.ndarray,
    Q: jnp.ndarray,
    *,
    ridge=RIDGE,
    fp_ridge=1e-05,
    tol: float = 1e-1,
    maxiter: int = 200,
    history_size: int = 5,
    beta: float = 1.0,
    implicit_diff: bool = True,
) -> jnp.ndarray:
    """
    Stationary covariance via jaxopt.AndersonAcceleration on T(P)=F P Fᵀ + Qr.
    """
    dim = F.shape[0]
    P0 = Q + ridge * jnp.eye(dim, dtype=F.dtype)
    FT = F.T

    def T(P, F, FT, Q):
        return F @ P @ FT + Q

    solver = AndersonAcceleration(
        fixed_point_fun=T,
        maxiter=int(maxiter),
        tol=float(tol),
        ridge=fp_ridge,
        history_size=int(history_size),
        beta=float(beta),
        implicit_diff=bool(implicit_diff),
    )

    @jax.jit
    def _run(P0, F, FT, Q):
        P = solver.run(P0, F=F, FT=FT, Q=Q).params
        return 0.5 * (P + P.T)

    return _run(P0, F, FT, Q)


def stationary_cov_solve(
    F: jnp.ndarray,
    Q: jnp.ndarray,
    *,
    ridge: float = 0.0,
    fp_method: FpMethod = "jaxopt_anderson",
    fp_control: Optional[Dict[str, Any]] = None,
) -> jnp.ndarray:
    """
    Unified stationary covariance solver for Σ = F Σ Fᵀ + Q.

    fp_control keys (common):
      - tol: float
      - maxiter: int

    fp_control keys (anderson):
      - m: int
      - beta: float

    fp_control keys (jaxopt):
      - implicit_diff: bool
    """
    fp_control = fp_control or {}

    tol = float(fp_control.get("tol", 1e-1))
    maxiter = int(fp_control.get("maxiter", 200))

    if fp_method == "scan_fp":
        # scan_fp is jitted; maxiter must be static
        return _stationary_cov_scan_fp(F, Q, tol=tol, maxiter=maxiter, ridge=ridge)

    if fp_method == "jaxopt_fpi":
        implicit_diff = bool(fp_control.get("implicit_diff", True))
        return _stationary_cov_jaxopt_fpi(
            F, Q, tol=tol, maxiter=maxiter, ridge=ridge, implicit_diff=implicit_diff
        )

    if fp_method == "jaxopt_anderson":
        fp_ridge = float(fp_control.get("ridge", 1e-05))
        implicit_diff = bool(fp_control.get("implicit_diff", True))
        history_size = int(fp_control.get("history_size", 5))
        beta = float(fp_control.get("beta", 1.0))
        return _stationary_cov_jaxopt_anderson(
            F,
            Q,
            tol=tol,
            maxiter=maxiter,
            ridge=ridge,
            fp_ridge=fp_ridge,
            history_size=history_size,
            beta=beta,
            implicit_diff=implicit_diff,
        )

    raise ValueError(f"Unknown fp_method: {fp_method!r}")


# ---------------------------------------------------------------------
# 6) Extract lag-cov / lag-corr of W_t from state covariance P
# ---------------------------------------------------------------------
@jax.jit
def _covs_to_corrs(covs: Array, eps: float = 1e-10) -> Array:
    """
    covs: (n*L, n*L)
    Converts each lag-cov matrix to correlation using sd from covs.
    """
    sd = jnp.sqrt(jnp.clip(jnp.diag(covs), a_min=eps))
    denom = sd[:, None] * sd[None, :]

    corrs = covs / denom
    corrs = jnp.clip(corrs, -1.0, 1.0)

    # enforce exact 1 on diag
    idx = jnp.diag_indices(corrs.shape[0])
    return corrs.at[idx].set(1.0)


# ---------------------------------------------------------------------
# 7) One call: model corr -> build F,Q -> stationary corr (fixed point)
# ---------------------------------------------------------------------
def stationary_corr_from_model_corrs(
    corrs_model: Array,
    *,
    n: int,
    lag: int,
    fp_method: FpMethod = "jaxopt_anderson",
    fp_control: Optional[Dict[str, Any]] = None,
    ridge: float = RIDGE,
) -> Array:
    """
    One-step LDS (no horizon).

    Inputs:
        corrs_model: [n * (lag+1), n * (lag+1)]

    Steps:
        (A,B,D) from corrs_model
        W, Sigma_e from (A,B,D)
        build one-step companion (F,Q)
        solve stationary P = F P F^T + Q
    """
    A, B, D = _ABD_from_corrs_joint(corrs_model, lag=lag)
    W, Sigma_e = _cov_par(A, B, D)

    F = _F_onestep(W, n=n, lag=lag)
    Q = _Q_onestep(Sigma_e, n=n, lag=lag)

    cov = stationary_cov_solve(
        F, Q, fp_method=fp_method, fp_control=fp_control, ridge=ridge
    )
    return _covs_to_corrs(cov)


# ---------------------------------------------------------------------
# 8) Fit ALL parameters at once (base + lagr), with stationary-corr correction
# ---------------------------------------------------------------------
def jax_fit_all(
    *,
    base: Literal["sep", "fs"],
    lagrangian="exp",
    # model inputs (for correlation model evaluation):
    lag: int,
    h: Any,
    h_lds: Any,
    u: Any,
    cor_emp: Any,
    # init/fixed (constrained space):
    par_init: Dict[str, Any],
    par_fixed: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Transform]] = None,
    # fixed-point controls:
    fp_method: FpMethod = "jaxopt_anderson",
    fp_control: Optional[Dict[str, Any]] = None,
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
    """
    control = control or {}
    fp_control = fp_control or {}

    lag = int(lag)
    lag_p1 = lag + 1

    # --- shape checks ---
    n_loc = h.shape[0]
    joint_dim = (lag_p1 * n_loc, lag_p1 * n_loc)
    if tuple(h_lds.shape) != joint_dim:
        raise ValueError(
            f"Unmatching shape for h_lds, must be {joint_dim}, got {h_lds.shape}"
        )
    if tuple(cor_emp.shape) != joint_dim:
        raise ValueError(
            f"Unmatching shape for h_lds, must be {joint_dim}, got {h_lds.shape}"
        )

    # --- dtypes ---
    h = _as_dtype(h)
    h_lds = _as_dtype(h_lds)
    u = _as_dtype(u)
    cor_emp = _as_dtype(cor_emp)

    tf = transforms if transforms is not None else make_transforms()

    # Parameter list depends on whether lagrangian is used
    base_param_names = ("nugget", "c", "gamma", "a", "alpha", "beta")
    lds_param_names = ("lam", "lds_nugget", "lds_c", "lds_gamma")
    param_names = base_param_names + lds_param_names

    free_names, x0_free, fixed_uncon = _split_free_fixed(
        param_names, par_init, par_fixed, tf
    )

    def _model_corrs_from_params(par_all_con: Dict[str, Array]) -> Array:
        par_base = _build_par_base(base, par_all_con)
        Cb = cor_base_stat(base=base, par_base=par_base, h=h, u=u)
        Cb = _cov_joint(Cb)

        par_lagr, lam = _build_par_lds(lagrangian, par_all_con)
        Cl = cor_lagr_lds(lagrangian=lagrangian, par_lagr=par_lagr, h_lds=h_lds)
        return (1.0 - lam) * Cb + lam * Cl

    def f(x_free: Array, ridge=ridge) -> Array:
        x_full = _assemble_full_uncon(x_free, free_names, fixed_uncon, param_names)
        par_all_con = _unconstrained_to_constrained_dict(x_full, param_names, tf)

        corrs_model = _model_corrs_from_params(par_all_con)
        corrs_stat = stationary_corr_from_model_corrs(
            corrs_model,
            n=n_loc,
            lag=lag,
            fp_method=fp_method,
            fp_control=fp_control,
            ridge=ridge,
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
    par_lagr_hat, lam_hat = _build_par_lds(lagrangian, par_all_hat)

    obj_val = float(f(x_star_free))

    notes_fp = {
        "fp_method": fp_method,
        "fp_control": {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in fp_control.items()
        },
        "ridge": float(ridge),
    }

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
                "lag": int(lag),
                "one_step_lds": True,
                **notes_fp,
            },
        }
    )
