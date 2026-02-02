from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


# Optimizers: prefer jaxopt LBFGS; fallback to optax Adam if needed
try:
    import jaxopt

    _HAVE_JAXOPT = True
except Exception:
    _HAVE_JAXOPT = False


try:
    import optax

    _HAVE_OPTAX = True
except Exception:
    _HAVE_OPTAX = False

from correlation import DTYPE, _as_dtype, cor_stat, cor_base_stat
from estimate_transform import Transform, make_transforms

Array = jax.Array


# -----------------------------------------------------------------------------
# Utilities: packing/unpacking + fixed parameters
# -----------------------------------------------------------------------------
def _tree_to_py(x: Any) -> Any:
    """Convert JAX arrays to python floats/lists for easy reticulate conversion."""
    if isinstance(x, (jax.Array, jnp.ndarray)):
        if x.shape == ():
            return float(x)
        return jnp.asarray(x).tolist()
    if isinstance(x, dict):
        return {k: _tree_to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_tree_to_py(v) for v in x]
    return x


@dataclass(frozen=True)
class ParamSpec:
    """
    Defines which parameters are optimized for a given estimation call.
    param_names: canonical ordered list of parameter names in the flat dict.
    """

    param_names: Tuple[str, ...]


def _pack_init_unconstrained(
    par_init_con: Dict[str, Any],
    param_names: Tuple[str, ...],
    tf: Dict[str, Transform],  # Dict[str, Transform]
) -> Array:
    vals = []
    for k in param_names:
        v_con = _as_dtype(par_init_con[k])
        vals.append(tf[k].inverse(v_con))
    return jnp.stack(vals).astype(DTYPE)


def _unconstrained_to_constrained_dict(
    x_uncon: Array,
    param_names: Tuple[str, ...],
    tf: Dict[str, Transform],
) -> Dict[str, Array]:
    x_uncon = _as_dtype(x_uncon)
    out: Dict[str, Array] = {}
    for i, k in enumerate(param_names):
        out[k] = tf[k].forward(x_uncon[i])
    return out


def _split_free_fixed(
    param_names: Tuple[str, ...],
    par_init_con: Dict[str, Any],
    par_fixed_con: Optional[Dict[str, Array]],
    tf: Dict[str, Transform],
) -> Tuple[Tuple[str, ...], Array, Dict[str, Array]]:
    """
    Returns:
      free_names: names we optimize
      x0_free: unconstrained init vector for free params
      fixed_uncon: dict of fixed param values in unconstrained space
    """
    fixed_uncon: Dict[str, Array] = {}
    free_names: List[str] = []

    for k in param_names:
        if par_fixed_con is not None and k in par_fixed_con:
            fixed_uncon[k] = tf[k].inverse(_as_dtype(par_fixed_con[k]))
        else:
            free_names.append(k)

    # Build x0 for free only
    x0_list = []
    for k in free_names:
        x0_list.append(tf[k].inverse(_as_dtype(par_init_con[k])))
    x0_free = (
        jnp.stack(x0_list).astype(DTYPE) if x0_list else jnp.zeros((0,), dtype=DTYPE)
    )

    return tuple(free_names), x0_free, fixed_uncon


def _assemble_full_uncon(
    x_free: Array,
    free_names: Tuple[str, ...],
    fixed_uncon: Dict[str, Array],
    param_names: Tuple[str, ...],
) -> Array:
    """
    Reconstruct full unconstrained vector x_full in canonical order.
    """
    # map free vector to dict
    free_map = {k: _as_dtype(x_free[i]) for i, k in enumerate(free_names)}
    full = []
    for k in param_names:
        if k in fixed_uncon:
            full.append(fixed_uncon[k])
        else:
            full.append(free_map[k])
    return jnp.stack(full).astype(DTYPE)


# -----------------------------------------------------------------------------
# Parameter structure adapters (match mcgf-style nested lists)
# -----------------------------------------------------------------------------
def _build_par_base(base: str, par_all_con: Dict[str, Array]) -> Dict[str, Any]:
    if base == "fs":
        # cor_fs(nugget, c, gamma, a, alpha, beta, h, u)
        return {
            "nugget": par_all_con["nugget"],
            "c": par_all_con["c"],
            "gamma": par_all_con["gamma"],
            "a": par_all_con["a"],
            "alpha": par_all_con["alpha"],
            "beta": par_all_con["beta"],
        }
    if base == "sep":
        return {
            "par_s": {
                "nugget": par_all_con["nugget"],
                "c": par_all_con["c"],
                "gamma": par_all_con["gamma"],
            },
            "par_t": {
                "a": par_all_con["a"],
                "alpha": par_all_con["alpha"],
            },
        }
    raise ValueError(f"Unknown base='{base}'")


def _build_par_lagr(
    lagrangian: str, par_all_con: Dict[str, Array]
) -> Tuple[Optional[Dict[str, Any]], Array]:
    if lagrangian == "none":
        return None, jnp.asarray(0.0, dtype=DTYPE)
    return (
        {
            "v1": par_all_con["v1"],
            "v2": par_all_con["v2"],
            "k": par_all_con["k"],
        },
        par_all_con["lam"],
    )


# -----------------------------------------------------------------------------
# Objective functions (WLS)
# -----------------------------------------------------------------------------
def _wls_loss(cor_model, cor_emp, eps: float = 1e-6):
    """
    JAX equivalent of mcgf:::obj_wls

    R:
      summand <- ((cor_emp - fitted)/(1 - fitted))^2
      summand[is.infinite(summand)] <- NA
      sum(summand, na.rm = TRUE)
    """
    fitted = cor_model
    emp = cor_emp

    denom = 1.0 - fitted
    n = fitted.shape[0]
    diag0 = jnp.eye(n, dtype=bool)
    if fitted.ndim == 3:
        diag0 = diag0[..., None] & (jnp.arange(fitted.shape[2]) == 0)

    use = ~diag0 & (denom != 0)

    numer = jnp.where(use, emp - fitted, 0.0)
    denom = jnp.where(use, denom, 1.0)  # never divide by 0

    return jnp.sum((numer / denom) ** 2)


def _obj_base_wls(
    x_free: Array,
    *,
    base: str,
    h: Array,
    u: Array,
    cor_emp: Array,
    param_names: Tuple[str, ...],
    free_names: Tuple[str, ...],
    fixed_uncon: Dict[str, Array],
    tf: Dict[str, Transform],
) -> Array:
    x_full = _assemble_full_uncon(x_free, free_names, fixed_uncon, param_names)
    par_all = _unconstrained_to_constrained_dict(x_full, param_names, tf)
    par_base = _build_par_base(base, par_all)

    cor_model = cor_stat(
        base=base,
        lagrangian="none",
        par_base=par_base,
        par_lagr=None,
        lam=0.0,
        h=h,
        u=u,
        base_fixed=False,
    )
    return _wls_loss(cor_model, cor_emp)


def _obj_lagr_wls(
    x_free: Array,
    *,
    lagrangian: str,
    base_corr: Array,
    h1: Array,
    h2: Array,
    u: Array,
    cor_emp: Array,
    param_names: Tuple[str, ...],
    free_names: Tuple[str, ...],
    fixed_uncon: Dict[str, Array],
    tf: Dict[str, Transform],
) -> Array:
    x_full = _assemble_full_uncon(x_free, free_names, fixed_uncon, param_names)
    par_all = _unconstrained_to_constrained_dict(x_full, param_names, tf)
    par_lagr, lam = _build_par_lagr(lagrangian, par_all)

    cor_model = cor_stat(
        base=_as_dtype(base_corr),
        lagrangian=lagrangian,
        par_base=None,
        par_lagr=par_lagr,
        lam=lam,
        h1=h1,
        h2=h2,
        u=u,
        base_fixed=True,
    )
    return _wls_loss(cor_model, cor_emp)


# -----------------------------------------------------------------------------
# Optimizer runners
# -----------------------------------------------------------------------------
def _run_lbfgs(value_and_grad_fn, x0: Array, maxiter: int = 2000) -> Dict[str, Any]:
    solver = jaxopt.LBFGS(fun=value_and_grad_fn, value_and_grad=True, maxiter=maxiter)
    res = solver.run(x0)
    x_star = res.params
    state = res.state
    return {
        "x_star": x_star,
        "n_iter": int(state.iter_num),
        "converged": bool(state.error <= solver.tol),
        "message": "ok" if bool(state.error <= solver.tol) else "not_converged",
    }


def _run_adam(
    value_and_grad_fn, x0: Array, maxiter: int = 2000, lr: float = 1e-2
) -> Dict[str, Any]:
    if not _HAVE_OPTAX:
        raise RuntimeError("Need jaxopt or optax installed for optimization.")
    opt = optax.adam(lr)
    opt_state = opt.init(x0)

    def step(carry, _):
        x, opt_state = carry
        val, g = value_and_grad_fn(x)
        updates, opt_state2 = opt.update(g, opt_state, x)
        x2 = optax.apply_updates(x, updates)
        return (x2, opt_state2), val

    (xT, _), vals = jax.lax.scan(step, (x0, opt_state), xs=None, length=maxiter)
    return {
        "x_star": xT,
        "n_iter": int(maxiter),
        "converged": True,  # ADAM: treat as "ran maxiter"
        "message": "ok_adam",
        "trace": vals,
    }


def _run_adamw(
    value_and_grad_fn,
    x0: Array,
    maxiter: int = 2000,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    if not _HAVE_OPTAX:
        raise RuntimeError("Need optax installed for AdamW optimization.")
    opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = opt.init(x0)

    def step(carry, _):
        x, opt_state = carry
        val, g = value_and_grad_fn(x)
        updates, opt_state2 = opt.update(g, opt_state, x)
        x2 = optax.apply_updates(x, updates)
        return (x2, opt_state2), val

    (xT, _), vals = jax.lax.scan(step, (x0, opt_state), xs=None, length=maxiter)
    return {
        "x_star": xT,
        "n_iter": int(maxiter),
        "converged": True,  # treat as "ran maxiter"
        "message": "ok_adamw",
        "trace": vals,
    }


# -----------------------------------------------------------------------------
# Public entrypoints for R routing
# -----------------------------------------------------------------------------


def _run_optimizer(
    value_and_grad_fn,
    x0: Array,
    *,
    method: str = "auto",
    control: Dict[str, Any],
) -> Dict[str, Any]:
    m = (control.get("method") or method or "auto").lower()

    # normalize aliases
    if m in ("l-bfgs", "l-bfgs-b"):
        m = "lbfgs"
    if m in ("optax", "optax_adam"):
        m = "adam"
    if m in ("wadam", "adamw_optax"):
        m = "adamw"

    if m == "auto":
        # prefer LBFGS if available, else Adam (or AdamW if you prefer)
        if _HAVE_JAXOPT:
            return _run_lbfgs(
                value_and_grad_fn,
                x0,
                maxiter=int(control.get("maxiter", 2000)),
            )
        # fallback default when no jaxopt:
        return _run_adamw(
            value_and_grad_fn,
            x0,
            maxiter=int(control.get("maxiter_adam", control.get("maxiter", 2000))),
            lr=float(control.get("lr", 1e-2)),
            weight_decay=float(control.get("weight_decay", 1e-4)),
        )

    if m == "lbfgs":
        if not _HAVE_JAXOPT:
            raise RuntimeError("method='lbfgs' requested but jaxopt is not installed.")
        return _run_lbfgs(
            value_and_grad_fn,
            x0,
            maxiter=int(control.get("maxiter", 2000)),
        )

    if m == "adam":
        return _run_adam(
            value_and_grad_fn,
            x0,
            maxiter=int(control.get("maxiter_adam", control.get("maxiter", 2000))),
            lr=float(control.get("lr", 1e-2)),
        )

    if m == "adamw":
        return _run_adamw(
            value_and_grad_fn,
            x0,
            maxiter=int(control.get("maxiter_adam", control.get("maxiter", 2000))),
            lr=float(control.get("lr", 1e-2)),
            weight_decay=float(control.get("weight_decay", 1e-4)),
        )

    raise ValueError(f"Unknown method='{m}'. Use 'auto', 'lbfgs', 'adam', or 'adamw'.")


def jax_fit_base_one(
    *,
    base: str,  # "sep" or "fs"
    h: Any,
    u: Any,
    cor_emp: Any,
    par_init: Dict[str, Any],
    par_fixed: Optional[Dict[str, Any]] = None,
    method: str = "auto",
    control: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Transform]] = None,
) -> Dict[str, Any]:
    """
    Base-only WLS fit. Returns an 'optim-like' result dict compatible with R wrappers.
    """
    control = control or {}
    maxiter = int(control.get("maxiter", 200))

    h = _as_dtype(h)
    u = _as_dtype(u)
    cor_emp = _as_dtype(cor_emp)

    tf = transforms if transforms is not None else make_transforms()

    # Parameters needed for base models
    base_param_names = ("nugget", "c", "gamma", "a", "alpha", "beta")
    free_names, x0_free, fixed_uncon = _split_free_fixed(
        base_param_names, par_init, par_fixed, tf
    )

    def f(x):
        return _obj_base_wls(
            x,
            base=base,
            h=h,
            u=u,
            cor_emp=cor_emp,
            param_names=base_param_names,
            free_names=free_names,
            fixed_uncon=fixed_uncon,
            tf=tf,
        )

    value_and_grad = jax.value_and_grad(f)

    opt_res = _run_optimizer(value_and_grad, x0_free, method=method, control=control)

    x_star_free = opt_res["x_star"]
    x_star_full = _assemble_full_uncon(
        x_star_free, free_names, fixed_uncon, base_param_names
    )
    par_all_hat = _unconstrained_to_constrained_dict(x_star_full, base_param_names, tf)
    par_base_hat = _build_par_base(base, par_all_hat)

    obj_val = float(f(x_star_free))

    return _tree_to_py(
        {
            "par_base": par_base_hat,
            "objective": obj_val,
            "converged": opt_res["converged"],
            "n_iter": opt_res["n_iter"],
            "message": opt_res.get("message", ""),
            "backend": "jax",
        }
    )


def jax_fit_lagr_one(
    *,
    lagrangian: str,  # "lagr_tri"|"lagr_exp"|"lagr_askey"
    base_corr: Any,  # dense base correlation array on same grid as cor_emp
    h1: Any,
    h2: Any,
    u: Any,
    cor_emp: Any,
    par_init: Dict[str, Any],
    par_fixed: Optional[Dict[str, Any]] = None,
    method: str = "auto",
    control: Optional[Dict[str, Any]] = None,
    transforms: Optional[Dict[str, Transform]] = None,
) -> Dict[str, Any]:
    """
    Lagrangian-only WLS fit with base fixed (regime-switching friendly).
    """
    control = control or {}
    maxiter = int(control.get("maxiter", 200))

    base_corr = _as_dtype(base_corr)
    h1 = _as_dtype(h1)
    h2 = _as_dtype(h2)
    u = _as_dtype(u)
    cor_emp = _as_dtype(cor_emp)

    tf = transforms if transforms is not None else make_transforms()

    lagr_param_names = ("lam", "v1", "v2", "k")
    free_names, x0_free, fixed_uncon = _split_free_fixed(
        lagr_param_names, par_init, par_fixed, tf
    )

    def f(x):
        return _obj_lagr_wls(
            x,
            lagrangian=lagrangian,
            base_corr=base_corr,
            h1=h1,
            h2=h2,
            u=u,
            cor_emp=cor_emp,
            param_names=lagr_param_names,
            free_names=free_names,
            fixed_uncon=fixed_uncon,
            tf=tf,
        )

    value_and_grad = jax.value_and_grad(f)

    opt_res = _run_optimizer(value_and_grad, x0_free, method=method, control=control)

    x_star_free = opt_res["x_star"]
    x_star_full = _assemble_full_uncon(
        x_star_free, free_names, fixed_uncon, lagr_param_names
    )
    par_all_hat = _unconstrained_to_constrained_dict(x_star_full, lagr_param_names, tf)
    par_lagr_hat, lam_hat = _build_par_lagr(lagrangian, par_all_hat)

    obj_val = float(f(x_star_free))

    return _tree_to_py(
        {
            "par_lagr": par_lagr_hat,
            "lam": lam_hat,
            "objective": obj_val,
            "converged": opt_res["converged"],
            "n_iter": opt_res["n_iter"],
            "message": opt_res.get("message", ""),
            "backend": "jax",
        }
    )


def compute_base_corr(
    *,
    base: str,
    par_base: Dict[str, Any],
    h: Any,
    u: Any,
) -> Dict[str, Any]:
    """
    Convenience: compute and return base correlation array so R can cache it for regime switching.
    """
    h = _as_dtype(h)
    u = _as_dtype(u)

    cor_base = cor_stat(
        base=base,
        lagrangian="none",
        par_base=par_base,
        par_lagr=None,
        lam=0.0,
        h=h,
        u=u,
        base_fixed=False,
    )

    return _tree_to_py({"base_corr": cor_base})


# ------------------------------------------------------------------------------
# Check inputs' shape
# ------------------------------------------------------------------------------
def _assert_same_shape(name_a, a, name_b, b):
    if a.shape != b.shape:
        raise ValueError(f"{name_a}.shape={a.shape} != {name_b}.shape={b.shape}")


def _check_base_inputs(h, u, cor_emp, w=None):
    _assert_same_shape("h", h, "cor_emp", cor_emp)
    _assert_same_shape("u", u, "cor_emp", cor_emp)
    if w is not None:
        _assert_same_shape("weights", w, "cor_emp", cor_emp)


def _check_lagr_inputs(h1, h2, u, cor_emp, base_corr, w=None):
    _assert_same_shape("h1", h1, "cor_emp", cor_emp)
    _assert_same_shape("h2", h2, "cor_emp", cor_emp)
    _assert_same_shape("u", u, "cor_emp", cor_emp)
    _assert_same_shape("base_corr", base_corr, "cor_emp", cor_emp)
    if w is not None:
        _assert_same_shape("weights", w, "cor_emp", cor_emp)
