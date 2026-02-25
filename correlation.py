from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Any, Dict, Literal

Array = jax.Array
DTYPE = jnp.float32


def _as_dtype(x: Any) -> Array:
    return jnp.asarray(x, dtype=DTYPE)


# ------------------------------------------------------------------------------#
# Add and set spatial nuggets
# ------------------------------------------------------------------------------#
@jax.jit
def add_nugget(x: Array, nugget: Array) -> Array:
    x = _as_dtype(x)
    nugget = _as_dtype(nugget)

    if x.ndim == 2:
        n = x.shape[0]
        I = jnp.eye(n, dtype=DTYPE)
        return (1.0 - nugget) * x + nugget * I

    if x.ndim == 3:
        n = x.shape[0]
        I = jnp.eye(n, dtype=DTYPE)[:, :, None]  # broadcast over T
        return (1.0 - nugget) * x + nugget * I

    raise ValueError("add_nugget expects 2D (n,n) or 3D (n,n,T).")


@jax.jit
def set_nugget(x: Array, nugget: Array, set_to: Array) -> Array:
    x = _as_dtype(x)
    set_to = _as_dtype(set_to)

    # Allow (n,n) set_to for (n,n,T) x
    if x.ndim == 3 and set_to.ndim == 2 and set_to.shape == x.shape[:2]:
        set_to = set_to[:, :, None]  # (n,n,1)

    # Make set_to match x exactly via broadcasting
    set_to = jnp.broadcast_to(set_to, x.shape)

    nugget = _as_dtype(nugget)
    corr = (1.0 - nugget) * x

    if x.ndim == 2:
        n = x.shape[0]
        mask = jnp.eye(n, dtype=bool)
        return jnp.where(mask, set_to, corr)

    if x.ndim == 3:
        n = x.shape[0]
        mask = jnp.eye(n, dtype=bool)[:, :, None]
        return jnp.where(mask, set_to, corr)

    raise ValueError("set_nugget expects 2D (n,n) or 3D (n,n,T).")


@jax.jit
def set_nugget_with_mask(
    x: Array, nugget: Array, set_to: Array, diag_mask_bool: Array
) -> Array:
    # diag_mask_bool: (n,n) or (n,n,1) broadcastable to x
    nugget = _as_dtype(nugget)
    corr = (1.0 - nugget) * x
    return jnp.where(diag_mask_bool, set_to, corr)


# ------------------------------------------------------------------------------#
# Base kernels (dense)
# ------------------------------------------------------------------------------#
@jax.jit
def cor_cauchy(
    x: Array, a: Array, alpha: Array, nu: Array = 1.0, nugget: Array = 0.0
) -> Array:
    """
    Cauchy kernel:
        base = (a*|x|^(2*alpha) + 1)^(-nu)
        if nugget>0: add_nugget(base, nugget)
    """
    x = _as_dtype(x)
    a = _as_dtype(a)
    alpha = _as_dtype(alpha)
    nu = _as_dtype(nu)
    nugget = _as_dtype(nugget)

    base = (a * _abs_pow_safe(x, 2.0 * alpha) + 1.0) ** (-nu)
    return (
        add_nugget(base, nugget)
        if base.ndim in (2, 3)
        else (1.0 - nugget) * base + nugget
    )


@jax.jit
def cor_exp(x: Array, c: Array, gamma: Array = 0.5, nugget: Array = 0.0) -> Array:
    """
    Exponential powered kernel:
        base = exp(-c * |x|^(2*gamma))
        if nugget>0: add_nugget(base, nugget)
    x can be (n,n) or (n,n,T) or any dense lag array.
    """
    x = _as_dtype(x)
    c = _as_dtype(c)
    gamma = _as_dtype(gamma)
    nugget = _as_dtype(nugget)

    base = jnp.exp(-c * jnp.abs(x) ** (2.0 * gamma))
    return (
        add_nugget(base, nugget)
        if base.ndim in (2, 3)
        else (1.0 - nugget) * base + nugget
    )


# ------------------------------------------------------------------------------#
# Lagrangian kernels (dense)
# ------------------------------------------------------------------------------#
@jax.jit
def cor_lagr_askey(
    h1: Array, h2: Array, u: Array, v1: Array, v2: Array, k: Array = 2.0, eps=1e-6
) -> Array:
    """
    Askey Lagrangian:
        max(1 - ||h - u v||/(k||v||), 0)^(3/2)
    """
    h1 = _as_dtype(h1)
    h2 = _as_dtype(h2)
    u = _as_dtype(u)
    v1 = _as_dtype(v1)
    v2 = _as_dtype(v2)
    k = _as_dtype(k)

    v_norm = jnp.sqrt(v1 * v1 + v2 * v2)
    v_norm = jnp.maximum(v_norm, eps)  # avoid div by zero
    h_vu_norm = jnp.sqrt((h1 - v1 * u) ** 2 + (h2 - v2 * u) ** 2)
    lagr = 1.0 - h_vu_norm / (k * v_norm)
    return jnp.maximum(lagr, 0.0) ** 1.5


@jax.jit
def cor_lagr_exp(
    h1: Array, h2: Array, u: Array, v1: Array, v2: Array, k: Array = 2.0, eps=1e-6
) -> Array:
    """
    Exponential Lagrangian:
      exp( - ||h - u v|| / (k||v||) )
    """
    h1 = _as_dtype(h1)
    h2 = _as_dtype(h2)
    u = _as_dtype(u)
    v1 = _as_dtype(v1)
    v2 = _as_dtype(v2)
    k = _as_dtype(k)

    v_norm = jnp.sqrt(v1 * v1 + v2 * v2)
    v_norm = jnp.maximum(v_norm, eps)  # avoid div by zero
    h_vu_norm = jnp.sqrt((h1 - v1 * u) ** 2 + (h2 - v2 * u) ** 2)
    return jnp.exp(-h_vu_norm / (k * v_norm))


@jax.jit
def cor_lagr_tri(
    h1: Array, h2: Array, u: Array, v1: Array, v2: Array, k: Array = 2.0, eps=1e-6
) -> Array:
    """
    Triangular Lagrangian:
        max(1 - | (h·v)/||v|| - ||v||u | / (k||v||), 0)
    """
    h1 = _as_dtype(h1)
    h2 = _as_dtype(h2)
    u = _as_dtype(u)
    v1 = _as_dtype(v1)
    v2 = _as_dtype(v2)
    k = _as_dtype(k)

    v_norm = jnp.sqrt(v1 * v1 + v2 * v2)
    v_norm = jnp.maximum(v_norm, eps)  # avoid div by zero
    term = jnp.abs((h1 * v1 + h2 * v2) / v_norm - v_norm * u)
    lagr = 1.0 - term / (k * v_norm)
    return jnp.maximum(lagr, 0.0)


# ------------------------------------------------------------------------------#
# Composite kernels (dense)
# ------------------------------------------------------------------------------#


def _abs_pow_safe(x, p, tiny=1e-12):
    """Compute |x|^p with finite grads w.r.t. p at x=0."""
    ax = jnp.abs(x)
    # keep log finite everywhere (even where ax==0)
    logax = jnp.log(jnp.maximum(ax, tiny))
    y = jnp.exp(p * logax)
    # enforce exact 0 at ax==0 (and keep grads finite)
    return jnp.where(ax == 0, 0.0, y)


@jax.jit(static_argnames=("spatial", "temporal"))
def cor_sep(
    h: Array,
    u: Array,
    spatial: Literal["exp", "cauchy"],
    temporal: Literal["exp", "cauchy"],
    par_s: Dict[str, Array],
    par_t: Dict[str, Array],
) -> Array:
    """
    Separable model:
        C = C_t(u) * C_s(h)
    Nugget handling (diagonal-only):
        corr = set_nugget(C, nugget, set_to=C_t(u))
    which matches the common mcgf separable base behavior you’ve been using.

    Notes:
    - Nugget is applied via set_nugget(), i.e., diagonal replaced by temporal diagonal.
    - This works for (n,n) and (n,n,T) as long as h and u share shape.
    """
    h = _as_dtype(h)
    u = _as_dtype(u)
    nugget = _as_dtype(par_s.get("nugget", 0.0))

    # temporal
    if temporal == "cauchy":
        Ct = cor_cauchy(
            u,
            a=par_t["a"],
            alpha=par_t["alpha"],
            nu=_as_dtype(par_t.get("nu", 1.0)),
            nugget=0.0,
        )
    elif temporal == "exp":
        Ct = cor_exp(
            u,
            c=par_t["c"],
            gamma=_as_dtype(par_t.get("gamma", 0.5)),
            nugget=0.0,
        )
    else:
        raise ValueError("cor_sep: temporal must be 'exp' or 'cauchy'.")

    # spatial
    if spatial == "exp":
        Cs = cor_exp(
            h,
            c=par_s["c"],
            gamma=_as_dtype(par_s.get("gamma", 0.5)),
            nugget=0.0,
        )
    elif spatial == "cauchy":
        Cs = cor_cauchy(
            h,
            a=par_s["a"],
            alpha=par_s["alpha"],
            nu=_as_dtype(par_s.get("nu", 1.0)),
            nugget=0.0,
        )
    else:
        raise ValueError("cor_sep: spatial must be 'exp' or 'cauchy'.")

    C = Ct * Cs
    return set_nugget(C, nugget=nugget, set_to=Ct)


@jax.jit
def cor_fs(
    h: Array,
    u: Array,
    c: Array,
    a: Array,
    alpha: Array,
    gamma: Array = 0.5,
    nugget: Array = 0.0,
    beta: Array = 0.0,
) -> Array:
    """
    Fully symmetric model (dense), with diagonal-only nugget.
    Common form used in mcgf literature:
        tu = a|u|^(2alpha) + 1
        C  = 1/tu * [ (1-nugget) * exp( - c |h|^(2gamma) / tu^(beta*gamma) ) + nugget * I ]
    Diagonal-only nugget: add_nugget(inner_exp, nugget) inside the bracket.
    """
    h = _as_dtype(h)
    u = _as_dtype(u)

    nugget = _as_dtype(nugget)
    c = _as_dtype(c)
    gamma = _as_dtype(gamma)
    a = _as_dtype(a)
    alpha = _as_dtype(alpha)
    beta = _as_dtype(beta)

    pu = _abs_pow_safe(u, 2.0 * alpha)
    ph = _abs_pow_safe(h, 2.0 * gamma)
    tu = a * pu + 1.0
    scale = tu ** (beta * gamma)
    exp_term = jnp.exp(-c * ph / scale)
    C_raw = exp_term / tu
    set_to = 1.0 / tu
    return set_nugget(C_raw, nugget=nugget, set_to=set_to)


@jax.jit(static_argnames=("base"))
def cor_base_stat(
    *,
    base: Literal["sep", "fs"],
    par_base: Dict[str, Any],
    h: Array,
    u: Array,
) -> Array:
    """
    Compute base via selected base model
    """
    if par_base is None:
        raise ValueError("cor_stat: par_base required when base_fixed=False.")
    if h is None or u is None:
        raise ValueError("cor_stat: h and u required when base_fixed=False.")

    h = _as_dtype(h)
    u = _as_dtype(u)

    if base == "sep":
        pb = {
            "par_s": par_base["par_s"],
            "par_t": par_base["par_t"],
            "h": h,
            "u": u,
            "spatial": "exp",
            "temporal": "cauchy",
        }
        return cor_sep(**pb)

    if base == "fs":
        pb = dict(par_base)
        pb.update({"h": h, "u": u})
        return cor_fs(**pb)

    raise ValueError("cor_stat: base must be 'sep' or 'fs' when base_fixed=False.")


@jax.jit(static_argnames=("lagrangian",))
def cor_lagr_stat(
    *,
    lagrangian: Literal["lagr_tri", "lagr_exp", "lagr_askey"],
    par_lagr: Dict[str, Any],
    h1: Array,
    h2: Array,
    u: Array,
) -> Array:
    if par_lagr is None:
        raise ValueError("cor_lagr_stat: par_lagr is required.")
    if h1 is None or h2 is None or u is None:
        raise ValueError("cor_lagr_stat: h1, h2, u are required.")

    h1 = _as_dtype(h1)
    h2 = _as_dtype(h2)
    u = _as_dtype(u)

    v1 = _as_dtype(par_lagr["v1"])
    v2 = _as_dtype(par_lagr["v2"])
    k = _as_dtype(par_lagr.get("k", 2.0))

    if lagrangian == "lagr_tri":
        return cor_lagr_tri(h1=h1, h2=h2, u=u, v1=v1, v2=v2, k=k)
    if lagrangian == "lagr_exp":
        return cor_lagr_exp(h1=h1, h2=h2, u=u, v1=v1, v2=v2, k=k)
    if lagrangian == "lagr_askey":
        return cor_lagr_askey(h1=h1, h2=h2, u=u, v1=v1, v2=v2, k=k)

    raise ValueError("cor_lagr_stat: invalid lagrangian choice.")


@jax.jit(static_argnames=("lagrangian",))
def cor_lagr_lds(
    *,
    lagrangian: Literal["exp"],
    par_lagr: Dict[str, Any],
    h_lds: Array,
) -> Array:
    if par_lagr is None:
        raise ValueError("cor_lagr_lds: par_lagr is required.")
    if h_lds is None:
        raise ValueError("cor_lagr_lds: h_lds is required.")

    h_lds = _as_dtype(h_lds)
    nugget = _as_dtype(par_lagr["nugget"])
    c = _as_dtype(par_lagr["c"])
    gamma = _as_dtype(par_lagr["gamma"])

    if lagrangian == "exp":
        return cor_exp(x=h_lds, nugget=nugget, c=c, gamma=gamma)

    raise ValueError("cor_lagr_lds: invalid lagrangian choice.")


@jax.jit(static_argnames=("base", "lagrangian"))
def _cor_stat_model(
    *,
    base: Literal["sep", "fs"],
    lagrangian: Literal["none", "lagr_tri", "lagr_exp", "lagr_askey"],
    par_base: Dict[str, Any],
    par_lagr: Dict[str, Any] | None,
    lam: Array | float,
    h: Array | None = None,
    h1: Array | None = None,
    h2: Array | None = None,
    u: Array | None = None,
) -> Array:
    """
    JAX-jitted correlation computation where the base correlation is specified
    by a model selector ("sep" or "fs").

    Static arguments
    ----------------
    base
        Base correlation model selector.
    lagrangian
        Lagrangian model selector.

    Behavior
    --------
    - If `lagrangian == "none"`, returns the base correlation only.
    - Otherwise computes
          C = (1 - lambda) * C_base + lambda * C_lagr
      where both components are computed internally.

    Notes
    -----
    - `base` and `lagrangian` are static to enable efficient specialization.
    - All arrays must have static shapes compatible with JAX tracing.
    """
    lam = _as_dtype(lam)

    if lagrangian == "none":
        return cor_base_stat(base=base, par_base=par_base, h=h, u=u)

    Cb = cor_base_stat(base=base, par_base=par_base, h=h, u=u)
    Cl = cor_lagr_stat(lagrangian=lagrangian, par_lagr=par_lagr, h1=h1, h2=h2, u=u)
    return (1.0 - lam) * Cb + lam * Cl


@jax.jit(static_argnames=("lagrangian",))
def _cor_stat_fixed(
    *,
    base: Array,
    lagrangian: Literal["lagr_tri", "lagr_exp", "lagr_askey"],
    par_lagr: Dict[str, Any],
    lam: Array | float,
    h1: Array,
    h2: Array,
    u: Array,
) -> Array:
    """
    JAX-jitted correlation computation with a precomputed base correlation matrix.

    Static arguments
    ----------------
    lagrangian
        Lagrangian model selector.

    Behavior
    --------
    Computes the mixture
        C = (1 - lambda) * C_base + lambda * C_lagr
    where `C_base` is supplied directly as a dense array.

    Notes
    -----
    - `base` is *not* static to avoid hashing errors with JAX arrays.
    - Shape compatibility between `base` and `h1` is assumed to be validated
      by the caller.
    """
    lam = _as_dtype(lam)

    Cb = _as_dtype(base)
    Cl = cor_lagr_stat(
        lagrangian=lagrangian,
        par_lagr=par_lagr,
        h1=h1,
        h2=h2,
        u=u,
    )
    return (1.0 - lam) * Cb + lam * Cl


def cor_stat(
    *,
    base: Literal["sep", "fs"] | Array = "sep",
    lagrangian: Literal["none", "lagr_tri", "lagr_exp", "lagr_askey"] = "none",
    par_base: Dict[str, Any] | None = None,
    par_lagr: Dict[str, Any] | None = None,
    lam: Array | float = 0.0,
    h: Array | None = None,
    h1: Array | None = None,
    h2: Array | None = None,
    u: Array | None = None,
    base_fixed: bool = False,
) -> Array:
    """
    General stationary correlation model with optional Lagrangian mixture.

        C = (1 - lambda) * C_base + lambda * C_lagr

    This is a Python-level dispatcher that selects the appropriate JAX-jitted
    implementation depending on whether the base correlation is supplied
    as a *model selector* ("sep" / "fs") or as a *precomputed dense matrix*.

    Parameters
    ----------
    base
        If `base_fixed = False`, a string selecting the base correlation model:
        - "sep" : separable spatial–temporal model
        - "fs"  : full-space model

        If `base_fixed = True`, a dense correlation array C_base.
    lagrangian
        Lagrangian model selector:
        - "none"        : base correlation only
        - "lagr_tri"    : triangular kernel
        - "lagr_exp"    : exponential kernel
        - "lagr_askey"  : Askey kernel
    par_base
        Dictionary of base-model parameters. Required when `base_fixed = False`.
    par_lagr
        Dictionary of Lagrangian parameters (e.g., v1, v2, k).
        Required when `lagrangian != "none"`.
    lam
        Mixture weight in [0, 1]. Ignored when `lagrangian = "none"`.
    h
        Spatial distance array for base correlation.
    h1, h2
        Directional distance arrays for Lagrangian correlation.
    u
        Temporal lag array.
    base_fixed
        If True, `base` is interpreted as a dense correlation matrix C_base.

    Returns
    -------
    Array
        Correlation array with the same shape as `h` / `h1`.

    Notes
    -----
    - This wrapper is *not* jitted. It performs validation and dispatch only.
    - Two separate jitted kernels are used internally:
        * `_cor_stat_model` for model-selected base correlations
        * `_cor_stat_fixed` for precomputed base correlations
    - This design avoids passing non-hashable JAX arrays as static arguments.
    """
    if base_fixed is True:
        if not isinstance(base, (jax.Array, jnp.ndarray)):
            raise TypeError("base_fixed=True requires `base` to be a dense array.")

        if lagrangian == "none":
            raise ValueError("cannot supply `base` when lagrangian='none'")

        if par_lagr is None:
            raise ValueError("par_lagr required when base_fixed=True")

        if h1 is None or h2 is None or u is None:
            raise ValueError("h1, h2, u are required when base_fixed=True")

        return _cor_stat_fixed(
            base=base,
            lagrangian=lagrangian,
            par_lagr=par_lagr,
            lam=lam,
            h1=h1,
            h2=h2,
            u=u,
        )

    if not isinstance(base, str):
        raise TypeError("base_fixed=False requires `base` to be 'sep' or 'fs'")

    if par_base is None:
        raise ValueError("par_base required when base_fixed=False")

    if lagrangian != "none" and par_lagr is None:
        raise ValueError("par_lagr required when lagrangian != 'none'")

    if lagrangian != "none" and (h1 is None or h2 is None or u is None):
        raise ValueError("h1, h2, u are required when lagrangian != 'none'")

    return _cor_stat_model(
        base=base,
        lagrangian=lagrangian,
        par_base=par_base,
        par_lagr=par_lagr,
        lam=lam,
        h=h,
        h1=h1,
        h2=h2,
        u=u,
    )
