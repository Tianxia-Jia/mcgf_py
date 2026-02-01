from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import nn

Array = jax.Array

from correlation import DTYPE, _as_dtype

# -------------------------
# helpers: safe numerics
# -------------------------
def _clamp01(x: Array, eps: float = 1e-8) -> Array:
    # Keep probabilities away from {0,1} to avoid inf in logit, etc.
    return jnp.clip(x, eps, 1.0 - eps)


# -------------------------
# Parameter transforms
# -------------------------
def positive(x_uncon: Array, eps: float) -> Array:
    """
    R -> (eps, +inf)
    Uses softplus for stable positivity.
    """
    x_uncon = jnp.asarray(x_uncon)
    return nn.softplus(x_uncon) + eps


def positive_inv(x_pos: Array, eps: float = 1e-8) -> Array:
    """
    (eps, +inf) -> R
    Inverse of softplus + eps:
      y = softplus(x) + eps
      softplus(x) = y - eps
      x = softplus^{-1}(y-eps) = log(exp(y-eps) - 1)
    """
    x_pos = jnp.asarray(x_pos)
    y = jnp.maximum(x_pos - eps, jnp.asarray(1e-30, dtype=x_pos.dtype))
    # stable-ish: log(expm1(y)) is better than log(exp(y)-1) when y small
    return jnp.log(jnp.expm1(y))


def lower_bounded(x_uncon: Array, lo: float, eps: float = 1e-8) -> Array:
    """
    R -> (lo+eps, +inf)
    """
    return lo + positive(x_uncon, eps=eps)


def lower_bounded_inv(x_val: Array, lo: float, eps: float = 1e-8) -> Array:
    """
    (lo+eps, +inf) -> R
    """
    return positive_inv(jnp.asarray(x_val) - lo, eps=eps)


def bounded(x_uncon: Array, lo: float, hi: float, eps: float = 1e-8) -> Array:
    """
    R -> (lo, hi) (open interval, with eps margin)
    Uses sigmoid with clamping away from boundaries.
    """
    x_uncon = jnp.asarray(x_uncon)
    s = jax.nn.sigmoid(x_uncon)  # (0,1)
    s = _clamp01(s, eps=eps)  # (eps,1-eps)
    return lo + (hi - lo) * s


def bounded_inv(x_val: Array, lo: float, hi: float, eps: float = 1e-8) -> Array:
    """
    (lo, hi) -> R
    Inverse of lo + (hi-lo)*sigmoid(x).
    """
    x_val = jnp.asarray(x_val)
    s = (x_val - lo) / (hi - lo)
    s = _clamp01(s, eps=eps)
    return jax.scipy.special.logit(s)


def unit_interval(x_uncon: Array, eps: float = 1e-8) -> Array:
    """
    R -> (0,1), stabilized
    """
    return bounded(x_uncon, lo=0.0, hi=1.0, eps=eps)


def unit_interval_inv(x_val: Array, eps: float = 1e-8) -> Array:
    """
    (0,1) -> R
    """
    return bounded_inv(x_val, lo=0.0, hi=1.0, eps=eps)


# ------------------------------------------------------------------------------
# A transform registry
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Transform:
    forward: Callable[[Array], Array]
    inverse: Callable[[Array], Array]


def make_transforms_tri(eps: float = 1e-8) -> Dict[str, Transform]:
    """
    Default transform choices for correlation-model parameters.
    """
    return {
        # spatial nugget effect in (0,1)
        "nugget": Transform(
            forward=lambda x: unit_interval(x, eps=eps),
            inverse=lambda y: unit_interval_inv(y, eps=eps),
        ),
        # c > 0, but restricted to (0,0.5] for numerical stability
        "c": Transform(
            forward=lambda x: bounded(x, lo=0.0, hi=0.5, eps=eps),
            inverse=lambda y: bounded_inv(y, lo=0.0, hi=0.5, eps=eps),
        ),
        # gamma in (0,0.5]
        "gamma": Transform(
            forward=lambda x: bounded(x, lo=0.0, hi=0.5, eps=eps),
            inverse=lambda y: bounded_inv(y, lo=0.0, hi=0.5, eps=eps),
        ),
        # a > 0
        "a": Transform(
            forward=lambda x: positive(x, eps=eps),
            inverse=lambda y: positive_inv(y, eps=eps),
        ),
        # alpha in (0,1]
        "alpha": Transform(
            forward=lambda x: unit_interval(x, eps=eps),
            inverse=lambda y: unit_interval_inv(y, eps=eps),
        ),
        # beta in [0,1]
        "beta": Transform(
            forward=lambda x: unit_interval(x, eps=eps),
            inverse=lambda y: unit_interval_inv(y, eps=eps),
        ),
        # mixture weight lambda in (0,1)
        "lam": Transform(
            forward=lambda x: unit_interval(x, eps=eps),
            inverse=lambda y: unit_interval_inv(y, eps=eps),
        ),
        # zonal wind (0,1)
        "v1": Transform(
            forward=lambda x: x,
            inverse=lambda y: y,
        ),
        # meridional wind (0,1)
        "v2": Transform(
            forward=lambda x: x,
            inverse=lambda y: y,
        ),
        # k > 0
        "k": Transform(
            forward=lambda x: positive(x, eps=eps),
            inverse=lambda y: positive_inv(y, eps=eps),
        ),
    }


def transform_params(
    theta_uncon: Dict[str, Array],
    spec: Dict[str, Transform],
) -> Dict[str, Array]:
    """
    Map unconstrained -> constrained.
    Keys in theta_uncon must be in spec.
    """
    out = {}
    for k, v in theta_uncon.items():
        if k not in spec:
            raise KeyError(f"Missing transform spec for key '{k}'")
        out[k] = spec[k].forward(v)
    return out


def inverse_transform_params(
    theta_con: Dict[str, Array],
    spec: Dict[str, Transform],
) -> Dict[str, Array]:
    """
    Map constrained -> unconstrained.
    """
    out = {}
    for k, v in theta_con.items():
        if k not in spec:
            raise KeyError(f"Missing transform spec for key '{k}'")
        out[k] = spec[k].inverse(v)
    return out


# ------------------------------------------------------------------------------
# A transform registry for parameters for base and lagrangian models
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamTransforms:
    base: Dict[str, Transform]
    lagr: Dict[str, Transform]


def transform_model_params(
    theta_base_uncon: Dict[str, Array],
    theta_lagr_uncon: Dict[str, Array],
    tfs: ParamTransforms,
) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    par_base = transform_params(theta_base_uncon, tfs.base)
    par_lagr = transform_params(theta_lagr_uncon, tfs.lagr)
    return par_base, par_lagr


def inverse_transform_model_params(
    par_base: Dict[str, Array],
    par_lagr: Dict[str, Array],
    tfs: ParamTransforms,
) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    theta_base_uncon = inverse_transform_params(par_base, tfs.base)
    theta_lagr_uncon = inverse_transform_params(par_lagr, tfs.lagr)
    return theta_base_uncon, theta_lagr_uncon
