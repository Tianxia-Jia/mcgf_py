
from rpy2 import robjects as ro

r_code = r"""
library(mcgf)
library(jsonlite)
library(reticulate)

data(sim1)

sim1_mcgf <- mcgf(sim1$data, dists = sim1$dists)
sim1_mcgf <- add_acfs(sim1_mcgf, lag_max = 5)
sim1_mcgf <- add_ccfs(sim1_mcgf, lag_max = 5)

cor_emp <- ccfs(sim1_mcgf)

dists <- sim1$dists
h  <- dists$h
h1 <- dists$h1
h2 <- dists$h2
u  <- mcgf:::to_ar(h, lag_max = 5)$u
h  <- mcgf:::to_ar(h, lag_max = 5)$h
print(u)

par_init <- c(c = 0.001, gamma = 0.5, a = 0.3, alpha = 0.5)
par_fixed <- c(nugget = 0)

np <- import("numpy", convert = FALSE)

out_npz <- "sim1_sep_inputs.npz"
np$savez(
    out_npz,
    cor_emp = cor_emp,
    h = h,
    h1 = h1,
    h2 = h2,
    u = u
)

meta <- list(
    lag_max = 5,
    model = "sep",
    par_init = as.list(par_init),
    par_fixed = as.list(par_fixed)
)

writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE), "sim1_sep_meta.json") 
"""
gold = ro.r(r_code)  # this is an R "list"

import json
import numpy as np

bundle = np.load("sim1_sep_inputs.npz", allow_pickle=False)
cor_emp = bundle["cor_emp"]
h = bundle["h"]
h1 = bundle["h1"]
h2 = bundle["h2"]
u = bundle["u"]

with open("sim1_sep_meta.json", "r") as f:
    meta = json.load(f)

par_init = meta["par_init"]  # dict
par_fixed = meta["par_fixed"]  # dict
lag_max = meta["lag_max"]

# -------------------------------------------------------------------------------
# Convert to JAX + run base-only WLS fit (using our helpers)
# -------------------------------------------------------------------------------
from estimate_backend import _check_base_inputs, jax_fit_base_one
from correlation import DTYPE
import jax.numpy as jnp

# validate shapes (h, u, cor_emp must match)
_check_base_inputs(h, u, cor_emp)

# Ensure JAX arrays (DTYPE) for consistency
h_j = jnp.asarray(h, dtype=DTYPE)
u_j = jnp.asarray(u, dtype=DTYPE)
cor_emp_j = jnp.asarray(cor_emp, dtype=DTYPE)

# Our base fitter expects a complete param dict for the base family.
# Even if "sep" doesn't use beta, we still include it to keep a unified interface.
par_init_full = {
    "nugget": float(par_init.get("nugget", par_fixed.get("nugget", 0.0))),
    "c": float(par_init["c"]),
    "gamma": float(par_init["gamma"]),
    "a": float(par_init["a"]),
    "alpha": float(par_init["alpha"]),
    "beta": float(par_init.get("beta", 0.0)),
}

# Fixed parameters (optional): e.g. nugget fixed by your R call
par_fixed_full = dict(par_fixed or {})
if "beta" not in par_fixed_full:
    # keep beta fixed at 0 for sep unless you explicitly want to estimate it
    par_fixed_full["beta"] = 0.0

control = {"maxiter": 5000}  # mirrors your old LBFGS maxiter

# LBFGS
res = jax_fit_base_one(
    base="fs",
    h=h_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_full,
    par_fixed=par_fixed_full,  # can be {} or None
    control=control,
    method = "auto",
)

# "res" is already python-friendly (floats/lists) for reticulate.
print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("par_base:", res["par_base"])

# If you still want a vector-like "x_hat" comparable to your old PAR_ORDER:
x_hat = [
    res["par_base"]["c"],
    res["par_base"]["gamma"],
    res["par_base"]["a"],
    res["par_base"]["alpha"],
]
c, gamma, a, alpha = x_hat
print(f"x_hat (c, gamma, a, alpha): {c:.4f}, {gamma:.4f}, {a:.4f}, {alpha:.4f}")

# ADAM
res = jax_fit_base_one(
    base="fs",
    h=h_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_full,
    par_fixed=par_fixed_full,  # can be {} or None
    control=control,
    method = "adam",
)
print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("par_base:", res["par_base"])

x_hat = [
    res["par_base"]["c"],
    res["par_base"]["gamma"],
    res["par_base"]["a"],
    res["par_base"]["alpha"],
]
c, gamma, a, alpha = x_hat
print(f"x_hat (c, gamma, a, alpha): {c:.4f}, {gamma:.4f}, {a:.4f}, {alpha:.4f}")