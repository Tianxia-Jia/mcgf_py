from rpy2 import robjects as ro

r_code = r"""
library(mcgf)
library(jsonlite)
library(reticulate)

set.seed(12345)
x <- stats::rnorm(10, -110)
y <- stats::rnorm(10, 50)
locations <- cbind(x, y)
h <- find_dists(locations, longlat = TRUE)

N <- 50000
lag <- 5

par_base <- list(nugget = 0, c = 0.002, gamma = 0.5, a = 5, alpha = 0.8, beta = 0.75)
par_lagr <- list(v1 = 50, v2 = 200, k = 1.5)

sim1 <- mcgf_sim(
    N = N, base = "fs", lagrangian = "lagr_tri",
    par_base = par_base, par_lagr = par_lagr, lambda = 0.25,
    dists = h, lag = lag
)
sim1 <- sim1[-c(1:(lag + 1)), ]
rownames(sim1) <- 1:nrow(sim1)

sim1 <- list(data = sim1, locations = locations, dists = h)

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

par_init <- c(c = 0.001, gamma = 0.5, a = 0.3, alpha = 0.5, beta = 0.5)
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
    par_base = par_base,
    par_lagr = par_lagr,
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

par_base = meta["par_base"]
par_lagr = meta["par_lagr"]
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
par_init_base = {
    "nugget": float(par_init.get("nugget", par_fixed.get("nugget", 0.0))),
    "c": 0.001,
    "gamma": 0.5,
    "a": 0.5,
    "alpha": 0.8,
    "beta": 0.5,
}

# Fixed parameters (optional): e.g. nugget fixed by your R call
par_fixed_full = dict(par_fixed or {})
control = {"maxiter": 5000}  # mirrors your old LBFGS maxiter

# LBFGS
res = jax_fit_base_one(
    base="fs",
    h=h_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_base,
    par_fixed=par_fixed_full,  # can be {} or None
    control={"maxiter": 5000, "max_stepsize": 0.01},
    method="auto",
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("par_base:", res["par_base"])

# ADAM
res = jax_fit_base_one(
    base="fs",
    h=h_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_base,
    par_fixed=par_fixed_full,  # can be {} or None
    control={"learning_rate": 0.0001},
    maxiter=100000,
    method="adam",
)
print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("par_base:", res["par_base"])

# ADAMW
res = jax_fit_base_one(
    base="fs",
    h=h_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_base,
    par_fixed=par_fixed_full,  # can be {} or None
    control={"learning_rate": 0.0001},
    maxiter=100000,
    method="adamw",
)
print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("par_base:", res["par_base"])

# -------------------------------------------------------------------------------
# Convert to JAX + run lagr-only WLS fit (using our helpers)
# -------------------------------------------------------------------------------
n, n2, T = cor_emp.shape
assert n == n2, cor_emp.shape

# broadcast (n,n) -> (n,n,T)
h1_3d = np.repeat(h1[..., None], T, axis=2)
h2_3d = np.repeat(h2[..., None], T, axis=2)

# validate shapes for lagr stage
from estimate_backend import _check_lagr_inputs, compute_base_corr, jax_fit_lagr_one

_check_lagr_inputs(h1_3d, h2_3d, u, cor_emp)

# jax arrays
h_j = jnp.asarray(h, dtype=DTYPE)
u_j = jnp.asarray(u, dtype=DTYPE)
h1_j = jnp.asarray(h1_3d, dtype=DTYPE)
h2_j = jnp.asarray(h2_3d, dtype=DTYPE)
cor_emp_j = jnp.asarray(cor_emp, dtype=DTYPE)

base_out = compute_base_corr(
    base="fs",
    par_base=par_base,
    h=h_j,
    u=u_j,
)
base_corr_j = base_out["base_corr"]

par_init_lagr = {
    "lam": 0.1,
    "v1": 100,
    "v2": 100,
    "k": 2,
}
par_fixed_full = {}

# LBFGS
res = jax_fit_lagr_one(
    lagrangian="lagr_tri",
    base_corr=base_corr_j,
    h1=h1_j,
    h2=h2_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_lagr,
    par_fixed=par_fixed_full,
    control={"maxiter": 5000, "max_stepsize": 0.01},
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("lam:", res["lam"])
print("par_lagr:", res["par_lagr"])

# ADAM
res = jax_fit_lagr_one(
    lagrangian="lagr_tri",  # "lagr_tri"
    base_corr=base_corr_j,  # fixed base
    h1=h1_j,
    h2=h2_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_lagr,
    par_fixed=par_fixed_full,
    control={"learning_rate": 0.0001},
    maxiter=100000,
    method="adam",
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("lam:", res["lam"])
print("par_lagr:", res["par_lagr"])

# ADAMW
res = jax_fit_lagr_one(
    lagrangian="lagr_tri",  # "lagr_tri"
    base_corr=base_corr_j,  # fixed base
    h1=h1_j,
    h2=h2_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_lagr,
    par_fixed=par_fixed_full,
    control={"learning_rate": 0.0001},
    maxiter=100000,
    method="adamw",
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("lam:", res["lam"])
print("par_lagr:", res["par_lagr"])


# -------------------------------------------------------------------------------
# Convert to JAX + run all WLS fit (using our helpers)
# -------------------------------------------------------------------------------
from estimate_backend import jax_fit_all_one

# LBFGS
res = jax_fit_all_one(
    base="fs",
    lagrangian="lagr_tri",
    h1=h1_j,
    h2=h2_j,
    u=u_j,
    h=h,
    cor_emp=cor_emp_j,
    par_init=par_init_base | par_init_lagr,
    par_fixed=par_fixed,
    control={"maxiter": 5000, "max_stepsize": 0.01},
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("lam:", res["lam"])
print("par_base:", res["par_base"])
print("par_lagr:", res["par_lagr"])

# ADAM
res = jax_fit_all_one(
    base="fs",
    lagrangian="lagr_tri",
    h1=h1_j,
    h2=h2_j,
    u=u_j,
    h=h,
    cor_emp=cor_emp_j,
    par_init=par_init_base | par_init_lagr,
    par_fixed=par_fixed,
    control={"learning_rate": 0.02},
    maxiter=500000,
    method="adam",
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("lam:", res["lam"])
print("par_base:", res["par_base"])
print("par_lagr:", res["par_lagr"])


# ADAMW
res = jax_fit_all_one(
    base="fs",
    lagrangian="lagr_tri",
    h1=h1_j,
    h2=h2_j,
    u=u_j,
    h=h,
    cor_emp=cor_emp_j,
    par_init=par_init_base | par_init_lagr,
    par_fixed=par_fixed,
    control={"learning_rate": 0.002},
    maxiter=100000,
    method="adamw",
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("lam:", res["lam"])
print("par_base:", res["par_base"])
print("par_lagr:", res["par_lagr"])
