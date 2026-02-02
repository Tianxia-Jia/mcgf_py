# backend_test_lagr.py
from rpy2 import robjects as ro

# -----------------------------------------------------------------------------
# 1) Run R: simulate data, compute empirical CCFs, fit base in R, export arrays+meta
# -----------------------------------------------------------------------------
r_code = r"""
library(mcgf)
library(reticulate)

data(sim1)
sim1_mcgf <- mcgf(sim1$data, dists = sim1$dists)
sim1_mcgf <- add_acfs(sim1_mcgf, lag_max = 5)
sim1_mcgf <- add_ccfs(sim1_mcgf, lag_max = 5)

# base fit in R (as you already do)
fit_sep <- fit_base(
    sim1_mcgf, model="sep", lag=5,
    par_init = c(c=0.001, gamma=0.5, a=0.3, alpha=0.5),
    par_fixed = c(nugget=0)
)
sim1_mcgf <- add_base(sim1_mcgf, fit_base = fit_sep)

# ---- build lag arrays (n,n,T) ----
lag_max <- 5
dists <- sim1$dists
h  <- dists$h
h1 <- dists$h1
h2 <- dists$h2

tmp <- mcgf:::to_ar(h, lag_max = lag_max)
u3 <- tmp$u        # (n,n,T)
h3 <- tmp$h        # (n,n,T)

# replicate h1/h2 across lag dim
n <- dim(h1)[1]
Tt <- dim(u3)[3]
h13 <- array(rep(h1, Tt), dim = c(n, n, Tt))
h23 <- array(rep(h2, Tt), dim = c(n, n, Tt))

# empirical corr for the lagrangian stage:
# you must set this to the SAME (n,n,T) object you use in R's fit_lagr
# (example placeholder; replace with your actual extraction)
cor_emp_lagr <- ccfs(sim1_mcgf)   # <- ensure this is (n,n,T) numeric array

# ---- base parameters (sep) -> nested structure expected by Python ----
p <- fit_sep$fit$par
par_base <- list(
    par_s = list(nugget = unname(p["nugget"]), c = unname(p["c"]), gamma = unname(p["gamma"])),
    par_t = list(a = unname(p["a"]), alpha = unname(p["alpha"]))
)
library(mcgf)
library(jsonlite)
library(reticulate)

data(sim1)

lag_max <- 5

sim1_mcgf <- mcgf(sim1$data, dists = sim1$dists)
sim1_mcgf <- add_acfs(sim1_mcgf, lag_max = lag_max)
sim1_mcgf <- add_ccfs(sim1_mcgf, lag_max = lag_max)

# empirical CCFs (must be (n,n,T) where T=lag_max+1)
cor_emp <- ccfs(sim1_mcgf)

# distances + lag grid
dists <- sim1$dists
h  <- dists$h
h1 <- dists$h1
h2 <- dists$h2

tmp <- mcgf:::to_ar(h, lag_max = lag_max)
u3 <- tmp$u      # (n,n,T)
h3 <- tmp$h      # (n,n,T)

# Fit base in R (sep), as in your example
fit_sep <- fit_base(
    sim1_mcgf,
    model = "sep",
    lag = lag_max,
    par_init = c(c=0.001, gamma=0.5, a=0.3, alpha=0.5),
    par_fixed = c(nugget = 0)
)
sim1_mcgf <- add_base(sim1_mcgf, fit_base = fit_sep)

# Extract base parameters in a structure Python expects for cor_stat(base="sep")
p <- fit_sep$fit$par
par_base <- list(
    par_s = list(
        nugget = unname(p["nugget"]),
        c      = unname(p["c"]),
        gamma  = unname(p["gamma"])
    ),
    par_t = list(
        a      = unname(p["a"]),
        alpha  = unname(p["alpha"])
    )
)

# Lagrangian init/fixed (match your R call)
par_lagr_init  <- list(v1 = 300, v2 = 300, lam = 0.15, k = 2)
par_lagr_fixed <- list(k = 2)

# Save arrays (note: h1/h2 are 2D; we expand to 3D in Python)
np <- import("numpy", convert = FALSE)

out_npz <- "sim1_lagr_inputs.npz"
np$savez(
    out_npz,
    cor_emp = cor_emp,
    h  = h3,
    u  = u3,
    h1 = h1,
    h2 = h2
)

meta <- list(
    lag_max = lag_max,
    base_model = "sep",
    par_base = par_base,
    lagr_model = "lagr_tri",
    par_lagr_init  = par_lagr_init,
    par_lagr_fixed = par_lagr_fixed
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE), "sim1_lagr_meta.json")
"""
ro.r(r_code)

# -----------------------------------------------------------------------------
# 2) Python: load bundle, expand (h1,h2)->3D, compute base_corr, then fit lagr
# -----------------------------------------------------------------------------
import json
import numpy as np

bundle = np.load("sim1_lagr_inputs.npz", allow_pickle=False)
cor_emp = bundle["cor_emp"]  # should be (n,n,T)
h = bundle["h"]  # (n,n,T)
u = bundle["u"]  # (n,n,T)
h1_2d = bundle["h1"]  # (n,n)
h2_2d = bundle["h2"]  # (n,n)

with open("sim1_lagr_meta.json", "r") as f:
    meta = json.load(f)

lag_max = int(meta["lag_max"])
par_base = meta["par_base"]
lagr_model = meta["lagr_model"]
par_init = meta["par_lagr_init"]
par_fixed = meta["par_lagr_fixed"]

# --- expand h1/h2 to (n,n,T) to match u/cor_emp ---
if cor_emp.ndim != 3:
    raise ValueError(f"Expected cor_emp to be 3D (n,n,T). Got shape={cor_emp.shape}")

n, n2, T = cor_emp.shape
assert n == n2, cor_emp.shape

# broadcast (n,n) -> (n,n,T)
h1 = np.repeat(h1_2d[..., None], T, axis=2)
h2 = np.repeat(h2_2d[..., None], T, axis=2)

# -----------------------------------------------------------------------------
# 3) Convert to JAX + run lagr-only fit with fixed base_corr
# -----------------------------------------------------------------------------
import jax.numpy as jnp
from correlation import DTYPE
from estimate_backend import _check_lagr_inputs, compute_base_corr, jax_fit_lagr_one

# validate shapes for lagr stage
_check_lagr_inputs(h1, h2, u, cor_emp)

# jax arrays
h_j = jnp.asarray(h, dtype=DTYPE)
u_j = jnp.asarray(u, dtype=DTYPE)
h1_j = jnp.asarray(h1, dtype=DTYPE)
h2_j = jnp.asarray(h2, dtype=DTYPE)
cor_emp_j = jnp.asarray(cor_emp, dtype=DTYPE)

# compute dense base correlation on the SAME (n,n,T) grid
par_base['par_s']['nugget'] = 0.0

base_out = compute_base_corr(
    base=meta["base_model"],  # "sep"
    par_base=par_base,
    h=h_j,
    u=u_j,
)
base_corr_j = base_out["base_corr"]

# IMPORTANT: include k in par_init even if fixed (your fitter expects it present)
par_init_full = {
    "lam": float(par_init["lam"]),
    "v1": float(par_init["v1"]),
    "v2": float(par_init["v2"]),
    "k": float(par_init.get("k", par_fixed.get("k", 2))),
}
par_fixed_full = dict(par_fixed or {})

# LBFGS
control = {"maxiter": 5000}
par_init_full = {
    "lam": 0.1,
    "v1": 100,
    "v2": 100,
    "k": 2,
}

res = jax_fit_lagr_one(
    lagrangian=lagr_model,  # "lagr_tri"
    base_corr=base_corr_j,  # fixed base
    h1=h1_j,
    h2=h2_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_full,
    par_fixed=par_fixed_full,
    control=control,
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("lam:", res["lam"])
print("par_lagr:", res["par_lagr"])

# ADAM
control = {"maxiter": 5000}
res = jax_fit_lagr_one(
    lagrangian=lagr_model,  # "lagr_tri"
    base_corr=base_corr_j,  # fixed base
    h1=h1_j,
    h2=h2_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_full,
    par_fixed=par_fixed_full,
    method="adam",
    control=control,
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("lam:", res["lam"])
print("par_lagr:", res["par_lagr"])


# ADAMW
control = {"maxiter": 5000}
res = jax_fit_lagr_one(
    lagrangian=lagr_model,  # "lagr_tri"
    base_corr=base_corr_j,  # fixed base
    h1=h1_j,
    h2=h2_j,
    u=u_j,
    cor_emp=cor_emp_j,
    par_init=par_init_full,
    par_fixed=par_fixed_full,
    method="adamw",
    control=control,
)

print("converged:", res["converged"])
print("n_iter:", res["n_iter"])
print("objective:", res["objective"])
print("lam:", res["lam"])
print("par_lagr:", res["par_lagr"])

