from rpy2 import robjects as ro

r_code = r"""
library(mcgf)
library(jsonlite)
library(reticulate)

N <- 50000
lag_max <- 6
lag <- lag_max - 1
n_loc <- 10

set.seed(12345)
x <- stats::rnorm(lag_max * n_loc, -110)
y <- stats::rnorm(lag_max * n_loc, 50)
locations <- cbind(x, y)

dists <- find_dists(locations[1:n_loc, ], longlat = TRUE)
u <- mcgf:::to_ar(dists$h, lag_max = lag)$u
h <- mcgf:::to_ar(dists$h, lag_max = lag)$h
h_novel <- find_dists(locations, longlat = TRUE)$h

par_base <- list(nugget = 0, c = 0.001, gamma = 0.5, a = 5, alpha = 0.7, beta = 0.7)
par_lagr <- list(c = 0.2, gamma = 0.25)
lambda <- 0.3

cor_base_ar <- do.call(cor_fs, c(par_base, list(h = h, u = u)))
cor_base <- mcgf:::cov_joint(cor_base_ar)

cor_novel <- do.call(cor_exp, c(list(x = h_novel, is.dist = TRUE), par_lagr))
cor_joint <- (1 - lambda) * cor_base + lambda * cor_novel

#------------------------------------------------------------------------------#
# Simulate a MCGF
#------------------------------------------------------------------------------#

# simulate
X_cov_par <- mcgf:::cov_par(cov = cor_joint, horizon = 1, n_var = n_loc, 
                            joint = T)
new_cov_chol <- chol(X_cov_par$cov_curr)

F_mat <- rbind(cbind(X_cov_par$weights, matrix(0, n_loc, n_loc)),
               cbind(diag(n_loc * lag), 
                     matrix(0, nrow = n_loc * lag, ncol = n_loc)))
max(Mod(eigen(F_mat)$values))

Sigma_mat <- rbind(cbind(X_cov_par$cov_curr, 
                         matrix(0, nrow = n_loc, ncol = n_loc * lag)),
                   matrix(0, nrow = n_loc * lag, ncol = n_loc * (lag + 1)))

set.seed(12345)
X <- matrix(0, nrow = lag + 1, ncol = n_loc)
for (n in 1:(N + nrow(X))) {
    X_past <- stats::embed(utils::tail(X, lag), lag)
    X_new_mean <- tcrossprod(X_cov_par$weights, X_past)
    
    X_new <- crossprod(new_cov_chol, stats::rnorm(length(X_new_mean)))
    X_new <- matrix(X_new + X_new_mean, ncol = n_loc, byrow = T)
    X <- rbind(X, X_new)
}
# plot.ts(X)

x_mcgf <- mcgf(X, dists = dists)
x_mcgf <- add_acfs(x_mcgf, lag_max = lag)
x_mcgf <- add_ccfs(x_mcgf, lag_max = lag)
attr(x_mcgf, "horizon") = 1
attr(x_mcgf, "lag") = lag

cor_emp <- ccov(x_mcgf, model = "empirical", cor = T)

np <- import("numpy", convert = FALSE)

out_npz <- "sim_novel_inputs.npz"
np$savez(
    out_npz,
    cor_emp = cor_emp,
    h = h,
    h_novel = h_novel,
    u = u
)

meta <- list(
    par_base = par_base,
    par_lagr = par_lagr,
    lambda = lambda,
    lag_max = lag_max,
    base = "fs",
    lagr = "exp"
)

writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE), "sim_novel_inputs.json") 
"""
gold = ro.r(r_code)

import json
import numpy as np
import time

bundle = np.load("sim_novel_inputs.npz", allow_pickle=False)
cor_emp = bundle["cor_emp"]
h = bundle["h"]
h_lds = bundle["h_novel"]
u = bundle["u"]

with open("sim_novel_inputs.json", "r") as f:
    meta = json.load(f)


lag_max = int(meta["lag_max"])
lag = lag_max - 1
par_base_true = meta["par_base"]
par_lagr_true = meta["par_lagr"]
lam_true = float(meta["lambda"])


# -------------------------------------------------------------------------------
# Convert to JAX + run base-only WLS fit (using our helpers)
# -------------------------------------------------------------------------------
import jax.numpy as jnp
from linear_dynamic_system_backend import jax_fit_all

# Convert to JAX arrays (dtype follows your correlation.DTYPE)
cor_emp_j = jnp.asarray(cor_emp)
h_j = jnp.asarray(h)
u_j = jnp.asarray(u)
h_lds_j = jnp.asarray(h_lds)

# Initial values (pick something reasonable but not too close)
par_init = {
    # base (fs): nugget, c, gamma, a, alpha, beta
    "c": 0.01,
    "gamma": 0.3,
    "a": 1.0,
    "alpha": 0.7,
    "beta": 0.5,
    # mixture + LDS(exp): lam, lds_nugget, lds_c, lds_gamma
    "lam": 0.2,
    "lds_c": 0.05,
    "lds_gamma": 0.3,
}

par_fixed = {"nugget": 0.0, "lds_nugget": 0.0}

# ============================================================
# Grid definitions (edit freely)
# ============================================================
fp_control = {"fp_tol": 1e-1, "fp_maxiter": 100}

fp_list = [
    ("scan_fp", dict(fp_method="scan_fp", fp_control=fp_control)),
    ("jaxopt_fpi", dict(fp_method="jaxopt_fpi", fp_control=fp_control)),
    (
        "jaxopt_anderson",
        dict(fp_method="jaxopt_anderson", fp_control=fp_control),
    ),
]

opt_list = [
    (
        "auto_lbfgs",
        dict(method="auto", control={"maxiter": 5000, "max_stepsize": 0.01}),
    ),
    ("adam", dict(method="adam", maxiter=10000, control={"learning_rate": 0.1})),
    ("adamw", dict(method="adamw", maxiter=10000, control={"learning_rate": 0.1})),
]


for opt_name, opt_kw in opt_list:
    for fp_name, fp_kw in fp_list:
        name = f"{opt_name}__{fp_name}"
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)

        t0 = time.perf_counter()
        res = jax_fit_all(
            base="fs",
            lagrangian="exp",
            lag=lag,
            h=h_j,
            u=u_j,
            h_lds=h_lds_j,
            cor_emp=cor_emp_j,
            par_init=par_init,
            par_fixed=par_fixed,
            **fp_kw,
            **opt_kw,
        )
        dt = time.perf_counter() - t0

        print("sec:", dt)
        print("converged:", res.get("converged"))
        print("n_iter:", res.get("n_iter"))
        print("objective:", res.get("objective"))
        print("lam:", res.get("lam"))
        print("par_base:", res.get("par_base"))
        print("par_lagr:", res.get("par_lagr"))

# # fastest and most accurate
# t0 = time.perf_counter()
# res = jax_fit_all(
#     base="fs",
#     lagrangian="exp",
#     lag=lag,
#     h=h_j,
#     u=u_j,
#     h_lds=h_lds_j,
#     cor_emp=cor_emp_j,
#     par_init=par_init,
#     par_fixed=par_fixed,
#     fp_method="jaxopt_fpi",
#     fp_control={"fp_tol": 1e-1, "fp_maxiter": 100},
#     method="adamw",
#     maxiter=10000,
#     control={"learning_rate": 0.1}
# )
# time.perf_counter() - t0
# res