from rpy2 import robjects as ro

# =============================================================================
# Regime-switching simulation for LDS backend test
#
# Two regimes r=1,2:
#   - distinct (par_base_r, par_lagr_r, lambda_r)
#   - distinct novel distance matrix h_novel_r (so cor_novel_r differs)
#
# Simulation:
#   - precompute regime-specific cov_par outputs: weights_r and chol(cov_curr_r)
#   - generate a regime label sequence with long contiguous blocks
#   - in the loop, use regime-dependent weights and chol to simulate X_t
#
# Empirical correlations per regime:
#   - split the full series into contiguous runs of each regime
#   - for each run with length > lag+1, compute ccov empirical correlation
#   - average correlations across runs weighted by run length
#
# Python estimation:
#   - estimate each regime separately using jax_fit_all with that regime's cor_emp
# =============================================================================


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
h_novel_1 <- find_dists(locations, longlat = TRUE)$h

set.seed(123)
x <- stats::rnorm(lag * n_loc, -110)
y <- stats::rnorm(lag * n_loc, 50)
locations <- rbind(locations[1:n_loc, ], cbind(x, y))
h_novel_2 <- find_dists(locations, longlat = TRUE)$h

# -----------------------------
# True parameters by regime
# -----------------------------

par_base_1 <- list(nugget = 0, c = 0.0010, gamma = 0.50, a = 5.0, alpha = 0.85, beta = 0.70)
par_lagr_1 <- list(c = 0.20, gamma = 0.25)
lambda_1   <- 0.30

par_base_2 <- list(nugget = 0, c = 0.002, gamma = 0.35, a = 4.5, alpha = 0.6, beta = 0.50)
par_lagr_2 <- list(c = 0.35, gamma = 0.15)
lambda_2   <- 0.1

# -----------------------------
# Build regime-specific joint correlations and cov_par
# -----------------------------
cor_base_ar_1 <- do.call(cor_fs, c(par_base_1, list(h = h, u = u)))
cor_base_1 <- mcgf:::cov_joint(cor_base_ar_1)
cor_novel_1 <- do.call(cor_exp, c(list(x = h_novel_1, is.dist = TRUE), par_lagr_1))
cor_joint_1 <- (1 - lambda_1) * cor_base_1 + lambda_1 * cor_novel_1

cor_base_ar_2 <- do.call(cor_fs, c(par_base_2, list(h = h, u = u)))
cor_base_2 <- mcgf:::cov_joint(cor_base_ar_2)
cor_novel_2 <- do.call(cor_exp, c(list(x = h_novel_2, is.dist = TRUE), par_lagr_2))
cor_joint_2 <- (1 - lambda_2) * cor_base_2 + lambda_2 * cor_novel_2

X_cov_par_1 <- mcgf:::cov_par(cov = cor_joint_1, horizon = 1, n_var = n_loc, joint = TRUE)
X_cov_par_2 <- mcgf:::cov_par(cov = cor_joint_2, horizon = 1, n_var = n_loc, joint = TRUE)

chol_1 <- chol(X_cov_par_1$cov_curr)
chol_2 <- chol(X_cov_par_2$cov_curr)

#------------------------------------------------------------------------------#
# Simulate a MCGF
#------------------------------------------------------------------------------#

# regime
tran_mat = matrix(c(0.95, 0.05, 0.03, 0.97), nrow = 2, byrow = T)
regime <- integer(N + lag + 1)

set.seed(123)
regime[1] = 1L
for (i in 2:length(regime)) {
    regime[i] <- sample(c(1L, 2L), 1, prob = tran_mat[regime[i - 1], ])
}

set.seed(12345)
X <- matrix(0, nrow = lag + 1, ncol = n_loc)

for (n in 1:N) {
    X_past <- stats::embed(utils::tail(X, lag), lag)
    
    if (regime[n] == 1L) {
        X_new_mean <- tcrossprod(X_cov_par_1$weights, X_past)
        X_new <- crossprod(chol_1, stats::rnorm(length(X_new_mean)))
    } else if (regime[n] == 2L) {
        X_new_mean <- tcrossprod(X_cov_par_2$weights, X_past)
        X_new <- crossprod(chol_2, stats::rnorm(length(X_new_mean)))
    } else {
        stop(paste("invalid regime index", regime[n]))
    }

    X_new <- matrix(X_new + X_new_mean, ncol = n_loc, byrow = T)
    X <- rbind(X, X_new)
}

# Drop initial zeros and align regime labels
X_sim <- X[-(1:(lag + 1)), , drop = FALSE]
stopifnot(nrow(X_sim) == N)

x_mcgf_rs <- mcgf_rs(X, dists = dists, label = regime)
x_mcgf_rs <- add_acfs(x_mcgf_rs, lag_max = lag)
x_mcgf_rs <- add_ccfs(x_mcgf_rs, lag_max = lag, ncores = 16)
attr(x_mcgf_rs, "horizon") = 1
attr(x_mcgf_rs, "lag") = c(lag, lag)

cor_emp <- ccov(x_mcgf_rs, model = "empirical", cor = T)

# -----------------------------
# Save for Python
# -----------------------------
np <- import("numpy", convert = FALSE)
out_npz <- "sim_novel_rs_inputs.npz"
np$savez(
    out_npz,
    cor_emp_1 = cor_emp$`Regime 1`,
    cor_emp_2 = cor_emp$`Regime 2`,
    h = h,
    u = u,
    h_novel_1 = h_novel_1,
    h_novel_2 = h_novel_2
)

meta <- list(
    N = N,
    lag_max = lag_max,
    lag = lag,
    n_loc = n_loc,
    regimes = list(
        r1 = list(par_base = par_base_1, par_lagr = par_lagr_1, lambda = lambda_1),
        r2 = list(par_base = par_base_2, par_lagr = par_lagr_2, lambda = lambda_2)
    ),
    base = "fs",
    lagr = "exp",
    regime = regime
)

writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE), "sim_novel_rs_meta.json")
"""
gold = ro.r(r_code)

import json
import numpy as np
import time

bundle = np.load("sim_novel_rs_inputs.npz", allow_pickle=False)
cor_emp_1 = bundle["cor_emp_1"]
cor_emp_2 = bundle["cor_emp_2"]
h = bundle["h"]
u = bundle["u"]
h_lds_1 = bundle["h_novel_1"]
h_lds_2 = bundle["h_novel_2"]

with open("sim_novel_rs_meta.json", "r") as f:
    meta = json.load(f)

lag_max = int(meta["lag_max"])
lag = lag_max - 1

par_true = meta["regimes"]
par_base_true_1 = meta["regimes"]["r1"]["par_base"]
par_lagr_true_1 = meta["regimes"]["r1"]["par_lagr"]
lam_true_1 = meta["regimes"]["r1"]["lambda"]

par_base_true_2 = meta["regimes"]["r2"]["par_base"]
par_lagr_true_2 = meta["regimes"]["r2"]["par_lagr"]
lam_true_2 = meta["regimes"]["r2"]["lambda"]

# -------------------------------------------------------------------------------
# Convert to JAX + run base-only WLS fit (using our helpers)
# -------------------------------------------------------------------------------
import jax.numpy as jnp
from linear_dynamic_system_backend import (
    jax_fit_all,
    jax_fit_base_one,
    jax_fit_lagr_one,
    compute_base_corr,
)


# Convert to JAX arrays (dtype follows your correlation.DTYPE)
h_j = jnp.asarray(h)
u_j = jnp.asarray(u)
cor_emp_j_list = [jnp.asarray(cor_emp_1), jnp.asarray(cor_emp_2)]
h_lds_j_list = [jnp.asarray(h_lds_1), jnp.asarray(h_lds_2)]

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
# Grid definitions
# ============================================================
def print_res(title, res, dt):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)
    print("sec:", dt)
    print("converged:", res.get("converged"))
    print("n_iter:", res.get("n_iter"))
    print("objective:", res.get("objective"))
    if "lam" in res:
        print("lam:", res.get("lam"))
    if "par_base" in res:
        print("par_base:", res.get("par_base"))
    if "par_lagr" in res:
        print("par_lagr:", res.get("par_lagr"))


fp_controls = [
    {"tol": 1e-1, "maxiter": 100},
    {"tol": 1e-2, "maxiter": 500},
    {"tol": 1e-3, "maxiter": 500},
]
fp_methods = ["jaxopt_fpi", "jaxopt_anderson"]

fp_list = []
for fp_control in fp_controls:
    for method in fp_methods:
        name = f"{method}_tol{fp_control['tol']}_iter{fp_control['maxiter']}"
        fp_list.append(
            (
                name,
                dict(fp_method=method, fp_control=fp_control),
            )
        )

opt_list = [
    # (
    #     "auto_lbfgs",
    #     dict(method="auto", control={"maxiter": 5000, "max_stepsize": 0.01}),
    # ),
    ("adam", dict(method="adam", maxiter=50000, control={"learning_rate": 0.1})),
    ("adamw", dict(method="adamw", maxiter=50000, control={"learning_rate": 0.1})),
]


for r in (1, 2):
    print("\n" + "#" * 80)
    print(f"Regime {r} fit (separate)")
    print("#" * 80)
    print("true:", par_true[f"r{r}"])

    cor_emp_r = cor_emp_j_list[r - 1]
    h_lds_r = h_lds_j_list[r - 1]

    for opt_name, opt_kw in opt_list:
        for fp_name, fp_kw in fp_list:
            name = f"reg{r}__{opt_name}__{fp_name}"
            print("\n" + "=" * 80)
            print(name)
            print("=" * 80)

            # ------------------------------------------------------------
            # 1) base-only fit
            # ------------------------------------------------------------
            t0 = time.perf_counter()
            res_base = jax_fit_base_one(
                base="fs",
                lag=lag,
                h=h_j,
                u=u_j,
                cor_emp=cor_emp_r,
                par_init=par_init,
                par_fixed={"nugget": 0.0},
                **fp_kw,
                **opt_kw,
            )
            dt = time.perf_counter() - t0
            print_res("base-only", res_base, dt)

            # Build fixed base corr from fitted base parameters
            base_corr_fit = compute_base_corr(
                base="fs",
                par_base=res_base["par_base"],
                h=h_j,
                u=u_j,
            )["base_corr"]

            # ------------------------------------------------------------
            # 2) lagr-only fit, conditional on fitted base
            # ------------------------------------------------------------
            t0 = time.perf_counter()
            res_lagr = jax_fit_lagr_one(
                lagrangian="exp",
                lag=lag,
                base_corr=base_corr_fit,
                h_lds=h_lds_r,
                cor_emp=cor_emp_r,
                par_init=par_init,
                par_fixed={"lds_nugget": 0.0},
                **fp_kw,
                **opt_kw,
            )
            dt = time.perf_counter() - t0
            print_res("lagr-only | fitted base", res_lagr, dt)

            # ------------------------------------------------------------
            # 3) joint fit
            # ------------------------------------------------------------
            t0 = time.perf_counter()
            res_joint = jax_fit_all(
                base="fs",
                lagrangian="exp",
                lag=lag,
                h=h_j,
                u=u_j,
                h_lds=h_lds_r,
                cor_emp=cor_emp_r,
                par_init=par_init,
                par_fixed=par_fixed,
                **fp_kw,
                **opt_kw,
            )
            dt = time.perf_counter() - t0
            print_res("joint fit", res_joint, dt)