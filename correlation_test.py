import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # import AFTER enabling


from rpy2 import robjects as ro
from rpy2.robjects import conversion, default_converter
from rpy2.robjects import numpy2ri

import correlation as cor

DTYPE = jnp.float64  # match R double


def r_to_numpy(x):
    with conversion.localconverter(default_converter + numpy2ri.converter):
        return np.array(x)


def report_header(title):
    print()
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(f"{'Case':<26} | {'OK':^3} | {'max_abs':>12} | {'max_rel':>12}")
    print("-" * 80)


def report(name, got, ref, rtol=1e-6, atol=1e-6):
    got = np.asarray(got)
    ref = np.asarray(ref)
    diff = np.abs(got - ref)
    denom = np.maximum(1e-12, np.abs(ref))
    ok = np.allclose(got, ref, rtol=rtol, atol=atol)
    sym = "✓" if ok else "✗"
    print(f"{name:<26} | {sym:^3} | {diff.max():12.3e} | {(diff/denom).max():12.3e}")


def report_slices(name, got, ref, rtol=1e-6, atol=1e-6):
    got = np.asarray(got)
    ref = np.asarray(ref)
    K = got.shape[-1]
    for k in range(K):
        report(f"{name}[..,{k}]", got[..., k], ref[..., k], rtol=rtol, atol=atol)


# -----------------------------
# 1) Run your R code in-memory
# -----------------------------
r_code = r"""
library(mcgf)

set.seed(20260128)

d  <- rdists(20, scale = 50)
h1 <- d$h1
h2 <- d$h2
h1[1, 2] <- h1[2, 1] <- 0.0
h2[1, 2] <- h2[2, 1] <- 0.0

h1[1, 3] <- h1[3, 1] <- 0.0
h2[1, 3] <- h2[3, 1] <- 0.0

h1[1, 4] <- h1[4, 1] <- 0.0
h2[1, 4] <- h2[4, 1] <- 0.0

h  <- sqrt(h1^2 + h2^2)

tt <- seq(0, 1.9, length.out = 20)
u  <- abs(outer(tt, tt, "-"))^1.3

par_s <- list(nugget = 0.05, c = 0.004, gamma = 0.25)
par_t <- list(a = 0.6, alpha = 0.35)
par_base <- list(par_s = par_s, par_t = par_t)
par_base_fs <- c(par_s, par_t, beta = 0.78)

par_lagr_1 <- list(v1 = 5, v2 = 10, k = 2)
par_lagr_2 <- list(v1 = 0.01, v2 = 100, k = 8)

Cb_fs <- cor_fs(
    nugget = par_s$nugget,
    c = par_s$c,
    gamma = par_s$gamma,
    a = par_t$a,
    alpha = par_t$alpha,
    beta = 0.78,
    h = h,
    u = u
)

C_mix_037 <- cor_stat(
    base = "sep",
    lagrangian = "lagr_tri",
    par_base = par_base,
    par_lagr = par_lagr_1,
    h = h,
    h1 = h1,
    h2 = h2,
    u = u,
    lambda = 0.37,
    base_fixed = FALSE
)

C_mix_lam0 <- cor_stat(
    base = "fs",
    lagrangian = "lagr_tri",
    par_base = par_base_fs,
    par_lagr = par_lagr_1,
    h = h,
    h1 = h1,
    h2 = h2,
    u = u,
    lambda = 0.0,
    base_fixed = FALSE
)

C_mix_lam1 <- cor_stat(
    base = "fs",
    lagrangian = "lagr_tri",
    par_base = par_base_fs,
    par_lagr = par_lagr_1,
    h = h,
    h1 = h1,
    h2 = h2,
    u = u,
    lambda = 1.0,
    base_fixed = FALSE
)

C_mix_fixed <- cor_stat(
    base = Cb_fs,
    lagrangian = "lagr_tri",
    par_lagr = par_lagr_1,
    h1 = h1,
    h2 = h2,
    u = u,
    lambda = 0.37,
    base_fixed = TRUE
)

C_mix_extreme <- cor_stat(
    base = "fs",
    lagrangian = "lagr_tri",
    par_base = par_base_fs,
    par_lagr = par_lagr_2,
    h = h,
    h1 = h1,
    h2 = h2,
    u = u,
    lambda = 0.37,
    base_fixed = FALSE
)

list(
    h1 = h1,
    h2 = h2,
    h = h,
    u = u,
    par_s = par_s,
    par_t = par_t,
    par_lagr_1 = par_lagr_1,
    par_lagr_2 = par_lagr_2,
    Cb_fs = Cb_fs,
    C_mix_037 = C_mix_037,
    C_mix_lam0 = C_mix_lam0,
    C_mix_lam1 = C_mix_lam1,
    C_mix_fixed = C_mix_fixed,
    C_mix_extreme = C_mix_extreme
)
"""
gold = ro.r(r_code)  # this is an R "list"


# Helper to pull a named element out of an R list
def r_get(name):
    return gold.rx2(name)


# matrices
h1 = jnp.asarray(np.array(r_get("h1")), dtype=DTYPE)
h2 = jnp.asarray(np.array(r_get("h2")), dtype=DTYPE)
h = jnp.asarray(np.array(r_get("h")), dtype=DTYPE)
u = jnp.asarray(np.array(r_get("u")), dtype=DTYPE)

# references
Cb_fs_ref = jnp.asarray(np.array(r_get("Cb_fs")), dtype=DTYPE)
C_mix_037_ref = jnp.asarray(np.array(r_get("C_mix_037")), dtype=DTYPE)
C_mix_lam0_ref = jnp.asarray(np.array(r_get("C_mix_lam0")), dtype=DTYPE)
C_mix_lam1_ref = jnp.asarray(np.array(r_get("C_mix_lam1")), dtype=DTYPE)
C_mix_fixed_ref = jnp.asarray(np.array(r_get("C_mix_fixed")), dtype=DTYPE)
C_mix_extreme_ref = jnp.asarray(np.array(r_get("C_mix_extreme")), dtype=DTYPE)

# params -> python dicts
par_s = r_get("par_s")
par_t = r_get("par_t")
par_lagr_1_r = r_get("par_lagr_1")
par_lagr_2_r = r_get("par_lagr_2")

par_base = {
    "par_s": {
        "nugget": jnp.asarray(par_s.rx2("nugget")[0]),
        "c": jnp.asarray(par_s.rx2("c")[0]),
        "gamma": jnp.asarray(par_s.rx2("gamma")[0]),
    },
    "par_t": {
        "a": jnp.asarray(par_t.rx2("a")[0]),
        "alpha": jnp.asarray(par_t.rx2("alpha")[0]),
    },
}

par_base_fs = {
    "nugget": jnp.asarray(par_s.rx2("nugget")[0]),
    "c": jnp.asarray(par_s.rx2("c")[0]),
    "gamma": jnp.asarray(par_s.rx2("gamma")[0]),
    "a": jnp.asarray(par_t.rx2("a")[0]),
    "alpha": jnp.asarray(par_t.rx2("alpha")[0]),
    "beta": 0.78,
}

par_lagr_1 = {
    "v1": jnp.asarray(par_lagr_1_r.rx2("v1")[0]),
    "v2": jnp.asarray(par_lagr_1_r.rx2("v2")[0]),
    "k": jnp.asarray(par_lagr_1_r.rx2("k")[0]),
}
par_lagr_2 = {
    "v1": jnp.asarray(par_lagr_2_r.rx2("v1")[0]),
    "v2": jnp.asarray(par_lagr_2_r.rx2("v2")[0]),
    "k": jnp.asarray(par_lagr_2_r.rx2("k")[0]),
}

# -----------------------------
# 2) Compute JAX versions
# -----------------------------
# NOTE: assumes your cor_fs + cor_stat are defined in Python already.

Cb_fs = cor.cor_fs(
    nugget=par_base_fs["nugget"],
    c=par_base_fs["c"],
    gamma=par_base_fs["gamma"],
    a=par_base_fs["a"],
    alpha=par_base_fs["alpha"],
    beta=par_base_fs["beta"],
    h=h,
    u=u,
)

C_mix_037 = cor.cor_stat(
    base="sep",
    lagrangian="lagr_tri",
    par_base=par_base,
    par_lagr=par_lagr_1,
    h=h,
    h1=h1,
    h2=h2,
    u=u,
    lam=0.37,
    base_fixed=False,
)

C_mix_lam0 = cor.cor_stat(
    base="fs",
    lagrangian="lagr_tri",
    par_base=par_base_fs,
    par_lagr=par_lagr_1,
    h=h,
    h1=h1,
    h2=h2,
    u=u,
    lam=0.0,
    base_fixed=False,
)

C_mix_lam1 = cor.cor_stat(
    base="fs",
    lagrangian="lagr_tri",
    par_base=par_base_fs,
    par_lagr=par_lagr_1,
    h=h,
    h1=h1,
    h2=h2,
    u=u,
    lam=1.0,
    base_fixed=False,
)

C_mix_fixed = cor.cor_stat(
    base=Cb_fs_ref,
    lagrangian="lagr_tri",
    par_lagr=par_lagr_1,
    h1=h1,
    h2=h2,
    u=u,
    lam=0.37,
    base_fixed=True,
)

C_mix_extreme = cor.cor_stat(
    base="fs",
    lagrangian="lagr_tri",
    par_base=par_base_fs,
    par_lagr=par_lagr_2,
    h=h,
    h1=h1,
    h2=h2,
    u=u,
    lam=0.37,
    base_fixed=False,
)


# -----------------------------
# 3) Compare
# -----------------------------
report("Cb_fs", Cb_fs, Cb_fs_ref)
report("C_mix_037", C_mix_037, C_mix_037_ref)
report("C_mix_lam0", C_mix_lam0, C_mix_lam0_ref)
report("C_mix_lam1", C_mix_lam1, C_mix_lam1_ref)
report("C_mix_fixed", C_mix_fixed, C_mix_fixed_ref)
report("C_mix_extreme", C_mix_extreme, C_mix_extreme_ref)


# ----------------------------
# 1) R reference (3D objects)
# ----------------------------
# - rdists(n=20, scale=50) per slice
# - force pairs (1,2), (1,3), (1,4) to 0 in h1 and h2 in every slice
# - recompute h
# - u varies by slice (hard non-linear)
# - compute Cb_fs and C_mix in R
r_code = r"""
library(mcgf)

set.seed(20260128)

n <- 20
K <- 4
scale <- 50

h1 <- array(0, dim=c(n, n, K))
h2 <- array(0, dim=c(n, n, K))

for (k in 1:K) {
    d <- rdists(n, scale=scale)
    hk1 <- d$h1
    hk2 <- d$h2
    
    # force exact zeros in selected off-diagonal pairs (1-based indices)
    idx <- list(c(1,2), c(1,3), c(1,4))
    for (p in idx) {
        i <- p[1]; j <- p[2]
        hk1[i, j] <- hk1[j, i] <- 0.0
        hk2[i, j] <- hk2[j, i] <- 0.0
    }
    
    h1[,,k] <- hk1
    h2[,,k] <- hk2
}

h <- sqrt(h1^2 + h2^2)

# hard u: varies across (i,j) AND slice k; includes zeros
tt <- seq(0, 1.9, length.out = n)
u2 <- abs(outer(tt, tt, "-"))^1.3
u <- array(0, dim=c(n, n, K))
for (k in 1:K) {
    u[,,k] <- (k-1)
}

# parameters (include nugget)
par_s <- list(nugget = 0.12, c = 0.004, gamma = 0.25)
par_t <- list(a = 0.6, alpha = 0.35)
par_base_fs <- c(par_s, par_t, beta = 0.78)

par_lagr <- list(v1 = 5, v2 = 10, k = 2)

Cb_fs <- cor_fs(
    nugget = par_s$nugget,
    c = par_s$c,
    gamma = par_s$gamma,
    a = par_t$a,
    alpha = par_t$alpha,
    beta = 0.78,
    h = h,
    u = u
)

C_mix <- cor_stat(
    base="fs",
    lagrangian="lagr_tri",
    par_base=par_base_fs,
    par_lagr=par_lagr,
    h=h, h1=h1, h2=h2, u=u,
    lambda=0.37,
    base_fixed=FALSE
)

list(h1=h1, h2=h2, h=h, u=u,
     Cb_fs=Cb_fs, C_mix=C_mix)
"""

gold = ro.r(r_code)

# pull 3D inputs + references
h1 = jnp.asarray(r_to_numpy(gold.rx2("h1")), dtype=DTYPE)
h2 = jnp.asarray(r_to_numpy(gold.rx2("h2")), dtype=DTYPE)
h = jnp.asarray(r_to_numpy(gold.rx2("h")), dtype=DTYPE)
u = jnp.asarray(r_to_numpy(gold.rx2("u")), dtype=DTYPE)

Cb_fs_ref = jnp.asarray(r_to_numpy(gold.rx2("Cb_fs")), dtype=DTYPE)
C_mix_ref = jnp.asarray(r_to_numpy(gold.rx2("C_mix")), dtype=DTYPE)

# params mirrored from r_code
par_base_fs = dict(nugget=0.12, c=0.004, gamma=0.25, a=0.6, alpha=0.35, beta=0.78)
par_lagr = dict(v1=5.0, v2=10.0, k=2.0)

# ----------------------------
# 2) JAX computations
# ----------------------------
# assumes your cor_fs/cor_stat exist in Python
Cb_fs = cor.cor_fs(
    nugget=par_base_fs["nugget"],
    c=par_base_fs["c"],
    gamma=par_base_fs["gamma"],
    a=par_base_fs["a"],
    alpha=par_base_fs["alpha"],
    beta=par_base_fs["beta"],
    h=h,
    u=u,
)

C_mix = cor.cor_stat(
    base="fs",
    lagrangian="lagr_tri",
    par_base=par_base_fs,
    par_lagr=par_lagr,
    h=h,
    h1=h1,
    h2=h2,
    u=u,
    lam=0.37,
    base_fixed=False,
)

# ----------------------------
# 3) Compare whole + slice-wise
# ----------------------------
report_header(
    "R vs JAX (3D rdists scale=50; forced zeros in (1,2),(1,3),(1,4); nugget on)"
)
report("Cb_fs (3D)", Cb_fs, Cb_fs_ref)
report("C_mix  (3D)", C_mix, C_mix_ref)

print("\nSlice-wise diagnostics:")
report_slices("Cb_fs", Cb_fs, Cb_fs_ref)
report_slices("C_mix", C_mix, C_mix_ref)

# Optional: confirm the forced-zero entries are indeed zero in all slices
# (0-based python indices: (0,1),(0,2),(0,3))
pairs = [(0, 1), (0, 2), (0, 3)]
for i, j in pairs:
    print(f"\nforced pairs ({i},{j}) check:")
    print("h1:", np.asarray(h1[i, j, :]))
    print("h2:", np.asarray(h2[i, j, :]))
    print("h :", np.asarray(h[i, j, :]))
