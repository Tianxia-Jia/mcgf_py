import time
import jax
import jax.numpy as jnp

from linear_dynamic_system_backend import (
    DTYPE,
    _ABD_from_corrs_joint,
    _cov_par,
    _F_onestep,
    _Q_onestep,
    stationary_corr_from_model_corrs,
    _cov_joint,
    _covs_to_corrs,
)

from correlation import cor_base_stat

print("JAX devices:", jax.devices())


# ============================================================
# Problem size (change freely)
# ============================================================

n = 20  # number of locations
lag = 6  # p
L = lag + 1
dim = n * L

key = jax.random.PRNGKey(0)


# ============================================================
# Build random stable F
# ============================================================

# case 1, isotropic cor:
key, sub = jax.random.split(key)
coords = jax.random.uniform(sub, (n, 2), minval=0.0, maxval=20.0, dtype=DTYPE)

diff = coords[:, None, :] - coords[None, :, :]
h = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
h = jnp.broadcast_to(h[:, :, None], (n, n, L))

u = jnp.arange(0, L)
u = jnp.broadcast_to(u, (n, n, L))
C_true = cor_base_stat(
    base="fs",
    par_base={"c": 0.001, "gamma": 0.25, "a": 5, "alpha": 0.8, "beta": 0.9},
    h=h,
    u=u,
)
corrs_model = _cov_joint(C_true)

print("corrs_model shape:", corrs_model.shape, "dtype:", corrs_model.dtype)

# case 2, psd mat:
key, sub = jax.random.split(key)
coords = [coords]
for l in range(L - 1):
    coords.append(
        jax.random.uniform(
            sub, (n, 2), minval=0.0 + 2 * l, maxval=20.0 + 2 * l, dtype=DTYPE
        )
    )
coords = jnp.vstack(coords)

diff = coords[:, None, :] - coords[None, :, :]
h_lds = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
u = jnp.arange(0, L)
u = jnp.broadcast_to(u, (n, n, L))
u_lds_abs = _cov_joint(u)

cor_lds = cor_base_stat(
    base="fs",
    par_base={"c": 0.1, "gamma": 0.25, "a": 5, "alpha": 0.8, "beta": 0.5},
    h=h_lds,
    u=u_lds_abs,
)
lam = 0.3
corrs_model = corrs_model * (1 - lam) + lam * cor_lds

print("corrs_model shape:", corrs_model.shape, "dtype:", corrs_model.dtype)


def stat_cov_vec(corrs_model, n, lag, ridge_D=1e-10):
    dim = n * (lag + 1)
    A, B, D = _ABD_from_corrs_joint(corrs_model, lag)
    W, Sigma_e = _cov_par(A, B, D, ridge_D)
    F = _F_onestep(W, n, lag)
    Q = _Q_onestep(Sigma_e, n, lag)
    vecPhi = jnp.linalg.solve(
        jnp.eye(dim * dim, dtype=corrs_model.dtype) - jnp.kron(F, F),
        Q.reshape(-1, order="F"),
    )
    return 0.5 * (
        vecPhi.reshape((dim, dim), order="F") + vecPhi.reshape((dim, dim), order="F").T
    )


P_cov = stat_cov_vec(corrs_model, n, lag)
abs(P_cov - corrs_model).sum()
P_cor = _covs_to_corrs(P_cov)
abs(P_cor - corrs_model).sum()

A, B, D = _ABD_from_corrs_joint(P_cor, lag)
W, Sigma_e = _cov_par(A, B, D)
abs(W @ D @ W.T + Sigma_e - A).sum()

A, B, D = _ABD_from_corrs_joint(corrs_model, lag)
W, Sigma_e = _cov_par(A, B, D)
abs(W @ D @ W.T + Sigma_e - A).sum()


# ============================================================
# Benchmark stationary fixed-point solvers
# ============================================================

methods = ["scan_fp", "jaxopt_fpi", "jaxopt_anderson"]

# Controls (tune these)
fp_control = dict(
    tol=1e-1,
    maxiter=10000,
    # anderson-only:
    history_size=5,
    beta=1.0,
    fp_ridge=1e-5,
    # jaxopt:
    implicit_diff=True,
)

ridge = 1e-10  # Lyapunov ridge (added as Q + ridge*I internally)
n_runs = 1

print("\nBenchmarking stationary_corr_from_model_corrs")
print("------------------------------------------------")
print(f"{'method':18s}  {'avg ms':>10s}  {'max abs err':>15s}")

for method in methods:
    # ----------------------------
    # Warmup (compile)
    # ----------------------------
    out = stationary_corr_from_model_corrs(
        corrs_model,
        n=n,
        lag=lag,
        fp_method=method,
        fp_control=fp_control,
        ridge=ridge,
    )

    # compute error after warmup
    err = abs(out - P_cor).sum()

    # ----------------------------
    # Timed runs
    # ----------------------------
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = stationary_corr_from_model_corrs(
            corrs_model,
            n=n,
            lag=lag,
            fp_method=method,
            fp_control=fp_control,
            ridge=ridge,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = 1000.0 * sum(times) / len(times)

    print(f"{method:18s}  {avg_ms:10.3f}  {float(err):15.6e}")

print("\nDone.")
