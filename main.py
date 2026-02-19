import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------- settings ----------
N = 300
a = 0.05
eps = 0.4
lambdas = [0.0, 1.0, 2.0, 5.0]
N_cf = 300_000
N_cor = 100
maxN = 32

# --- Lepage-style binning + bootstrap settings ---
binsize = 100          # aim for ~50–100 bins: 10_000/100 = 100 bins
nboot = 500
rng_seed = 12345

# --- Manual plateau windows (indices in DeltaE array) ---
# DeltaE[t] corresponds to t = (t_index)*a where t_index = 0..N-2
# Edit these by hand after looking at the plot.
plateau_windows = {
    0.0: (0, 31),   # example: t = 0.4 .. 1.2
    1.0: (0, 22),
    2.0: (0, 22),
    5.0: (0, 17),
}

# ---------- theory curve: solve Schrödinger ground state ----------
def ground_density(lam, L=4.0, M=3000):
    x = np.linspace(-L, L, M)
    dx = x[1] - x[0]
    V = 0.5 * x**2 + lam * x**4

    main = 1/dx**2 + V
    off  = (-0.5/dx**2) * np.ones(M - 1)
    H = sp.diags([off, main, off], [-1, 0, 1], format="csc")

    E_vals, psi = spla.eigsh(H, k=2, which="SA")
    psi = psi[:, 0]
    psi /= np.sqrt(np.sum(psi**2) * dx)
    return x, psi**2, E_vals


# ---------- correlator on one configuration ----------
@njit(cache=True, fastmath=True)
def corr_one_config(x):
    Nloc = x.size
    G = np.zeros(Nloc)
    for n in range(Nloc):
        s = 0.0
        for j in range(Nloc):
            jp = j + n
            if jp >= Nloc:
                jp -= Nloc
            s += x[jp] * x[j]
        G[n] = s / Nloc
    return G


# ---------- Metropolis update ----------
@njit(cache=True, fastmath=True)
def sweep(x, lam, eps, a):
    Nloc = x.size
    acc = 0
    for i in range(Nloc):
        ip = 0 if i + 1 == Nloc else i + 1
        im = Nloc - 1 if i == 0 else i - 1

        xo = x[i]
        xn = xo + (2.0 * np.random.random() - 1.0) * eps

        Si = ((xo - x[im])**2 + (x[ip] - xo)**2) / (2*a) + a*(0.5*xo**2 + lam*xo**4)
        Sf = ((xn - x[im])**2 + (x[ip] - xn)**2) / (2*a) + a*(0.5*xn**2 + lam*xn**4)

        dS = Sf - Si
        if dS <= 0.0 or np.random.random() < np.exp(-dS):
            x[i] = xn
            acc += 1
    return acc


# ---------- Lepage binning + bootstrap ----------
def bin_ensemble(G, binsize):
    """
    G: shape (N_cf, N_t)
    returns binned correlators: shape (N_bins, N_t)
    """
    Ncf, Nt = G.shape
    Nbins = Ncf // binsize
    if Nbins < 2:
        raise ValueError("binsize too large: need at least 2 bins for bootstrap.")
    G = G[:Nbins * binsize]  # drop remainder
    return G.reshape(Nbins, binsize, Nt).mean(axis=1)


def bootstrap_deltaE_boot_from_binned(G_binned, a, nboot=500, seed=0):
    """
    Returns bootstrap replicas of ΔE(t) (so you can get a plateau error properly),
    plus the mean and std across replicas.
    """
    rng = np.random.default_rng(seed)
    Nbins, Nt = G_binned.shape
    DeltaE_boot = np.empty((nboot, Nt - 1))

    for b in range(nboot):
        idx = rng.integers(0, Nbins, size=Nbins)      # resample bins WITH replacement
        G_mean_b = G_binned[idx].mean(axis=0)         # mean correlator for this bootstrap copy
        DeltaE_boot[b] = np.log(G_mean_b[:-1] / G_mean_b[1:]) / a

    DeltaE_mean = DeltaE_boot.mean(axis=0)
    DeltaE_err  = DeltaE_boot.std(axis=0, ddof=1)
    return DeltaE_mean, DeltaE_err, DeltaE_boot


def plateau_from_bootstrap(DeltaE_boot, tmin, tmax):
    """
    Manual plateau: average ΔE(t) over [tmin:tmax) for each bootstrap replica,
    then take mean±std across replicas.
    """
    plateau_boot = DeltaE_boot[:, tmin:tmax].mean(axis=1)
    return plateau_boot.mean(), plateau_boot.std(ddof=1)


# compile once
x_tmp = np.zeros(N)
sweep(x_tmp, 0.0, eps, a)
corr_one_config(x_tmp)

# ---------- run + plot ----------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.ravel()
axes2 = axes2.ravel()

bins = np.linspace(-3, 3, 300)

for i, (ax, lam) in enumerate(zip(axes, lambdas)):
    x = np.zeros(N)

    # thermalise
    for _ in range(10 * N_cor):
        sweep(x, lam, eps, a)

    x2_list = []
    x4_list = []

    samples = np.empty(N_cf * N)
    k = 0
    acc = 0
    G = np.empty((N_cf, N))

    for cfg in range(N_cf):
        for _ in range(N_cor):
            acc += sweep(x, lam, eps, a)

        samples[k:k+N] = x
        k += N

        x2_list.append(np.mean(x*x))
        x4_list.append(np.mean(x*x*x*x))

        G[cfg] = corr_one_config(x)

    # ---------- Lepage: bin + bootstrap ----------
    G_binned = bin_ensemble(G, binsize)

    DeltaE_mean, DeltaE_err, DeltaE_boot = bootstrap_deltaE_boot_from_binned(
        G_binned, a, nboot=nboot, seed=rng_seed + i
    )

    # ---------- Manual plateau choice (YOU choose these indices) ----------
    tmin, tmax = plateau_windows[lam]
    E_gap, E_gap_err = plateau_from_bootstrap(DeltaE_boot, tmin, tmax)
    # ---------------------------------------------------------------------

    x2_vals = np.array(x2_list)
    x4_vals = np.array(x4_list)

    x2_mc = np.mean(x2_vals)
    x4_mc = np.mean(x4_vals)

    x2_err = np.std(x2_vals, ddof=1) / np.sqrt(N_cf)
    x4_err = np.std(x4_vals, ddof=1) / np.sqrt(N_cf)

    # Numerical Schrodinger Moments
    x_vals, prob, E_vals = ground_density(lam)
    dx = x_vals[1] - x_vals[0]
    x2_sc = np.sum((x_vals**2) * prob) * dx
    x4_sc = np.sum((x_vals**4) * prob) * dx

    E0 = E_vals[0]
    E1 = E_vals[1]
    deltaE1_sc = E1 - E0

    # If you want E1 from your MC gap using SC E0:
    E1_mc = E0 + E_gap
    E1_mc_err = E_gap_err

    print(f"<x^2>_MC = {x2_mc:.5f} ± {x2_err:.5f} <x^2>_SC = {x2_sc:.5f}")
    print(f"<x^4>_MC = {x4_mc:.5f} ± {x4_err:.5f} <x^4>_SC = {x4_sc:.5f}")
    print(f"E0_MC = {(x2_mc + 3 * lam * x4_mc):.5f} E0_SC = {E0:.5f}")
    print(f"lambda={lam:>3}, acceptance={acc/(N_cf*N_cor*N):.3f}")
    print(f"binsize={binsize}, N_bins={G_binned.shape[0]}, nboot={nboot}")
    print(f"manual plateau: indices [{tmin}:{tmax}) -> t in [{tmin*a:.2f}, {tmax*a:.2f})")
    print(f"(E1-E0)_MC = {E_gap:.5f} ± {E_gap_err:.5f}   (E1-E0)_SC = {deltaE1_sc:.5f}")
    print(f"E1_MC = {E1_mc:.5f} ± {E1_mc_err:.5f}         E1_SC = {E1:.5f}\n")

    # ---------- histogram ----------
    ax.hist(samples, bins=bins, density=True, alpha=0.8)
    ax.plot(x_vals, prob, "r-", lw=1)

    ax.set_title(fr"$\lambda={lam}$", pad=8)
    ax.set_xlim(-3.2, 3.2)
    ax.set_xlabel("x", labelpad=6)
    ax.set_ylabel("Probability density")

    # ---------- DeltaE plot ----------
    n_vals = np.arange(maxN)
    t_vals = n_vals * a

    axes2[i].axhline(deltaE1_sc,
                     linestyle='--',
                     color='red',
                     linewidth=1)

    axes2[i].errorbar(t_vals,
                      DeltaE_mean[:maxN],
                      yerr=DeltaE_err[:maxN],
                      fmt='o',
                      color='black')

    # show your chosen plateau band on the plot
    axes2[i].axvspan(tmin*a, tmax*a, alpha=0.15)

    axes2[i].set_ylim(deltaE1_sc - 0.5, deltaE1_sc + 0.5)
    axes2[i].set_title(fr"$\lambda={lam}$")
    axes2[i].set_xlabel("t")
    axes2[i].set_ylabel("E1 - E0")

fig.subplots_adjust(hspace=0.35)
fig2.subplots_adjust(hspace=0.35)
plt.tight_layout()
plt.show()
