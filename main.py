import numpy as np
import matplotlib.pyplot as plt
from numba import njit

N = 100
lambdas = [0.0, 0.5, 1.0, 2.0]
epsilon = 1.4
a = 0.5

@njit(cache=True, fastmath=True)
def update_numba(x, lam, epsilon, a):
    N = x.shape[0]
    accepted = 0

    for i in range(N):
        ip = i + 1
        if ip == N:
            ip = 0
        im = i - 1
        if im == -1:
            im = N - 1

        x_old = x[i]
        zeta = (2.0 * np.random.random() - 1.0) * epsilon

        kin_i = ((x_old - x[im])**2 + (x[ip] - x_old)**2) / (2.0 * a)
        pot_i = a * (0.5 * x_old**2 + lam * x_old**4)
        S_i = kin_i + pot_i

        x_new = x_old + zeta

        kin_f = ((x_new - x[im])**2 + (x[ip] - x_new)**2) / (2.0 * a)
        pot_f = a * (0.5 * x_new**2 + lam * x_new**4)
        S_f = kin_f + pot_f

        dS = S_f - S_i

        if dS <= 0.0 or np.random.random() < np.exp(-dS):
            x[i] = x_new
            accepted += 1
        else:
            x[i] = x_old

    return accepted


N_cf = 1000000
N_cor = 30

# ---------------- Step 1: sample multiple time slices per configuration ----------------
# Pick a stride (spacing) between time slices to reduce within-path correlation.
# Start with 10; if you want more conservative, try 15.
stride = 10
n_slices = (N + stride - 1) // stride  # number of slices collected per configuration

# ---------------- Step 3: Freedman–Diaconis binning ----------------
# We'll use bins="fd" in hist() rather than a fixed number of bins.

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
axes = axes.flatten()

# warm up compile once
x0 = np.zeros(N)
_ = update_numba(x0, 0.0, epsilon, a)

for ax, lam in zip(axes, lambdas):
    x = np.zeros(N)

    # allocate enough space for all collected samples
    samples = np.empty(N_cf * n_slices, dtype=np.float64)
    idx = 0

    total_acc = 0
    total_prop = 0

    # thermalisation
    for _ in range(10 * N_cor):
        total_acc += update_numba(x, lam, epsilon, a)
        total_prop += N

    # sampling
    for _ in range(N_cf):
        for _ in range(N_cor):
            total_acc += update_numba(x, lam, epsilon, a)
            total_prop += N

        # Step 1: take multiple (subsampled) time slices from the same configuration
        for t in range(0, N, stride):
            samples[idx] = x[t]
            idx += 1

    samples = samples[:idx]  # trim to filled portion

    acc_rate = total_acc / total_prop
    print(f"lambda = {lam}, acceptance rate = {acc_rate:.3f}", flush=True)

    # Step 3: FD binning
    bins = np.linspace(-3, 3, 120)
    ax.hist(samples, bins=bins, density=True, alpha=0.8)

    ax.set_title(fr"$\lambda = {lam}$")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylabel("Probability density")
    ax.set_xlabel("x")

plt.tight_layout()
plt.show()
