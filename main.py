import numpy as np
import matplotlib.pyplot as plt
print("started")

N = 100
rng = np.random.default_rng()

lambdas = [0.0, 0.5, 1.0, 2.0]
epsilon = 1.4
a = 0.5


def local_action(x, i, lam):
    ip = (i + 1) % N
    im = (i - 1) % N
    kinetic = ((x[i] - x[im])**2 + (x[ip] - x[i])**2) / (2 * a)
    potential = a * (0.5 * x[i]**2 + lam * x[i]**4)
    return kinetic + potential


def update(x, lam):
    accepted = 0
    proposed = N

    for i in range(N):
        zeta = rng.uniform(-epsilon, epsilon)
        x_old = x[i]

        S_i = local_action(x, i, lam)
        x[i] = x_old + zeta
        S_f = local_action(x, i, lam)

        dS = S_f - S_i
        if dS <= 0 or rng.uniform(0, 1) < np.exp(-dS):
            accepted += 1
        else:
            x[i] = x_old

    return accepted, proposed



N_cf = 20000
N_cor = 40

# ---- create subplots ONCE ----
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
axes = axes.flatten()
acceptance_results = {}
for ax, lam in zip(axes, lambdas):

    x = np.zeros(N)
    samples = []

    total_acc = 0
    total_prop = 0

    # thermalisation
    for _ in range(10 * N_cor):
        acc, prop = update(x, lam)
        total_acc += acc
        total_prop += prop

    # sampling
    for _ in range(N_cf):
        for _ in range(N_cor):
            acc, prop = update(x, lam)
            total_acc += acc
            total_prop += prop
        samples.append(x[N // 2])

    
    acceptance_results[lam] = total_acc / total_prop
    print(f"lambda = {lam}, acceptance rate = {acceptance_results[lam]:.3f}", flush=True)



    bins = np.linspace(-3, 3, 80)
    ax.hist(samples, bins=bins, density=True, alpha=0.8)
    ax.set_title(fr"$\lambda = {lam}$")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylabel("Probability density")
    ax.set_xlabel("x")

plt.tight_layout()
plt.show()
