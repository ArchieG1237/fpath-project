import numpy as np
import matplotlib.pyplot as plt

N = 100
x = np.zeros(N)
rng = np.random.default_rng()


epsilon = 1.4
a = 0.5


def local_action(x, i):
    ip = (i + 1) % N
    im = (i - 1) % N
    return(
        (a*x[i]**2)/2 + x[i] * (x[i] - x[im] - x[ip])/a
    )
    #return ((x[i] - x[im])**2 + (x[ip] - x[i])**2)/(2*a) + a*(x[i]**2)/2


def update(x):
    swaps = 0
    for i in range(0,N):
        zeta = rng.uniform(-epsilon, epsilon)
        x_temp = x[i]
        S_i = local_action(x, i)
        x[i] = x[i] + zeta
        S_f = local_action(x, i)
        dS = S_f - S_i
        if dS > 0 and np.exp(-dS) < rng.uniform(0,1):
            x[i] = x_temp
        else:
            swaps += 1

    #print((swaps / N) * 100)    
    return x

N_cf = 4000
N_cor = 20

samples = []

for i in range(10*N_cor):
    update(x)

for i in range(N_cf):
    for j in range(N_cor):
        update(x)
    samples.append(x[N//2])


plt.figure(figsize=(6, 4))

# numerical solution (histogram)
plt.hist(
    samples,
    bins=80,
    density=True,
    label="Numerical solution",
)

# bin centres
counts, bins = np.histogram(samples, bins=80, density=True)

centres = 0.5 * (bins[1:] + bins[:-1])

# analytic solution
psi2 = (1/np.sqrt(np.pi)) * np.exp(-centres**2)
plt.plot(
    centres,
    psi2,
    label=r"Analytic solution",
    linewidth=2
)

plt.xlim(-3.2, 3.2)
plt.xlabel("x")
plt.ylabel("Probability density")

plt.legend(frameon=False)   # clean look for reports
plt.tight_layout()
plt.show()

counts_raw, bins = np.histogram(samples, bins=80, density=False)
bin_width = bins[1] - bins[0]
centres = 0.5 * (bins[1:] + bins[:-1])

N_samples = len(samples)

rho_num = counts_raw / (N_samples * bin_width)
sigma = np.sqrt(counts_raw) / (N_samples * bin_width)
rho_ana = (1/np.sqrt(np.pi)) * np.exp(-centres**2)

mask = counts_raw > 0

chi2 = np.sum((rho_num[mask] - rho_ana[mask])**2 / sigma[mask]**2)
ndof = np.sum(mask) - 1

print(f"Chi^2 = {chi2:.2f}")
print(f"Chi^2 / ndof = {chi2/ndof:.2f}")


