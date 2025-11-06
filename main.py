import numpy as np
import matplotlib.pyplot as plt

# --- constants ---
m = 1.0
T = 4.0
N = 7
a = T / N
V = lambda x: 0.5 * x**2
M = 100000

# --- random paths ---
paths = np.random.uniform(-2.5, 2.5, size=(M, N-1))

# --- functions ---
def energy(x0, xN, path):
    E = 0
    x_prev = x0
    for j in range(N-1):
        x_next = path[j]
        mid = 0.5 * (x_prev + x_next)
        E += (m/2) * ((x_next - x_prev)/a)**2 + V(mid)
        x_prev = x_next
    mid = 0.5 * (x_prev + xN)
    E += (m/2) * ((xN - x_prev)/a)**2 + V(mid)
    return E

def G_hat(x0, xN):
    total = 0
    for i in range(M):
        E = energy(x0, xN, paths[i])
        total += np.exp(-a * E)
    G = (m / (2 * np.pi * a))**(N/2) * (total / M)
    return G

# --- compute path-integral ---
x_vals = np.linspace(0, 2, 41)
G_vals = np.array([G_hat(x, x) for x in x_vals])

dx = x_vals[1] - x_vals[0]
Z = dx * np.sum(G_vals)
psi2 = 0.5 * (G_vals / Z)

# --- analytical solution ---
psi2_exact = (1 / np.sqrt(np.pi)) * np.exp(-x_vals**2)

# --- check normalisation and value at x=1 ---
print("∫|ψ₀|² dx ≈", np.trapezoid(psi2, x_vals))
idx_1 = np.argmin(np.abs(x_vals - 1))
print("|ψ₀(1)|² numeric =", psi2[idx_1])
print("|ψ₀(1)|² analytic =", psi2_exact[idx_1])

# --- plot ---
plt.plot(x_vals, psi2, label='Path integral', lw=2)
plt.plot(x_vals, psi2_exact, '--', label='Analytical $|ψ₀(x)|²$', lw=2)
plt.xlabel('x')
plt.ylabel('|ψ₀(x)|²')
plt.legend()
plt.show()
