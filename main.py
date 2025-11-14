import numpy as np
import math
import vegas
import matplotlib.pyplot as plt

# --- constants ---
m = 1.0
T = 4.0
N = 9
a = T / N
V = lambda x: 0.5 * x**2
L = 1.8

# --- Euclidean action ---
def energy(x0, xN, xs):
    E = 0.0
    x_prev = x0
    for x in xs:
        mid = 0.5*(x_prev + x)
        E += 0.5*m*((x - x_prev)/a)**2 + V(mid)
        x_prev = x
    mid = 0.5*(x_prev + xN)
    E += 0.5*m*((xN - x_prev)/a)**2 + V(mid)
    return E

def make_integrand(x0, xN):
    def f(xs):
        return math.exp(-a * energy(x0, xN, xs))
    return f

# --- VEGAS propagator ---
def G_hat(x0, xN):
    integ = vegas.Integrator([(-L, L)] * (N-1))

    # warmup
    integ(make_integrand(x0, xN), nitn=5, neval=4000)

    # main
    result = integ(make_integrand(x0, xN), nitn=8, neval=20000)

    A = (m/(2*math.pi*a))**(N/2)
    return A * result.mean

# -------------------------------
# COMPUTE SCATTER POINTS (10 pts)
# -------------------------------
x_scatter = np.linspace(0, 2, 15)
G_scatter = np.array([G_hat(x, x) for x in x_scatter])

# -------------------------------
# COMPUTE NORMALISATION USING MANY POINTS
# -------------------------------
x_fine = np.linspace(0, 2, 41)
G_fine = np.array([G_hat(x, x) for x in x_fine])
Z = np.trapz(G_fine, x_fine)

psi2_scatter = 0.5 * (G_scatter / Z)

# -------------------------------
# ANALYTIC CURVE FOR PLOTTING
# -------------------------------
x_plot = np.linspace(0, 2, 200)
psi2_exact = (1/np.sqrt(np.pi)) * np.exp(-x_plot**2)

# -------------------------------
# PLOT
# -------------------------------
plt.figure(figsize=(7,5))

plt.scatter(x_scatter, psi2_scatter, color='tab:blue', 
            label='VEGAS |ψ(x)|²', s=45)

plt.plot(x_plot, psi2_exact, '--', color='tab:orange', 
         linewidth=2, label='Analytic |ψ(x)|²')

plt.xlim(0, 2)
plt.xlabel('x')
plt.ylabel('|ψ(x)|²')
plt.legend()
plt.grid(False)
plt.show()
