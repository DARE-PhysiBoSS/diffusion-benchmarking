import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def adi_btcs_2d_heat(Nx, Ny, Nt, Lx, Ly, T, alpha, u0_func, bc_func):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = T / Nt
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    rx = alpha * dt / (dx**2)
    ry = alpha * dt / (dy**2)

    u = u0_func(X, Y)
    unew = np.empty_like(u)

    Ax = np.vstack((
        -rx * np.ones(Nx - 2), 
        (1 + 2 * rx) * np.ones(Nx - 2), 
        -rx * np.ones(Nx - 2)
    ))

    Ay = np.vstack((
        -ry * np.ones(Ny - 2), 
        (1 + 2 * ry) * np.ones(Ny - 2), 
        -ry * np.ones(Ny - 2)
    ))

    for n in range(Nt):
        u_star = np.copy(u)
        for j in range(1, Ny - 1):
            rhs = ry * u[1:-1, j - 1] + (1 - 2 * ry) * u[1:-1, j] + ry * u[1:-1, j + 1]
            u_star[1:-1, j] = solve_banded((1, 1), Ax, rhs)
        u_star = bc_func(u_star, x, y, n * dt + dt/2)

        for i in range(1, Nx - 1):
            rhs = rx * u_star[i - 1, 1:-1] + (1 - 2 * rx) * u_star[i, 1:-1] + rx * u_star[i + 1, 1:-1]
            unew[i, 1:-1] = solve_banded((1, 1), Ay, rhs)
        u = bc_func(unew, x, y, (n + 1) * dt)

    return x, y, u

def u0_func(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def bc_func(u, x, y, t):
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    return u

def exact_solution(x, y, t, alpha):
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2 * np.pi**2 * alpha * t)

# Parameters
Lx = Ly = 1000.0
T = 10
alpha = 0.01

grid_sizes = [21, 41, 61, 81, 101]  # Nx = Ny
errors = []
dx_values = []

for N in grid_sizes:
    Nx = Ny = N
    dx = Lx / (Nx - 1)
    dt = 0.01  # keep stable and consistent time stepping
    Nt = int(T / dt)
    dt = T / Nt  # recompute exact dt

    x, y, u_num = adi_btcs_2d_heat(Nx, Ny, Nt, Lx, Ly, T, alpha, u0_func, bc_func)
    u_exact = exact_solution(x, y, T, alpha)

    error = np.sqrt(np.mean((u_num - u_exact)**2))
    errors.append(error)
    dx_values.append(dx)
    print(f"N={N}, dx={dx:.5f}, dt={dt:.2e}, Error={error:.2e}")

# Plotting
plt.figure(figsize=(8, 6))
plt.loglog(dx_values, errors, 'o-', label='L2 Error')
plt.loglog(dx_values, [e*(dx_values[0]/d)**2 for d, e in zip(dx_values, errors)], 'k--', label='O(dx²) reference')
plt.xlabel("Grid spacing Δx")
plt.ylabel("L2 Error")
plt.title("Convergence of ADI-BTCS for 2D Heat Equation")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()