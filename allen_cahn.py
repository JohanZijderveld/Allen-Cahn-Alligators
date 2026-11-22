import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Allen-Cahn: u_t - eps^2 u_xx + u^3 - u = 0 on (a, b) x (0, T)


def exact(x, t, eps):
    # travelling wave solution, speed c = 3*eps/sqrt(2)
    c = 3 * eps / np.sqrt(2)
    return 0.5 * (1 - np.tanh((x - c * t) / (2 * np.sqrt(2) * eps)))


def exact_constant(t, c0):
    # solution of c' = c - c^3 with c(0) = c0 (for u(x,t) = c(t))
    return c0 * np.exp(t) / np.sqrt(1 - c0**2 + c0**2 * np.exp(2 * t))


def energy(U, h, eps):
    # discrete version of E(u) = int eps^2/2 u_x^2 + 1/4 (u^2 - 1)^2 dx
    ux = (U[1:] - U[:-1]) / h
    return h * np.sum(eps**2 / 2 * ux**2) + h * np.sum((U**2 - 1)**2 / 4)


def forward_euler(U, h, tau, eps):
    # update interior points, everything at the old time level
    U_new = U.copy()
    lap = (U[2:] - 2 * U[1:-1] + U[:-2]) / h**2
    U_new[1:-1] = U[1:-1] + tau * (eps**2 * lap - U[1:-1]**3 + U[1:-1])
    return U_new


def semi_implicit(U, h, tau, eps, A, left, right):
    # diffusion at the new time level, u^3 - u at the old one
    # solve A U_new = U - tau(U^3 - U) + boundary terms
    r = tau * eps**2 / h**2
    rhs = U[1:-1] - tau * (U[1:-1]**3 - U[1:-1])
    rhs[0] += r * left
    rhs[-1] += r * right
    U_new = U.copy()
    U_new[1:-1] = spsolve(A, rhs)
    return U_new


def solve(method, h, tau, eps=0.01, a=-2, b=4, T=10, u0=None, bc=None, save_every=100):
    # method is "forward" or "semi"
    # u0(x) is the initial condition, bc(t) gives (left, right) boundary values
    # default to the exact travelling wave for both
    if u0 is None:
        u0 = lambda x: exact(x, 0, eps)
    if bc is None:
        bc = lambda t: (exact(a, t, eps), exact(b, t, eps))

    N = round((b - a) / h)
    steps = round(T / tau)
    x = np.linspace(a, b, N + 1)
    U = u0(x)

    # matrix for the semi-implicit scheme: (1 + 2r) on the diagonal, -r either side
    r = tau * eps**2 / h**2
    A = diags([-r, 1 + 2 * r, -r], [-1, 0, 1], shape=(N - 1, N - 1), format="csc")

    E = [energy(U, h, eps)]
    saved = [U.copy()]

    for n in range(1, steps + 1):
        t = n * tau
        left, right = bc(t)
        if method == "forward":
            U = forward_euler(U, h, tau, eps)
        else:
            U = semi_implicit(U, h, tau, eps, A, left, right)
        U[0], U[-1] = left, right

        E.append(energy(U, h, eps))
        if n % save_every == 0:
            saved.append(U.copy())

        if np.max(np.abs(U)) > 1e6:
            print("blew up at t =", t)
            break

    return x, U, np.array(E), saved