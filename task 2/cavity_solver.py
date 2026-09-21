import subprocess
from datetime import datetime

import numpy as np

# ==========================================================================
# PARAMS  (problem statement, Part 2)
# ==========================================================================
L = 0.1          # m, cavity width
H = 0.1          # m, cavity height
RHO = 1000.0     # kg/m3
U_LID = 1.0      # m/s, lid velocity (top wall)

N_PRESSURE_ITS = 50   # Jacobi sweeps per timestep (pressure field is warm-
                       # started from the previous step, so a full re-solve
                       # to tight tolerance every step is not needed)


def nu_from_Re(Re):
    """Kinematic viscosity for the desired Reynolds number, Re = U*L/nu."""
    return U_LID * L / Re


# ==========================================================================
# PROJECTION-METHOD SOLVER
#   Follows the algorithm in the assignment appendix (Eqs 9-13): predict a
#   non-divergence-free u* from the momentum equation (ignoring pressure),
#   solve a pressure Poisson equation so that correcting u* by -dt/rho*grad(p)
#   gives a divergence-free field, then apply that correction.
#
#   Staggered (MAC) grid, matching the class demo's array/indexing
#   convention:
#     u : shape (NX+1, NY+2)  x-velocity on vertical cell faces,
#                              +1 ghost row top and bottom
#     v : shape (NX+2, NY+1)  y-velocity on horizontal cell faces,
#                              +1 ghost column left and right
#     p : shape (NX+2, NY+2)  cell centres, +1 ghost layer all round
# ==========================================================================
def solve_cavity(Re, NX, NY, dt, n_steps, brinkman_k=None,
                  check_every=200, tol_steady=1.0e-5, verbose=True):
    """
    Solve lid-driven cavity flow (or the Brinkman-modified porous cavity,
    if brinkman_k is given) up to n_steps timesteps, stopping early once
    the field stops changing (see tol_steady).
    """
    nu = nu_from_Re(Re)
    dx = L / NX
    dy = H / NY
    dxdy = dx * dy

    xnodes = np.linspace(dx / 2., L - dx / 2., NX)   # pressure-cell centres
    ynodes = np.linspace(dy / 2., H - dy / 2., NY)

    u = np.zeros((NX + 1, NY + 2))
    v = np.zeros((NX + 2, NY + 1))
    p = np.zeros((NX + 2, NY + 2))

    J_u_x = np.zeros((NX, NY))
    J_u_y = np.zeros((NX - 1, NY + 1))
    J_v_x = np.zeros((NX + 1, NY - 1))
    J_v_y = np.zeros((NX, NY))

    ut = np.zeros_like(u)
    vt = np.zeros_like(v)
    p_next = np.zeros_like(p)

    # BOUNDARY CONDITIONS -------------------------------------------------
    # No-slip on left/right/bottom walls, moving lid (u=U_LID, v=0) on top.
    # Dirichlet components that sit exactly ON a u- or v-node are set
    # directly; components that fall BETWEEN a ghost row/column and the
    # first interior row/column are enforced by mirroring the ghost value
    # (ghost = 2*wall_value - interior), the same "ghost cell" convention
    # used for the boundary conductances in Task 1.
    def apply_u_bc(uf):
        uf[0, :] = 0.0                      # left wall (u exactly on node)
        uf[-1, :] = 0.0                     # right wall
        uf[:, 0] = -uf[:, 1]                # bottom wall, u=0 -> mirror
        uf[:, -1] = 2.0 * U_LID - uf[:, -2]  # top wall (lid), u=U_LID -> mirror

    def apply_v_bc(vf):
        vf[:, 0] = 0.0                      # bottom wall (v exactly on node)
        vf[:, -1] = 0.0                     # top wall (lid is tangential only)
        vf[0, :] = -vf[1, :]                # left wall, v=0 -> mirror
        vf[-1, :] = -vf[-2, :]              # right wall, v=0 -> mirror

    def apply_p_bc(pf):
        # Homogeneous Neumann (zero normal gradient) on all four walls -
        # no flow through any wall, so no pressure gradient normal to it.
        pf[0, :] = pf[1, :]
        pf[-1, :] = pf[-2, :]
        pf[:, 0] = pf[:, 1]
        pf[:, -1] = pf[:, -2]

    apply_u_bc(u)
    apply_v_bc(v)

    inv_k = 0.0 if brinkman_k is None else nu / brinkman_k
    beta = 1.0 / (2.0 / dx**2 + 2.0 / dy**2)

    sim_time = 0.0
    u_prev_check = u.copy()
    steps_run = 0

    for step in range(n_steps):
        # 1. PREDICT u*, v* : convective + diffusive flux, momentum eqn
        #    (pressure omitted - this is Eq. 11 of the appendix)
        J_u_x[:, :] = 0.25 * (u[:-1, 1:-1] + u[1:, 1:-1])**2 \
            - nu * (u[1:, 1:-1] - u[:-1, 1:-1]) / dx
        J_u_y[:, :] = 0.25 * (u[1:-1, 1:] + u[1:-1, :-1]) * (v[1:-2, :] + v[2:-1, :]) \
            - nu * (u[1:-1, 1:] - u[1:-1, :-1]) / dy

        J_v_x[:, :] = 0.25 * (u[:, 2:-1] + u[:, 1:-2]) * (v[1:, 1:-1] + v[:-1, 1:-1]) \
            - nu * (v[1:, 1:-1] - v[:-1, 1:-1]) / dx
        J_v_y[:, :] = 0.25 * (v[1:-1, 1:] + v[1:-1, :-1])**2 \
            - nu * (v[1:-1, 1:] - v[1:-1, :-1]) / dy

        ut[1:-1, 1:-1] = u[1:-1, 1:-1] - (dt / dxdy) * (
            dy * (J_u_x[1:, :] - J_u_x[:-1, :]) + dx * (J_u_y[:, 1:] - J_u_y[:, :-1]))
        vt[1:-1, 1:-1] = v[1:-1, 1:-1] - (dt / dxdy) * (
            dy * (J_v_x[1:, :] - J_v_x[:-1, :]) + dx * (J_v_y[:, 1:] - J_v_y[:, :-1]))

        # Task 2b only: Brinkman drag term, -nu/k * u, added explicitly
        if brinkman_k is not None:
            ut[1:-1, 1:-1] -= dt * inv_k * u[1:-1, 1:-1]
            vt[1:-1, 1:-1] -= dt * inv_k * v[1:-1, 1:-1]

        apply_u_bc(ut)
        apply_v_bc(vt)

        # 2. PRESSURE POISSON: lap(p) = rho/dt * div(u*)   (Eq. 12)
        divergence = (ut[1:, 1:-1] - ut[:-1, 1:-1]) / dx + \
                     (vt[1:-1, 1:] - vt[1:-1, :-1]) / dy
        rhs = RHO / dt * divergence
        for _ in range(N_PRESSURE_ITS):
            apply_p_bc(p)
            p_next[1:-1, 1:-1] = beta * (
                (p[2:, 1:-1] + p[:-2, 1:-1]) / dx**2 +
                (p[1:-1, 2:] + p[1:-1, :-2]) / dy**2 - rhs)
            p[1:-1, 1:-1] = p_next[1:-1, 1:-1]
        apply_p_bc(p)

        # 3. CORRECT u*, v* to be divergence-free  (Eq. 13)
        u[1:-1, 1:-1] = ut[1:-1, 1:-1] - (dt / RHO) * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / dx
        v[1:-1, 1:-1] = vt[1:-1, 1:-1] - (dt / RHO) * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / dy
        apply_u_bc(u)
        apply_v_bc(v)

        sim_time += dt
        steps_run = step + 1

        if (steps_run % check_every == 0) or (step == n_steps - 1):
            change = np.max(np.abs(u - u_prev_check)) / U_LID
            u_prev_check = u.copy()
            if verbose:
                print(f"  step {steps_run:6d}/{n_steps}  t={sim_time:7.4f} s  "
                      f"max|du|/U_lid={change:.3e}")
            if not np.isfinite(change):
                raise FloatingPointError("Simulation diverged (NaN/Inf) - reduce dt.")
            if change < tol_steady:
                if verbose:
                    print(f"  -> steady state reached at t={sim_time:.4f} s "
                          f"(step {steps_run})")
                break

    return dict(u=u, v=v, p=p, x=xnodes, y=ynodes, dx=dx, dy=dy, nu=nu,
                sim_time=sim_time, steps=steps_run, NX=NX, NY=NY)


def cell_centre_velocity(result):
    """Interpolate the staggered u, v fields onto pressure-cell centres."""
    u, v = result["u"], result["v"]
    uc = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])   # shape (NX, NY)
    vc = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])   # shape (NX, NY)
    return uc, vc


def stamp_figure(fig):
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd="/home/claude/mech6480_a1",
        ).decode().strip()
    except Exception:
        git_hash = "no-git"
    fig.text(0.995, 0.005, f"Generated {datetime.now():%Y-%m-%d %H:%M}  |  git {git_hash}",
              ha="right", va="bottom", fontsize=7, color="0.5")
