import time
import subprocess
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ==========================================================================
# PARAMS   (Table 1 of the assignment)
# ==========================================================================
LX = 1000.          # m
LY = 1000.          # m
H_BC = 100.          # m, Dirichlet head on all four edges

K = 2.0e-6          # m/s, hydraulic conductivity
B = 5.0             # m, aquifer thickness
T = K * B           # m2/s, transmissivity

QI = 0.02            # m3/s, injection rate
QE = QI / 4.0         # m3/s, extraction rate (per well)

# wells: name, x, y, Q   (+ve Q = injection, -ve Q = extraction)
WELLS = [
    ("Injector",    500., 500.,  QI),
    ("Extractor 1", 800., 500., -QE),
    ("Extractor 2", 250., 500., -QE),
    ("Extractor 3", 500., 720., -QE),
    ("Extractor 4", 500., 220., -QE),
]


# ==========================================================================
# GRID + CREATE MATRIX + SOLVE, all wrapped in one function so the
# convergence study below can call it once per resolution.
# ==========================================================================
def solve_case(NX, NY):
    """
    Assemble and solve the steady-state FVM system for hydraulic head
    on an NX x NY grid.

    This follows the same matrix set-up we used for 2D unsteady
    diffusion (wk5_2d_unsteady.py: id(), A, b, EAST/WEST/NORTH/SOUTH
    loop), with three differences:
      1. No unsteady term - this problem is steady-state, so there is
         no rho*dx*dy/dt contribution to aP, and no b[idx] += ...*T_old.
      2. Well source terms are added after the main loop.
      3. A is assembled SPARSE and solved with a sparse solver, not
         np.linalg.inv(), because the convergence study needs grids up
         to 241 x 241 (~58,000 unknowns) - a dense 58,000 x 58,000
         array would need ~27 TB, and its inverse is not something we
         actually need here anyway (unlike the unsteady case, we only
         solve this system ONCE, so pre-inverting buys us nothing).
    """
    dx = LX / NX
    dy = LY / NY
    x = np.linspace(dx / 2., LX - dx / 2., NX)
    y = np.linspace(dy / 2., LY - dy / 2., NY)

    # locate the (i,j) cell containing each well - the assignment
    # requires wells to sit strictly inside a cell, not on a face
    well_idx = []
    for name, wx, wy, Q in WELLS:
        fi, fj = wx / dx, wy / dy
        if abs(fi - round(fi)) < 1e-9 or abs(fj - round(fj)) < 1e-9:
            raise ValueError(f"{name} sits on a cell face for NX={NX} "
                              f"- choose a different resolution.")
        well_idx.append((name, int(fi), int(fj), Q))

    # CREATE MATRIX
    def id(i, j):
        return i * NY + j

    N = NX * NY
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)

    aE = T * dy / dx     # east/west face conductance
    aN = T * dx / dy     # north/south face conductance

    for i in range(NX):
        for j in range(NY):
            idx = id(i, j)
            # EAST
            if i < (NX - 1):
                A[idx, id(i + 1, j)] += -aE
                A[idx, idx] += aE
            else:
                A[idx, idx] += 2. * aE          # one-sided difference
                b[idx] += 2. * aE * H_BC
            # WEST
            if i > 0:
                A[idx, id(i - 1, j)] += -aE
                A[idx, idx] += aE
            else:
                A[idx, idx] += 2. * aE
                b[idx] += 2. * aE * H_BC
            # NORTH
            if j < (NY - 1):
                A[idx, id(i, j + 1)] += -aN
                A[idx, idx] += aN
            else:
                A[idx, idx] += 2. * aN
                b[idx] += 2. * aN * H_BC
            # SOUTH
            if j > 0:
                A[idx, id(i, j - 1)] += -aN
                A[idx, idx] += aN
            else:
                A[idx, idx] += 2. * aN
                b[idx] += 2. * aN * H_BC

    # WELL SOURCE TERMS
    # S_h = +/- Q/(dx*dy) in the well cell, so the volume integral of
    # S_h over that cell (what actually appears on the RHS) is just
    # +/- Q directly.
    for name, i, j, Q in well_idx:
        b[id(i, j)] += Q

    A = A.tocsr()
    tic = time.time()
    h = spla.spsolve(A, b).reshape(NX, NY)
    toc = time.time()

    return x, y, dx, dy, h, (toc - tic), well_idx


# ==========================================================================
# DARCY FLUX
# ==========================================================================
def face_darcy_flux(h, dx, dy):
    """
    Darcy flux q = -K grad(h) evaluated AT CELL FACES (not yet averaged
    to centres). x_flux has shape (NX+1, NY): x_flux[i,:] is the WEST
    face of cell i (and the EAST face of cell i-1); x_flux[0,:] and
    x_flux[-1,:] are the domain boundary faces. Same convention as
    w4_2d_unsteady.py. Task 1b reuses these face values directly for its
    upwind convection scheme.
    """
    NX, NY = h.shape
    x_flux = np.zeros((NX + 1, NY))
    y_flux = np.zeros((NX, NY + 1))

    x_flux[1:-1, :] = -K * (h[1:, :] - h[:-1, :]) / dx
    x_flux[0, :] = -K * (h[0, :] - H_BC) / (dx / 2.)
    x_flux[-1, :] = -K * (H_BC - h[-1, :]) / (dx / 2.)

    y_flux[:, 1:-1] = -K * (h[:, 1:] - h[:, :-1]) / dy
    y_flux[:, 0] = -K * (h[:, 0] - H_BC) / (dy / 2.)
    y_flux[:, -1] = -K * (H_BC - h[:, -1]) / (dy / 2.)

    return x_flux, y_flux


def darcy_flux(h, dx, dy):
    """Face-based Darcy flux, averaged onto cell centres for plotting."""
    x_flux, y_flux = face_darcy_flux(h, dx, dy)
    qx = 0.5 * (x_flux[:-1, :] + x_flux[1:, :])
    qy = 0.5 * (y_flux[:, :-1] + y_flux[:, 1:])
    qmag = np.sqrt(qx**2 + qy**2)
    return qx, qy, qmag


# ==========================================================================
# Plot stamping (date + git hash, per submission requirements)
# ==========================================================================
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


def pick_resolution(target):
    """Smallest NX >= target for which no well sits on a cell face."""
    n = target
    while True:
        dx = LX / n
        dy = LY / n
        ok = True
        for name, wx, wy, Q in WELLS:
            fi, fj = wx / dx, wy / dy
            if abs(fi - round(fi)) < 1e-9 or abs(fj - round(fj)) < 1e-9:
                ok = False
                break
        if ok:
            return n
        n += 1


if __name__ == "__main__":
    OUT = "outputs"

    # ----------------------------------------------------------------
    # GRID CONVERGENCE STUDY
    # ----------------------------------------------------------------
    targets = [20, 40, 80, 120, 160, 200, 240]
    resolutions = [pick_resolution(t) for t in targets]

    runs = []
    for NX in resolutions:
        x, y, dx, dy, h, t_solve, well_idx = solve_case(NX, NX)
        qx, qy, qmag = darcy_flux(h, dx, dy)
        runs.append(dict(NX=NX, x=x, y=y, dx=dx, dy=dy, h=h,
                          qx=qx, qy=qy, qmag=qmag, t=t_solve, well_idx=well_idx))
        print(f"NX={NX:4d}  dx={dx:6.2f} m  t_solve={t_solve*1e3:7.2f} ms  "
              f"h=[{h.min():9.2f}, {h.max():9.2f}] m  "
              f"|q|=[{qmag.min():.3e}, {qmag.max():.3e}] m/s")

    # profile line at x = 500 m, all resolutions
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for r in runs:
        icol = np.argmin(np.abs(r["x"] - 500.0))
        ax.plot(r["h"][icol, :], r["y"], label=f"NX={r['NX']} (dx={r['dx']:.1f} m)")
    ax.set_xlabel("Hydraulic head, h [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Task 1a: vertical head profile at x = 500 m, all resolutions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(f"{OUT}/1a_convergence_profile_x500.png", dpi=200)
    plt.close(fig)

    # convergence metric away from the well singularities (see write-up):
    # a bulk monitor point, and domain min/max excluding a fixed PHYSICAL
    # radius around each well (a fixed number of cells would shrink with
    # dx and let the singularity back in as the grid is refined).
    MONITOR = (650.0, 610.0)
    R_EXCLUDE = 40.0

    h_monitor, h_away = [], []
    for r in runs:
        x, y, h = r["x"], r["y"], r["h"]
        im = np.argmin(np.abs(x - MONITOR[0]))
        jm = np.argmin(np.abs(y - MONITOR[1]))
        h_monitor.append(h[im, jm])

        X, Y = np.meshgrid(x, y, indexing="ij")
        mask = np.ones_like(h, dtype=bool)
        for name, wx, wy, Q in WELLS:
            mask &= ((X - wx)**2 + (Y - wy)**2) > R_EXCLUDE**2
        h_away.append((h[mask].min(), h[mask].max()))

    dh_monitor = [np.nan] + [abs(h_monitor[k] - h_monitor[k-1]) for k in range(1, len(h_monitor))]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    dxs = [r["dx"] for r in runs]
    axs[0].semilogy(dxs[1:], dh_monitor[1:], "o-")
    axs[0].set_xlabel("Grid spacing, dx [m]")
    axs[0].set_ylabel("|change in h at bulk\nmonitor point| [m]")
    axs[0].set_title(f"Convergence at bulk point {MONITOR}\n(away from well singularities)")
    axs[0].grid(alpha=0.3)
    axs[0].invert_xaxis()

    t_solve = [r["t"] * 1e3 for r in runs]
    axs[1].plot(dxs, t_solve, "s-", color="tab:red")
    axs[1].set_xlabel("Grid spacing, dx [m]")
    axs[1].set_ylabel("Solve time [ms]")
    axs[1].set_title("Compute cost vs resolution")
    axs[1].grid(alpha=0.3)
    axs[1].invert_xaxis()
    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(f"{OUT}/1a_convergence_metrics.png", dpi=200)
    plt.close(fig)

    print("\nNX, dx, monitor-point change vs previous, solve time, h_min/max away from wells:")
    for r, dh, tt, (hmn, hmx) in zip(runs, dh_monitor, t_solve, h_away):
        dh_str = "   -   " if np.isnan(dh) else f"{dh:7.3f}"
        print(f"  NX={r['NX']:4d}  dx={r['dx']:6.2f} m  dH_mon={dh_str} m  "
              f"t={tt:7.2f} ms  h(away)=[{hmn:7.2f}, {hmx:7.2f}] m")

    print("\nWell-cell head values by resolution (these do NOT grid-converge - "
          "see write-up note on the 2D point-source singularity):")
    for r in runs:
        vals = ", ".join(f"{name}={r['h'][i, j]:.1f}" for name, i, j, Q in r["well_idx"])
        print(f"  NX={r['NX']:4d}: {vals}")

    # ----------------------------------------------------------------
    # SELECTED RESOLUTION: contour plots of head and Darcy flux
    # ----------------------------------------------------------------
    NX_SEL = pick_resolution(160)     # see write-up for justification
    sel = [r for r in runs if r["NX"] == NX_SEL][0]
    h, qx, qy, qmag = sel["h"], sel["qx"], sel["qy"], sel["qmag"]
    x, y = sel["x"], sel["y"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cs = ax.contourf(x, y, h.T, levels=40, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="Hydraulic head, h [m]")
    for name, i, j, Q in sel["well_idx"]:
        ax.plot(x[i], y[j], "r^" if Q > 0 else "wv", ms=7, mec="k")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Task 1a: hydraulic head, NX={NX_SEL} (dx={sel['dx']:.2f} m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(f"{OUT}/1a_head_contour.png", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for a, field, title, cmap in zip(
            axs, (qx, qy, qmag),
            ("Darcy flux $q_x$ [m/s]", "Darcy flux $q_y$ [m/s]", "|q| [m/s]"),
            ("RdBu_r", "RdBu_r", "magma")):
        kw = dict(levels=40, cmap=cmap)
        if field is not qmag:
            vmax = np.max(np.abs(field))
            kw.update(vmin=-vmax, vmax=vmax)
        cs = a.contourf(x, y, field.T, **kw)
        fig.colorbar(cs, ax=a)
        a.set_title(title)
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
        a.set_aspect("equal")
    fig.suptitle(f"Task 1a: Darcy flux components, NX={NX_SEL} (dx={sel['dx']:.2f} m)")
    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(f"{OUT}/1a_flux_contours.png", dpi=200)
    plt.close(fig)

    print(f"\nSelected resolution for reporting: NX={NX_SEL} (dx={sel['dx']:.2f} m)")
    print("Figures written to ./outputs/")
