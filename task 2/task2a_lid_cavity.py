import time
import numpy as np
import matplotlib.pyplot as plt

from cavity_solver import (
    solve_cavity, cell_centre_velocity, nu_from_Re, stamp_figure,
    L, H, U_LID,
)

REFDATA = "refdata"
OUT = "outputs"

# ==========================================================================
# REFERENCE DATA (Ghia-style benchmark, provided with the assignment)
#   ux file columns: y, x05, x10, x50, x90, x95  (u at x = that fraction*L)
#   uy file columns: x, y05, y10, y50, y90, y99  (v at y = that fraction*H)
#   x, y columns are normalised 0-1; we want the x=0.5L / y=0.5H columns.
# ==========================================================================
def load_ref_u(Re):
    data = np.loadtxt(f"{REFDATA}/lid_cavity_data_ux_re{Re}.txt", skiprows=1)
    return data[:, 0], data[:, 3]     # y/H,  u at x=L/2


def load_ref_v(Re):
    data = np.loadtxt(f"{REFDATA}/lid_cavity_data_uy_re{Re}.txt", skiprows=1)
    return data[:, 0], data[:, 3]     # x/L,  v at y=H/2


# ==========================================================================
# GRID / TIMESTEP SENSITIVITY  (Re = 1000)
#   dt is set as a fixed fraction of dx (CFL-based); this was checked to sit
#   comfortably below both the convective (dt < dx/U_lid) and diffusive
#   (dt < dx^2/4nu) explicit stability limits for every case run here.
# ==========================================================================
def run_sensitivity():
    print("=== Grid sensitivity (Re = 1000) ===")
    resolutions = [21, 31, 41, 51, 61]
    runs = []
    for NX in resolutions:
        dx = L / NX
        dt = 0.2 * dx
        t0 = time.time()
        res = solve_cavity(1000, NX, NX, dt, 80000, check_every=1000,
                            tol_steady=1e-5, verbose=False)
        wall = time.time() - t0
        runs.append(dict(NX=NX, res=res, wall=wall))
        print(f"  NX={NX:3d}  dx={dx:.5f} m  dt={dt:.2e} s  steps={res['steps']:6d}  "
              f"t_steady={res['sim_time']:.2f} s  wall={wall:.1f} s")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for run in runs:
        res = run["res"]
        uc, _ = cell_centre_velocity(res)
        icol = np.argmin(np.abs(res["x"] - L / 2))
        ax.plot(uc[icol, :] / U_LID, res["y"] / H, label=f"NX={run['NX']}")
    ax.set_xlabel("u / U$_{lid}$")
    ax.set_ylabel("y / H")
    ax.set_title("Task 2a: grid sensitivity, u along x = L/2 (Re = 1000, steady state)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(f"{OUT}/2a_grid_sensitivity.png", dpi=200)
    plt.close(fig)

    print("\nmax|change in u-profile| between successive resolutions:")
    for k in range(1, len(runs)):
        a, b = runs[k - 1], runs[k]
        ua, _ = cell_centre_velocity(a["res"])
        ub, _ = cell_centre_velocity(b["res"])
        ia = np.argmin(np.abs(a["res"]["x"] - L / 2))
        ib = np.argmin(np.abs(b["res"]["x"] - L / 2))
        prof_a = np.interp(b["res"]["y"], a["res"]["y"], ua[ia, :])
        diff = np.max(np.abs(prof_a - ub[ib, :]))
        print(f"  NX={a['NX']:3d} -> NX={b['NX']:3d}: {diff:.4f} "
              f"({diff / U_LID * 100:.1f}% of U_lid)")

    return runs


# ==========================================================================
# FULL RUN AT SELECTED RESOLUTION, FOR ONE Re
# ==========================================================================
def run_case(Re, NX):
    dx = L / NX
    dt = 0.2 * dx
    print(f"\n=== Re = {Re}, NX=NY={NX} (dx={dx:.5f} m), dt={dt:.3e} s ===")
    t0 = time.time()
    res = solve_cavity(Re, NX, NX, dt, 120000, check_every=1000,
                        tol_steady=1e-5, verbose=True)
    wall = time.time() - t0
    print(f"  wall time: {wall:.1f} s")
    np.savez(f"{OUT}/2a_fields_re{Re}.npz",
             u=res["u"], v=res["v"], p=res["p"], x=res["x"], y=res["y"],
             sim_time=res["sim_time"], steps=res["steps"], NX=NX)
    return res


def make_profile_figure(res, Re):
    uc, vc = cell_centre_velocity(res)
    icol = np.argmin(np.abs(res["x"] - L / 2))
    jrow = np.argmin(np.abs(res["y"] - H / 2))
    y_ref, u_ref = load_ref_u(Re)
    x_ref, v_ref = load_ref_v(Re)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    axs[0].plot(uc[icol, :] / U_LID, res["y"] / H, "b-", lw=2, label="This solution")
    axs[0].plot(u_ref, y_ref, "ko", ms=4, mfc="none", label="Reference data")
    axs[0].set_xlabel("u / U$_{lid}$"); axs[0].set_ylabel("y / H")
    axs[0].set_title(f"Horizontal velocity along x = L/2 (Re = {Re})")
    axs[0].legend(); axs[0].grid(alpha=0.3)

    axs[1].plot(res["x"] / L, vc[:, jrow] / U_LID, "b-", lw=2, label="This solution")
    axs[1].plot(x_ref, v_ref, "ko", ms=4, mfc="none", label="Reference data")
    axs[1].set_xlabel("x / L"); axs[1].set_ylabel("v / U$_{lid}$")
    axs[1].set_title(f"Vertical velocity along y = H/2 (Re = {Re})")
    axs[1].legend(); axs[1].grid(alpha=0.3)

    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(f"{OUT}/2a_profiles_re{Re}.png", dpi=200)
    plt.close(fig)


def make_contour_figure(res, Re):
    uc, vc = cell_centre_velocity(res)
    speed = np.sqrt(uc**2 + vc**2)
    X, Y = np.meshgrid(res["x"], res["y"], indexing="ij")

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    cs = ax.contourf(X, Y, speed, levels=40, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="Velocity magnitude [m/s]")
    ax.streamplot(res["x"], res["y"], uc.T, vc.T, color="white",
                  density=1.3, linewidth=0.7, arrowsize=0.8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Task 2a: velocity magnitude and streamlines, Re = {Re}\n"
                 f"steady at t = {res['sim_time']:.2f} s")
    ax.set_aspect("equal")
    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(f"{OUT}/2a_streamlines_re{Re}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_sensitivity()

    NX_SEL = 61   # see write-up: <4% of U_lid change beyond this resolution

    for Re in (1000, 2500):
        res = run_case(Re, NX_SEL)
        make_profile_figure(res, Re)
        make_contour_figure(res, Re)

    print(f"\nSelected resolution for reporting: NX=NY={NX_SEL}")
    print("Figures written to ./outputs/")
