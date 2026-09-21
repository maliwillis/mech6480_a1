import time
import numpy as np
import matplotlib.pyplot as plt

from cavity_solver import (
    solve_cavity, cell_centre_velocity, stamp_figure, L, H, U_LID,
)
from task2a_lid_cavity import load_ref_u, load_ref_v

REFDATA = "refdata"
OUT = "outputs"

# ==========================================================================
# Brinkman-modified momentum equation (Eq. 8):
#   du/dt + div(uu) = -1/rho grad(p) + nu*lap(u) - (nu/k)*u
#
# The extra term is a LINEAR DRAG proportional to local velocity, applied
# during the velocity-prediction step in cavity_solver.solve_cavity()
# (before the pressure projection - it is just another explicit source
# term alongside the convective/diffusive flux divergence):
#
#     if brinkman_k is not None:
#         ut[1:-1, 1:-1] -= dt * (nu / brinkman_k) * u[1:-1, 1:-1]
#         vt[1:-1, 1:-1] -= dt * (nu / brinkman_k) * v[1:-1, 1:-1]
#
# No other part of the projection method changes: u* still gets corrected
# by the same pressure-Poisson / velocity-correction steps.
# ==========================================================================
K_PERM = 5.0e-5   # m^2, permeability
RE = 1000
NX_SEL = 61        # same resolution justified/selected in Task 2a


def run_case():
    dx = L / NX_SEL
    dt = 0.2 * dx
    print(f"=== Task 2b: Re={RE}, k={K_PERM:.1e} m^2, NX=NY={NX_SEL} (dx={dx:.5f} m), "
          f"dt={dt:.3e} s ===")
    t0 = time.time()
    res = solve_cavity(RE, NX_SEL, NX_SEL, dt, 120000, brinkman_k=K_PERM,
                        check_every=1000, tol_steady=1e-5, verbose=True)
    print(f"  wall time: {time.time() - t0:.1f} s")
    np.savez(f"{OUT}/2b_fields_re{RE}.npz",
              u=res["u"], v=res["v"], p=res["p"], x=res["x"], y=res["y"],
              sim_time=res["sim_time"], steps=res["steps"], NX=NX_SEL)
    return res


def make_comparison_figure(res_porous):
    # load the empty-cavity (Task 2a) result at the same Re for comparison
    data_2a = np.load(f"{OUT}/2a_fields_re{RE}.npz")
    x2a, y2a = data_2a["x"], data_2a["y"]
    u2a, v2a = data_2a["u"], data_2a["v"]

    uc_p, vc_p = cell_centre_velocity(res_porous)
    uc_a = 0.5 * (u2a[:-1, 1:-1] + u2a[1:, 1:-1])
    vc_a = 0.5 * (v2a[1:-1, :-1] + v2a[1:-1, 1:])

    icol = np.argmin(np.abs(res_porous["x"] - L / 2))
    jrow = np.argmin(np.abs(res_porous["y"] - H / 2))
    icol_a = np.argmin(np.abs(x2a - L / 2))
    jrow_a = np.argmin(np.abs(y2a - H / 2))

    y_ref, u_ref = load_ref_u(RE)
    x_ref, v_ref = load_ref_v(RE)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    axs[0].plot(uc_p[icol, :] / U_LID, res_porous["y"] / H, "b-", lw=2,
                label="Porous cavity (this solution)")
    axs[0].plot(uc_a[icol_a, :] / U_LID, y2a / H, "g--", lw=1.5,
                label="Empty cavity (Task 2a)")
    axs[0].plot(u_ref, y_ref, "ko", ms=4, mfc="none", label="Reference data (empty cavity)")
    axs[0].set_xlabel("u / U$_{lid}$"); axs[0].set_ylabel("y / H")
    axs[0].set_title(f"Horizontal velocity along x = L/2 (Re = {RE})")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

    axs[1].plot(res_porous["x"] / L, vc_p[:, jrow] / U_LID, "b-", lw=2,
                label="Porous cavity (this solution)")
    axs[1].plot(x2a / L, vc_a[:, jrow_a] / U_LID, "g--", lw=1.5,
                label="Empty cavity (Task 2a)")
    axs[1].plot(x_ref, v_ref, "ko", ms=4, mfc="none", label="Reference data (empty cavity)")
    axs[1].set_xlabel("x / L"); axs[1].set_ylabel("v / U$_{lid}$")
    axs[1].set_title(f"Vertical velocity along y = H/2 (Re = {RE})")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)

    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(f"{OUT}/2b_profiles_re{RE}.png", dpi=200)
    plt.close(fig)


def make_contour_figure(res_porous):
    uc, vc = cell_centre_velocity(res_porous)
    speed = np.sqrt(uc**2 + vc**2)
    X, Y = np.meshgrid(res_porous["x"], res_porous["y"], indexing="ij")

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    cs = ax.contourf(X, Y, speed, levels=40, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="Velocity magnitude [m/s]")
    ax.streamplot(res_porous["x"], res_porous["y"], uc.T, vc.T, color="white",
                  density=1.3, linewidth=0.7, arrowsize=0.8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Task 2b: velocity magnitude and streamlines, saturated porous "
                 f"cavity\nRe = {RE}, k = {K_PERM:.1e} m$^2$, steady at "
                 f"t = {res_porous['sim_time']:.2f} s")
    ax.set_aspect("equal")
    fig.tight_layout()
    stamp_figure(fig)
    fig.savefig(f"{OUT}/2b_streamlines_re{RE}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    res = run_case()
    make_comparison_figure(res)
    make_contour_figure(res)
    print("Figures written to ./outputs/")
