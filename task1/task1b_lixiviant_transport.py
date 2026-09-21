import time
import numpy as np
import matplotlib.pyplot as plt

from task1a_steady_aquifer import (
    WELLS, QI, QE, K,
    solve_case, face_darcy_flux, pick_resolution, stamp_figure,
)


# PARAMS
PHI = 0.25            # porosity
D = 1.0e-8            # m2/s, lixiviant diffusion coefficient
B = 5.0               # m, aquifer thickness
C_INJ = 5.0           # kg/m3, injected lixiviant concentration
C_BC = 0.0            # kg/m3, far-field concentration (Dirichlet)

SECONDS_PER_DAY = 86400.
T_END = 100. * SECONDS_PER_DAY   # 100 days

SAFETY = 0.9            # fraction of the explicit stability limit to use


# GRID + STEADY FLOW FIELD (From Task 1a -> NX = 161
NX = pick_resolution(160)
NY = NX
x, y, dx, dy, h, t_solve, well_idx = solve_case(NX, NY)
x_flux, y_flux = face_darcy_flux(h, dx, dy)     # steady Darcy flux at faces

print(f"Grid: NX=NY={NX}, dx=dy={dx:.3f} m  (reusing Task 1a resolution)")
print(f"Darcy flux magnitude range: "
      f"[{np.sqrt(0.5*(x_flux[:-1]**2+x_flux[1:]**2)).min():.2e}, "
      f"{np.max(np.abs(x_flux)):.2e}] m/s (x-faces, similar for y)")


# STABILITY: choose dt from the explicit boundedness criterion
De_face = np.full((NX + 1, NY), PHI * D * dy / dx)
De_face[0, :] *= 2.; De_face[-1, :] *= 2.        # boundary: half distance
Dn_face = np.full((NX, NY + 1), PHI * D * dx / dy)
Dn_face[:, 0] *= 2.; Dn_face[:, -1] *= 2.

Fe_face = x_flux * dy      # convective face coefficient (m^2/s), "F" in Patankar notation
Fn_face = y_flux * dx

# Per-cell neighbour coefficients for an upwind convection + central
aE = De_face[1:, :] + np.maximum(-Fe_face[1:, :], 0.)
aW = De_face[:-1, :] + np.maximum(Fe_face[:-1, :], 0.)
aN = Dn_face[:, 1:] + np.maximum(-Fn_face[:, 1:], 0.)
aS = Dn_face[:, :-1] + np.maximum(Fn_face[:, :-1], 0.)

sum_nb = aE + aW + aN + aS
sp_extra = np.zeros((NX, NY))
for name, i, j, Q in [w for w in well_idx if w[3] < 0]:
    sp_extra[i, j] = QE / B

dt_bound = PHI * dx * dy / (sum_nb + sp_extra)
dt = SAFETY * dt_bound.min()
nsteps = int(np.ceil(T_END / dt))
dt = T_END / nsteps          # land exactly on T_END

print(f"Explicit stability limit dt_max = {dt_bound.min():.1f} s "
      f"(tightest cell, safety factor {SAFETY})")
print(f"Using dt = {dt:.1f} s -> {nsteps} steps to reach {T_END/SECONDS_PER_DAY:.0f} days")

# WELL SOURCE TERMS
inj_name, inj_i, inj_j, _ = [w for w in well_idx if w[3] > 0][0]
ext_wells = [w for w in well_idx if w[3] < 0]     # (name, i, j, Q)

R_inj = QI * C_INJ / (B * dx * dy)     # constant, added every step

# TIME MARCHING (explicit, upwind convection + central diffusion)
C = np.zeros((NX, NY))          # initial condition: C = 0 everywhere

snapshot_days = [0, 20, 40, 60, 80, 100]
snapshots = {}
well_history = {name: [] for name, i, j, Q in ext_wells}
time_history = []

tic = time.time()
sim_time = 0.0
for step in range(nsteps):
    # CONVECTIVE + DIFFUSIVE FACE FLUX 
    # combined, upwind for convection, central difference for diffusion
    Cx_flux = np.zeros((NX + 1, NY))
    Cy_flux = np.zeros((NX, NY + 1))

    # internal x-faces
    C_up = np.where(x_flux[1:-1, :] >= 0, C[:-1, :], C[1:, :])   # upwind value
    Cx_flux[1:-1, :] = x_flux[1:-1, :] * C_up - PHI * D * (C[1:, :] - C[:-1, :]) / dx
    # boundary x-faces (far-field C_BC = 0)
    C_up_w = np.where(x_flux[0, :] >= 0, C_BC, C[0, :])
    Cx_flux[0, :] = x_flux[0, :] * C_up_w - PHI * D * (C[0, :] - C_BC) / (dx / 2.)
    C_up_e = np.where(x_flux[-1, :] >= 0, C[-1, :], C_BC)
    Cx_flux[-1, :] = x_flux[-1, :] * C_up_e - PHI * D * (C_BC - C[-1, :]) / (dx / 2.)

    # internal y-faces
    C_up = np.where(y_flux[:, 1:-1] >= 0, C[:, :-1], C[:, 1:])
    Cy_flux[:, 1:-1] = y_flux[:, 1:-1] * C_up - PHI * D * (C[:, 1:] - C[:, :-1]) / dy
    # boundary y-faces
    C_up_s = np.where(y_flux[:, 0] >= 0, C_BC, C[:, 0])
    Cy_flux[:, 0] = y_flux[:, 0] * C_up_s - PHI * D * (C[:, 0] - C_BC) / (dy / 2.)
    C_up_n = np.where(y_flux[:, -1] >= 0, C[:, -1], C_BC)
    Cy_flux[:, -1] = y_flux[:, -1] * C_up_n - PHI * D * (C_BC - C[:, -1]) / (dy / 2.)

    # UPDATE C:  PHI*dC/dt = -div(total flux) + R_well
    C_new = C - (dt / (PHI * dx * dy)) * (
        dy * (Cx_flux[1:, :] - Cx_flux[:-1, :]) +
        dx * (Cy_flux[:, 1:] - Cy_flux[:, :-1])
    )

    # WELL SOURCE TERMS (added after the flux update, per-cell)
    C_new[inj_i, inj_j] += (dt / PHI) * R_inj
    for name, i, j, Q in ext_wells:
        R_ext = -QE * C[i, j] / (B * dx * dy)     # uses C at START of step
        C_new[i, j] += (dt / PHI) * R_ext

    C = C_new
    sim_time += dt

    for name, i, j, Q in ext_wells:
        well_history[name].append(C[i, j])
    time_history.append(sim_time)

    day = sim_time / SECONDS_PER_DAY
    for sd in snapshot_days:
        if sd not in snapshots and day >= sd:
            snapshots[sd] = C.copy()

snapshots[100] = C.copy()   # guarantee the final state is captured
toc = time.time()

print(f"\nSimulation complete: {nsteps} steps in {toc - tic:.2f} s "
      f"({(toc - tic) / nsteps * 1e3:.3f} ms/step)")
print("\nLixiviant concentration at each extraction well after 100 days:")
for name, i, j, Q in ext_wells:
    print(f"  {name}: C = {C[i, j]:.4f} kg/m3")

# PLOTS
OUT = "outputs"
time_history = np.array(time_history) / SECONDS_PER_DAY

fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
vmax = C_INJ
for ax, sd in zip(axs.flat, snapshot_days):
    field = snapshots[sd]
    cs = ax.contourf(x, y, field.T, levels=np.linspace(0, vmax, 41),
                      cmap="viridis", extend="max")
    for name, i, j, Q in well_idx:
        ax.plot(x[i], y[j], "r^" if Q > 0 else "wv", ms=6, mec="k")
    ax.set_title(f"t = {sd} days")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
fig.colorbar(cs, ax=axs, shrink=0.85, label="Lixiviant concentration, C [kg/m$^3$]")
fig.suptitle(f"Task 1b: lixiviant concentration evolution, NX={NX} (dx={dx:.2f} m)")
stamp_figure(fig)
fig.savefig(f"{OUT}/1b_concentration_timeseries.png", dpi=200)
plt.close(fig)

print(f"\nFigures written to ./{OUT}/")
