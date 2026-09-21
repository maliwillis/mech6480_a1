import time
import numpy as np
import matplotlib.pyplot as plt

from task1a_steady_aquifer import (
    WELLS, QI, QE, K,
    solve_case, face_darcy_flux, pick_resolution, stamp_figure,
)

# PARAMS
PHI = 0.25
D = 1.0e-8
B = 5.0
C_INJ = 5.0
C_BC = 0.0

SECONDS_PER_DAY = 86400.
T_END = 100. * SECONDS_PER_DAY
SAFETY = 0.9

# New reaction parameters 
Y = 0.05            # kg Cu dissolved per kg lixiviant reacted
KR = 1.0e-7         # /s, leaching reaction rate constant


# GRID + STEADY FLOW FIELD 
# Task 1a, same resolution, held fixed

NX = pick_resolution(160)
NY = NX
x, y, dx, dy, h, t_solve, well_idx = solve_case(NX, NY)
x_flux, y_flux = face_darcy_flux(h, dx, dy)

inj_name, inj_i, inj_j, _ = [w for w in well_idx if w[3] > 0][0]
ext_wells = [w for w in well_idx if w[3] < 0]

# STABILITY
De_face = np.full((NX + 1, NY), PHI * D * dy / dx)
De_face[0, :] *= 2.; De_face[-1, :] *= 2.
Dn_face = np.full((NX, NY + 1), PHI * D * dx / dy)
Dn_face[:, 0] *= 2.; Dn_face[:, -1] *= 2.

Fe_face = x_flux * dy
Fn_face = y_flux * dx

aE = De_face[1:, :] + np.maximum(-Fe_face[1:, :], 0.)
aW = De_face[:-1, :] + np.maximum(Fe_face[:-1, :], 0.)
aN = Dn_face[:, 1:] + np.maximum(-Fn_face[:, 1:], 0.)
aS = Dn_face[:, :-1] + np.maximum(Fn_face[:, :-1], 0.)
sum_nb = aE + aW + aN + aS

sp_extra = np.zeros((NX, NY))
for name, i, j, Q in ext_wells:
    sp_extra[i, j] += QE / B         
sp_extra += KR * dx * dy      

dt_bound = PHI * dx * dy / (sum_nb + sp_extra)
dt = SAFETY * dt_bound.min()
nsteps = int(np.ceil(T_END / dt))
dt = T_END / nsteps

print(f"Grid: NX=NY={NX}, dx=dy={dx:.3f} m")
print(f"Explicit stability limit dt_max = {dt_bound.min():.1f} s")
print(f"Using dt = {dt:.1f} s -> {nsteps} steps to reach {T_END/SECONDS_PER_DAY:.0f} days")


def flux_divergence(Cfield):

    Cx = np.zeros((NX + 1, NY))
    Cy = np.zeros((NX, NY + 1))

    C_up = np.where(x_flux[1:-1, :] >= 0, Cfield[:-1, :], Cfield[1:, :])
    Cx[1:-1, :] = x_flux[1:-1, :] * C_up - PHI * D * (Cfield[1:, :] - Cfield[:-1, :]) / dx
    C_up_w = np.where(x_flux[0, :] >= 0, C_BC, Cfield[0, :])
    Cx[0, :] = x_flux[0, :] * C_up_w - PHI * D * (Cfield[0, :] - C_BC) / (dx / 2.)
    C_up_e = np.where(x_flux[-1, :] >= 0, Cfield[-1, :], C_BC)
    Cx[-1, :] = x_flux[-1, :] * C_up_e - PHI * D * (C_BC - Cfield[-1, :]) / (dx / 2.)

    C_up = np.where(y_flux[:, 1:-1] >= 0, Cfield[:, :-1], Cfield[:, 1:])
    Cy[:, 1:-1] = y_flux[:, 1:-1] * C_up - PHI * D * (Cfield[:, 1:] - Cfield[:, :-1]) / dy
    C_up_s = np.where(y_flux[:, 0] >= 0, C_BC, Cfield[:, 0])
    Cy[:, 0] = y_flux[:, 0] * C_up_s - PHI * D * (Cfield[:, 0] - C_BC) / (dy / 2.)
    C_up_n = np.where(y_flux[:, -1] >= 0, Cfield[:, -1], C_BC)
    Cy[:, -1] = y_flux[:, -1] * C_up_n - PHI * D * (C_BC - Cfield[:, -1]) / (dy / 2.)

    return dy * (Cx[1:, :] - Cx[:-1, :]) + dx * (Cy[:, 1:] - Cy[:, :-1])


# WELL SOURCE TERMS

R_inj = QI * C_INJ / (B * dx * dy)


# TIME MARCHING

C = np.zeros((NX, NY))       # lixiviant concentration
CCu = np.zeros((NX, NY))     # copper concentration

snapshot_days = [0, 20, 40, 60, 80, 100]
snap_C, snap_Cu = {}, {}
time_history = []
cu_history = {name: [] for name, i, j, Q in ext_wells}

tic = time.time()
sim_time = 0.0
for step in range(nsteps):
    Rleach = KR * C                      # first-order in lixiviant, everywhere

    C_new = C - (dt / (PHI * dx * dy)) * flux_divergence(C) - (dt / PHI) * Rleach
    CCu_new = CCu - (dt / (PHI * dx * dy)) * flux_divergence(CCu) + (dt / PHI) * Y * Rleach

    # WELL SOURCE TERMS
    C_new[inj_i, inj_j] += (dt / PHI) * R_inj
    # (no injector term for copper - R_well,Cu = 0 there)
    for name, i, j, Q in ext_wells:
        C_new[i, j] += (dt / PHI) * (-QE * C[i, j] / (B * dx * dy))
        CCu_new[i, j] += (dt / PHI) * (-QE * CCu[i, j] / (B * dx * dy))

    C, CCu = C_new, CCu_new
    sim_time += dt

    for name, i, j, Q in ext_wells:
        cu_history[name].append(CCu[i, j])
    time_history.append(sim_time)

    day = sim_time / SECONDS_PER_DAY
    for sd in snapshot_days:
        if sd not in snap_C and day >= sd:
            snap_C[sd] = C.copy()
            snap_Cu[sd] = CCu.copy()

snap_C[100] = C.copy()
snap_Cu[100] = CCu.copy()
toc = time.time()

print(f"\nSimulation complete: {nsteps} steps in {toc - tic:.2f} s")
print("\nConcentration at each extraction well after 100 days:")
for name, i, j, Q in ext_wells:
    print(f"  {name}: C = {C[i, j]:.4f} kg/m3   C_Cu = {CCu[i, j]:.5f} kg/m3")

# PLOTS
OUT = "outputs"
time_history = np.array(time_history) / SECONDS_PER_DAY

# --- time-series contour plots: lixiviant (top row) and copper (bottom) --
fig, axs = plt.subplots(2, 6, figsize=(22, 7.5))
for col, sd in enumerate(snapshot_days):
    cs1 = axs[0, col].contourf(x, y, snap_C[sd].T, levels=np.linspace(0, C_INJ, 41),
                                cmap="viridis", extend="max")
    axs[0, col].set_title(f"t = {sd} d")
    cu_max = max(snap_Cu[100].max(), 1e-6)
    cs2 = axs[1, col].contourf(x, y, snap_Cu[sd].T, levels=np.linspace(0, cu_max, 41),
                                cmap="plasma", extend="max")
    for a in (axs[0, col], axs[1, col]):
        for name, i, j, Q in well_idx:
            a.plot(x[i], y[j], "r^" if Q > 0 else "wv", ms=5, mec="k")
        a.set_aspect("equal")
        a.set_xlabel("x [m]")
axs[0, 0].set_ylabel("Lixiviant\ny [m]")
axs[1, 0].set_ylabel("Copper\ny [m]")
fig.colorbar(cs1, ax=axs[0, :], shrink=0.85, label="C [kg/m$^3$]")
fig.colorbar(cs2, ax=axs[1, :], shrink=0.85, label="C$_{Cu}$ [kg/m$^3$]")
fig.suptitle(f"Task 1c: lixiviant and copper concentration evolution, "
             f"NX={NX} (dx={dx:.2f} m)")
stamp_figure(fig)
fig.savefig(f"{OUT}/1c_concentration_timeseries.png", dpi=200)
plt.close(fig)

# --- history plot: copper concentration at each extraction well ---------
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for name, i, j, Q in ext_wells:
    ax.plot(time_history, cu_history[name], label=name)
ax.set_xlabel("Time [days]")
ax.set_ylabel("Copper concentration, C$_{Cu}$ [kg/m$^3$]")
ax.set_title("Task 1c: copper concentration at each extraction well over time")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
stamp_figure(fig)
fig.savefig(f"{OUT}/1c_copper_history.png", dpi=200)
plt.close(fig)

print(f"\nFigures written to ./{OUT}/")
