# MECH6480 Assignment 1 - Code

Python code for Assignment 1 (Part I: groundwater flow FVM, Part II: incompressible
Navier-Stokes lid-driven cavity). Each task is a standalone script; running it
regenerates the figures used in the submitted PDF into that task's `outputs/` folder.

## Requirements

Python 3, with `numpy`, `scipy`, and `matplotlib` installed.

## Part I - groundwater flow / in-situ leaching (`task1/`)

Run from inside the `task1/` folder, in this order (1b and 1c import the steady
flow field from 1a, so 1a does not need to be run first separately - each script
calls it automatically on import):

```
cd task1
python3 task1a_steady_aquifer.py     # steady-state head + Darcy flux, grid convergence
python3 task1b_lixiviant_transport.py  # transient lixiviant transport
python3 task1c_copper_leaching.py      # coupled lixiviant + copper (reactive) transport
```

Figures are written to `task1/outputs/`. `task1a_discretisation.pdf` (with its
`.tex` source) contains the worked FVM discretisation for Task 1a.

## Part II - lid-driven cavity flow (`task2/`)

Run from inside the `task2/` folder:

```
cd task2
python3 task2a_lid_cavity.py    # empty cavity, Re=1000 and Re=2500
python3 task2b_porous_cavity.py # saturated porous (Brinkman) cavity, Re=1000
```

`task2b_porous_cavity.py` must be run after `task2a_lid_cavity.py`, since it
loads the Task 2a Re=1000 result (`outputs/2a_fields_re1000.npz`) to plot the
empty-cavity comparison profile. `cavity_solver.py` holds the shared projection-
method solver (staggered/MAC grid) used by both scripts; it is not run directly.

Figures are written to `task2/outputs/`. Reference benchmark data (Ghia-style
profiles used to validate the solver) is in `task2/refdata/`.

## Notes

- `task2a_lid_cavity.py` runs a short grid-resolution sensitivity sweep before
  the full Re=1000/2500 runs; the full script takes several minutes to run
  (mostly the resolution sweep and the two steady-state solves).
- All figures are stamped with the generation date and the current Git commit
  hash, per the submission requirements.
