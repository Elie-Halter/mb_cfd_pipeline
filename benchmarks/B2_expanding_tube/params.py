"""B2 -- shared parameters. Units: mm, s; solver in CGS after scale 0.1."""
import numpy as np

R0 = 10.0        # reference radius (mm) -> 1 cm
LT = 50.0        # length (mm) -> 5 cm
H_MESH = 1.5     # tet size (mm), ~42 segments around the circumference: the polygonal volume
                 # deficit (~0.4 %) stays well below the 2 % flow-rate criterion
EPS = 0.10       # total radial strain: s(t) = 1 + (EPS/2)(1 - cos 2 pi t/T) -> R: R0 -> 1.1 R0
PERIOD = 1.0     # s
DT = 0.005       # s, nominal time step; displacement samples are aligned on it
N_STEPS = 300    # 1.5 periods at the nominal dt
N_SAMPLES = 200  # 201 displacement samples over [0, T] (sample spacing = nominal dt)
SNAPSHOTS = 60   # saved VTK snapshots per run, whatever the dt
RHO = 1.06       # g/cm3
MU = 0.04        # poise
FACES = ["wall", "z0", "z1"]

SETTLE = 0.5     # s: verdicts on t >= SETTLE (the fluid starts from rest)
NOSLIP_FROM = 1.05  # s: moving-wall no-slip judged on the converged second cycle

# Pass criteria
PASS_NOSLIP_ABS = 1e-4      # cm/s, fluid velocity at the wall vs prescribed wall velocity
PASS_FLOW_REL = 0.02        # induced cap flow vs -dV/dt (analytic)
PASS_MASS_REL = 0.02        # closed-boundary flux sum vs max|dV/dt|


def scale_fn(t):
    return 1.0 + 0.5 * EPS * (1.0 - np.cos(2.0 * np.pi * np.asarray(t) / PERIOD))


def scale_dot(t):
    return 0.5 * EPS * (2.0 * np.pi / PERIOD) * np.sin(2.0 * np.pi * np.asarray(t) / PERIOD)


def variant(dt_div):
    """Run settings for dt = DT / dt_div (temporal-order study): same physical time and the
    same number of snapshots."""
    n = N_STEPS * dt_div
    return dict(DT=DT / dt_div, NSTEPS=n, VTK_INC=n // SNAPSHOTS, RESTART_INC=n // 3)
