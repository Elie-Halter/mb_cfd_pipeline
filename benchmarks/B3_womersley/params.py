"""B3 -- shared parameters. Units: mm for the mesh; CGS in the solver (cm, g, s, dyn)."""
import numpy as np

R0 = 10.0          # radius (mm) -> 1 cm
LT = 60.0          # length (mm) -> 6 cm; analysis on the mid-plane z = L/2
RHO = 1.06         # g/cm3
MU = 0.04          # poise
NU = MU / RHO      # 0.03774 cm2/s
PERIOD = 1.0       # s
OMEGA = 2.0 * np.pi / PERIOD
R0_CM = R0 * 0.1
LT_CM = LT * 0.1
ALPHA = R0_CM * np.sqrt(OMEGA / NU)      # ~12.9 (aortic range 10-15)

P_AMP = 1333.2     # dyn/cm2 (~1 mmHg): p_in(t) = P_AMP sin(wt), p_out = 0
DT = 0.005         # s (200 steps per period)
N_CYCLES = 3       # analytic IC; amplitude/phase on the LAST cycle, periodicity vs the previous
VTK_INC = 5        # 40 snapshots per cycle
SLAB_HALF_CM = 0.3 # half-thickness of the analysis slab around z = L/2

# Radially graded meshes, h(r) = H_CORE - (H_CORE - H_WALL) (r/R)^H_POW (mm). The Stokes layer
# is delta = sqrt(2 nu / omega) ~ 1.1 mm. Three quasi self-similar grids (h ratio ~1.44):
H_POW = 3.0
GRIDS = {            # name: (H_WALL, H_CORE)       ~tets  (reference run)
    "g1": (0.90, 2.2),   # ~38k
    "g2": (0.55, 1.6),   # ~122k
    "g3": (0.35, 1.0),   # ~431k
}

# Pass criteria (flow rate through the mid-plane, fundamental fitted by least squares)
PASS_AMP_REL = 0.05
PASS_PHASE_DEG = 5.0
# Field accuracy: the area-weighted L2 error of the core profile (r < CORE_RMAX R, cycle 2) must
# DECREASE monotonically under refinement. It does not reach a few percent on isotropic grids:
# the Stokes layer is thinner than the isotropic element on all three grids.
CORE_RMAX = 0.8
REFINE_WINDOW = (1.0, 2.0)   # cycle 2, common to all grids
