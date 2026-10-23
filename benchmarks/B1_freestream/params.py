"""B1 -- shared parameters (gen_case / preflight / analyze). Units: mm, s; solver in CGS."""
import numpy as np

LX, LY, LZ = 20.0, 10.0, 10.0      # box (mm): [0,20] x [-5,5] x [-5,5]
H_MESH = 1.2                       # target tet size (mm) -> 6,320 tets
CENTER = (10.0, 0.0, 0.0)          # dilation centre (mm)
AMP = 0.08                         # s(t) max = 1 + AMP -> ~26 % volume swing
PERIOD = 1.0                       # s
U_INF = 10.0                       # cm/s (solver units)
N_SAMPLES = 100                    # displacement samples per period (101 points)
DT = 0.005                         # s
N_STEPS = 200                      # one period
FACES = ["x0", "x1", "y0", "y1", "z0", "z1"]
DIRICHLET_FACES = ["x0", "y0", "y1", "z0", "z1"]   # x1 = outlet (resistance)

# Pass criterion: the iterative solver (FSILS) preserves the free stream to its tolerance
# floor (~1e-7 per step, ~1.7e-6 worst case over the period), not to machine precision.
PASS_MAX_REL_ERR = 1e-4


def scale_fn(t):
    """Affine dilation factor, s(0) = 1 with zero initial mesh velocity."""
    return 1.0 + 0.5 * AMP * (1.0 - np.cos(2.0 * np.pi * t / PERIOD))
