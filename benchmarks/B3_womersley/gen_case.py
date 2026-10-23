#!/usr/bin/env python3
"""
B3 -- Womersley flow in a rigid tube: case generator (no solver run here).

Rigid z-axis tube R0 = 10 mm, L = 60 mm; alpha ~ 12.9. Forcing p_in = A sin(wt) (unsteady
Neumann, Fourier file), p_out = 0, no-slip wall. The IC is the exact analytic solution at t = 0:
without it the start-up transient decays as exp(-nu lambda_1^2 t / R^2), tau ~ 4.6 s, i.e.
~14 cycles. Radially graded meshes (gmsh MathEval size field), three grids for refinement.

Usage:   python3 gen_case.py --out <case_dir> [--grid g1 g2 g3]
Outputs: <case_dir>/mesh_<grid>/, ic_<grid>.vtu, pressure_inlet.dat
"""
import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import benchtools as bt
from params import R0, LT, R0_CM, LT_CM, RHO, NU, OMEGA, P_AMP, PERIOD, H_POW, GRIDS
from womersley import Womersley


def build_grid(out, name):
    hwall, hcore = GRIDS[name]
    mesh_dir = os.path.join(out, f"mesh_{name}")
    expr = f"{hcore} - ({hcore}-{hwall})*((sqrt(x*x+y*y))/{R0})^{H_POW}"
    g = bt.gen_cylinder_mesh(R0, LT, size_expr=expr)
    bt.write_svmp_mesh(g, mesh_dir, lambda c, n: bt.classify_cylinder(c, n, R0, LT),
                       {"wall": 1, "z0": 2, "z1": 3})
    import pyvista as pv
    gm = pv.read(os.path.join(mesh_dir, "mesh-complete.mesh.vtu"))
    X = np.asarray(gm.points)
    r_cm = np.hypot(X[:, 0], X[:, 1]) * bt.MM2CM
    w = Womersley(R0_CM, LT_CM, RHO, NU, OMEGA, P_AMP)
    vel = np.zeros((gm.n_points, 3))
    vel[:, 2] = w.u(np.minimum(r_cm, R0_CM), 0.0)
    bt.write_ic_vtu(os.path.join(mesh_dir, "mesh-complete.mesh.vtu"), os.path.join(out, f"ic_{name}.vtu"),
                    velocity=vel, pressure=np.zeros(gm.n_points))      # p(z,0) = 0 since sin(0) = 0
    print(f"[B3] {name}: {gm.n_cells} tets, {gm.n_points} nodes; max |u_z(t=0)| = {np.abs(vel[:, 2]).max():.2f} cm/s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", nargs="+", default=["g1"], choices=sorted(GRIDS))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    for name in a.grid:
        build_grid(out, name)
    bt.write_fourier_pressure_file(os.path.join(out, "pressure_inlet.dat"), P_AMP, PERIOD, npts=401, nmodes=64)
    print(f"[B3] case ready in {out}")


if __name__ == "__main__":
    main()
