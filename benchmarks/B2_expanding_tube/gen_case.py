#!/usr/bin/env python3
"""
B2 -- analytically expanding tube: case generator (no solver run here).

Geometry : z-axis cylinder, R0 = 10 mm, L = 50 mm (1 x 5 cm after scale 0.1), tets ~1.5 mm.
Motion   : uniform radial dilation R(t) = R0 s(t), s(t) = 1 + (eps/2)(1 - cos 2 pi t/T),
           eps = 0.10 -> R: 10 -> 11 mm (s^2 - 1 ~ 21 % volume swing), zero initial wall
           velocity. Wall AND caps prescribed ((x,y) -> s (x,y), z fixed): the caps stay planar
           and dilate in their plane, so the deformed domain is an exact cylinder and
           V(t) = pi R(t)^2 L analytically.
Fluid    : moving no-slip on the wall (Prescribed_displacement +
           Impose_on_state_variable_integral, the production code path); z0/z1 Neumann p = 0
           (fluid enters through the caps during expansion). Starts from rest.
Analytic : Q_z0(t) + Q_z1(t) = -dV/dt = -2 pi R L dR/dt; u_r(R,t) = dR/dt.

Usage:   python3 gen_case.py --out <case_dir>
Outputs: <case_dir>/mesh/, displacement_boundary.txt
"""
import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import benchtools as bt
from params import R0, LT, H_MESH, EPS, PERIOD, N_SAMPLES, FACES, scale_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="case directory to create")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    mesh_dir = os.path.join(out, "mesh")
    g = bt.gen_cylinder_mesh(R0, LT, h=H_MESH)
    face_ids = {nm: i + 1 for i, nm in enumerate(FACES)}
    info = bt.write_svmp_mesh(g, mesh_dir, lambda c, n: bt.classify_cylinder(c, n, R0, LT), face_ids)

    import pyvista as pv
    X = np.asarray(pv.read(os.path.join(mesh_dir, "mesh-complete.mesh.vtu")).points)
    gids_all = np.unique(np.concatenate([info[k] for k in FACES]))
    Xb = X[gids_all - 1]

    def pos_mm(t):
        s = scale_fn(t)
        return np.column_stack([Xb[:, 0] * s, Xb[:, 1] * s, Xb[:, 2]])

    times = np.linspace(0.0, PERIOD, N_SAMPLES + 1)
    bt.write_disp_file(os.path.join(out, "displacement_boundary.txt"), gids_all, pos_mm, times)

    dRdt_max = 0.5 * EPS * (2 * np.pi / PERIOD) * (R0 * bt.MM2CM)
    Qmax = 2 * np.pi * (R0 * bt.MM2CM) * (LT * bt.MM2CM) * dRdt_max
    print(f"[B2] R: {R0} -> {R0 * (1 + EPS)} mm; max volume swing = {(1 + EPS)**2 - 1:.1%}")
    print(f"[B2] max dR/dt = {dRdt_max:.4f} cm/s; max |Q_total| ~ {Qmax:.3f} cm3/s")
    print(f"[B2] case ready in {out}")


if __name__ == "__main__":
    main()
