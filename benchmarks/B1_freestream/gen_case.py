#!/usr/bin/env python3
"""
B1 -- free-stream preservation on a moving mesh: case generator (no solver run here).

Geometry : box [0,20]x[-5,5]x[-5,5] mm (= [0,2]x[-0.5,0.5]^2 cm after scale 0.1).
Motion   : affine dilation about c = (10,0,0) mm, s(t) = 1 + (A/2)(1 - cos 2 pi t/T),
           A = 0.08, T = 1 s (s: 1 -> 1.08 -> 1, ~26 % volume swing, zero initial mesh
           velocity). Prescribed on ALL six faces through the mesh equation; the interior
           follows the mesh equation -- the same code path as a production wall-only run.
Field    : u = (U,0,0), U = 10 cm/s, p = const -- an EXACT solution of the ALE Navier-Stokes
           equations for ANY mesh motion. Dirichlet General (full vector,
           Zero_out_perimeter false) on x0,y0,y1,z0,z1; resistance outlet on x1 (always an
           outflow since U >> max mesh velocity ~0.31 cm/s). Exact uniform IC (VTU).

Usage:   python3 gen_case.py --out <case_dir>
Outputs: <case_dir>/mesh/, displacement_allfaces.txt, genbc_<face>.dat, ic_uniform.vtu
"""
import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import benchtools as bt
from params import LX, LY, LZ, H_MESH, CENTER, AMP, PERIOD, U_INF, N_SAMPLES, FACES, \
    DIRICHLET_FACES, scale_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="case directory to create")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    mesh_dir = os.path.join(out, "mesh")
    g = bt.gen_box_mesh(0.0, -LY / 2, -LZ / 2, LX, LY, LZ, H_MESH)
    bounds = (0.0, LX, -LY / 2, LY / 2, -LZ / 2, LZ / 2)
    face_ids = {nm: i + 1 for i, nm in enumerate(FACES)}
    info = bt.write_svmp_mesh(g, mesh_dir, lambda c, n: bt.classify_box(c, n, bounds), face_ids)

    import pyvista as pv
    gm = pv.read(os.path.join(mesh_dir, "mesh-complete.mesh.vtu"))
    X = np.asarray(gm.points)                                  # mm, order = GID-1
    gids_all = np.unique(np.concatenate([info[k] for k in FACES]))
    Xb = X[gids_all - 1]
    c = np.asarray(CENTER)
    times = np.linspace(0.0, PERIOD, N_SAMPLES + 1)
    bt.write_disp_file(os.path.join(out, "displacement_allfaces.txt"), gids_all,
                       lambda t: c + scale_fn(t) * (Xb - c), times)

    for nm in DIRICHLET_FACES:                               # one file per face (exact nNo)
        bt.write_temp_spat_file(os.path.join(out, f"genbc_{nm}.dat"), info[nm], (U_INF, 0.0, 0.0))

    N = gm.n_points
    bt.write_ic_vtu(os.path.join(mesh_dir, "mesh-complete.mesh.vtu"), os.path.join(out, "ic_uniform.vtu"),
                    velocity=np.tile([U_INF, 0.0, 0.0], (N, 1)), pressure=np.zeros(N),
                    displacement=np.zeros((N, 3)))

    smax = scale_fn(PERIOD / 2)
    wmax = 0.5 * AMP * (2 * np.pi / PERIOD) * np.linalg.norm(np.abs(Xb - c).max(axis=0)) * bt.MM2CM
    print(f"[B1] max volume swing = {smax**3 - 1:.3%}")
    print(f"[B1] max mesh velocity ~{wmax:.3f} cm/s vs U = {U_INF} cm/s (x1 always an outflow)")
    print(f"[B1] case ready in {out}")


if __name__ == "__main__":
    main()
