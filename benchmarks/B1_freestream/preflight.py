#!/usr/bin/env python3
"""B1 -- pre-flight checks without the solver: XML, mesh, displacement, General BC files, IC.

Usage: python3 preflight.py --case <case_dir> --xml <rendered case.xml>
"""
import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import preflight_lib as pf
from params import LX, LY, LZ, CENTER, AMP, PERIOD, U_INF, FACES, DIRICHLET_FACES, scale_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--xml", help="rendered XML to check (optional)")
    a = ap.parse_args()
    case = os.path.abspath(a.case)
    if a.xml:
        pf.check_xml(a.xml, FACES)
    g, fd = pf.check_mesh(os.path.join(case, "mesh"), FACES)
    if g is None:
        return pf.summary()
    gids_all = np.unique(np.concatenate([fd[nm][0] for nm in FACES]))
    P = np.asarray(g.points)
    c = np.asarray(CENTER)
    pf.check_disp_file(os.path.join(case, "displacement_allfaces.txt"), g, gids_all, PERIOD,
                       pos_fn_mm=lambda t: c + scale_fn(t) * (P[gids_all - 1] - c))
    for nm in DIRICHLET_FACES:
        pf.check_genbc_file(os.path.join(case, f"genbc_{nm}.dat"), fd[nm][0])
    import pyvista as pv
    icp = os.path.join(case, "ic_uniform.vtu")
    if pf.check(os.path.exists(icp), f"exists: {icp}"):
        ic = pv.read(icp)
        pf.check(ic.n_points == g.n_points, "IC: same number of points as the mesh")
        pf.check(np.linalg.norm(np.asarray(ic.points) - P) < 1e-9, "IC: same point order")
        v = np.asarray(ic.point_data["Velocity"])
        pf.check(np.abs(v - np.array([U_INF, 0, 0])).max() < 1e-12, f"IC: uniform Velocity ({U_INF},0,0)")
        pf.check(np.abs(np.asarray(ic.point_data["Pressure"])).max() < 1e-12, "IC: Pressure = 0")
        pf.check(np.abs(np.asarray(ic.point_data["Displacement"])).max() < 1e-12, "IC: Displacement = 0")
    wmax = 0.5 * AMP * (2 * np.pi / PERIOD) * np.linalg.norm([LX / 2, LY / 2, LZ / 2]) * pf.MM2CM
    pf.check(U_INF > 5 * wmax, f"U={U_INF} >> max mesh velocity {wmax:.3f} cm/s (x1 never an inflow)")
    return pf.summary()


if __name__ == "__main__":
    sys.exit(main())
