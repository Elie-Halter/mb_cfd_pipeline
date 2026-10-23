#!/usr/bin/env python3
"""B2 -- pre-flight checks without the solver: XML, mesh, displacement file, analytic geometry.

Usage: python3 preflight.py --case <case_dir> [--xml <rendered case.xml>]
"""
import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import preflight_lib as pf
from params import R0, LT, PERIOD, DT, N_SAMPLES, FACES, scale_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--xml")
    a = ap.parse_args()
    case = os.path.abspath(a.case)
    if a.xml:
        pf.check_xml(a.xml, FACES)
    g, fd = pf.check_mesh(os.path.join(case, "mesh"), FACES)
    if g is None:
        return pf.summary()
    P = np.asarray(g.points)
    gw = fd["wall"][0]
    r_wall = np.hypot(P[gw - 1, 0], P[gw - 1, 1])
    pf.check(abs(r_wall.mean() - R0) < 0.02 * R0 and r_wall.std() < 0.02 * R0,
             f"wall nodes on r = R0 ({r_wall.mean():.3f} +- {r_wall.std():.3f} mm)")
    for nm, zval in (("z0", 0.0), ("z1", LT)):
        pf.check(np.abs(P[fd[nm][0] - 1, 2] - zval).max() < 1e-6, f"cap {nm} on the plane z = {zval} mm")
    gids_all = np.unique(np.concatenate([fd[nm][0] for nm in FACES]))

    def pos_fn(t):
        Q = P[gids_all - 1].copy()
        Q[:, :2] *= scale_fn(t)
        return Q

    pf.check_disp_file(os.path.join(case, "displacement_boundary.txt"), g, gids_all, PERIOD, pos_fn_mm=pos_fn)
    pf.check(abs(PERIOD / N_SAMPLES - DT) < 1e-12, f"sample spacing ({PERIOD / N_SAMPLES}) == nominal dt ({DT})")
    # the wall/cap rings must be in the file for both faces (requires patch 0015)
    s0 = set(fd["wall"][0]) & set(fd["z0"][0])
    s1 = set(fd["wall"][0]) & set(fd["z1"][0])
    pf.check(len(s0) > 0 and len(s1) > 0, f"shared rings wall/z0 = {len(s0)}, wall/z1 = {len(s1)} nodes")
    return pf.summary()


if __name__ == "__main__":
    sys.exit(main())
