#!/usr/bin/env python3
"""B3 -- pre-flight checks without the solver: analytic self-test, XML, mesh, IC, pressure file.

Usage: python3 preflight.py --case <case_dir> --grid g1 [--xml <rendered case.xml>]
"""
import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import preflight_lib as pf
from params import R0, R0_CM, LT_CM, RHO, NU, OMEGA, P_AMP, PERIOD, ALPHA, GRIDS
from womersley import Womersley, selftest

FACES = ["wall", "z0", "z1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--grid", default="g1", choices=sorted(GRIDS))
    ap.add_argument("--xml")
    a = ap.parse_args()
    case = os.path.abspath(a.case)
    selftest(verbose=False)
    pf.check(True, "analytic self-test (no-slip, ODE residual, Poiseuille limit, flow integral)")
    pf.check(10 < ALPHA < 15, f"alpha = {ALPHA:.2f} in the aortic range 10-15")
    if a.xml:
        pf.check_xml(a.xml, FACES)
    g, fd = pf.check_mesh(os.path.join(case, f"mesh_{a.grid}"), FACES)
    if g is None:
        return pf.summary()
    P = np.asarray(g.points)
    gw = fd["wall"][0]
    r_wall = np.hypot(P[gw - 1, 0], P[gw - 1, 1])
    pf.check(abs(r_wall.mean() - R0) < 0.02 * R0 and r_wall.std() < 0.02 * R0,
             f"wall nodes on r = R0 ({r_wall.mean():.3f} +- {r_wall.std():.3f} mm)")
    delta_mm = np.sqrt(2 * NU / OMEGA) * 10.0
    r_all = np.hypot(P[:, 0], P[:, 1])
    n_in = int(((r_all > R0 - delta_mm) & (r_all < R0 - 1e-6)).sum())
    print(f"    {n_in} interior nodes inside the Stokes layer delta = {delta_mm:.2f} mm")
    import pyvista as pv
    icp = os.path.join(case, f"ic_{a.grid}.vtu")
    if pf.check(os.path.exists(icp), f"exists: {icp}"):
        ic = pv.read(icp)
        pf.check(ic.n_points == g.n_points, "IC: same number of points")
        pf.check(np.linalg.norm(np.asarray(ic.points) - P) < 1e-9, "IC: same point order")
        w = Womersley(R0_CM, LT_CM, RHO, NU, OMEGA, P_AMP)
        v = np.asarray(ic.point_data["Velocity"])
        want = w.u(np.minimum(r_all * 0.1, R0_CM), 0.0)
        pf.check(np.abs(v[:, 2] - want).max() < 1e-8, "IC: u_z == Womersley(t=0)")
        pf.check(np.abs(v[:, :2]).max() < 1e-12, "IC: u_x = u_y = 0")
        pf.check(np.abs(v[gw - 1, 2]).max() < 1e-6 * np.abs(want).max(), "IC: no-slip at the wall")
        pf.check(np.abs(np.asarray(ic.point_data["Pressure"])).max() < 1e-12, "IC: p = 0")
    ppath = os.path.join(case, "pressure_inlet.dat")
    if pf.check(os.path.exists(ppath), f"exists: {ppath}"):
        toks = open(ppath).read().split()
        npts, nmodes = int(toks[0]), int(toks[1])
        arr = np.asarray(toks[2:], dtype=float).reshape(npts, 2)
        pf.check(nmodes <= npts // 2, f"Fourier modes {nmodes} <= npts/2")
        pf.check(abs(arr[0, 0]) < 1e-12 and abs(arr[-1, 0] - PERIOD) < 1e-9, "pressure: t spans [0, T]")
        pf.check(np.abs(arr[:, 1] - P_AMP * np.sin(OMEGA * arr[:, 0])).max() < 1e-6 * P_AMP,
                 "pressure: exactly A sin(wt)")
    return pf.summary()


if __name__ == "__main__":
    sys.exit(main())
