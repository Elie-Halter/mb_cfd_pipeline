#!/usr/bin/env python3
"""
B3r -- Womersley profile error under mesh refinement (grids g1 < g2 < g3).

Usage: python3 refinement.py LABEL:RUN_DIR [LABEL:RUN_DIR ...]   (coarse to fine)

Error = area-weighted L2 of u_z on the mid-plane slab, restricted to the flow CORE r < 0.8 R,
over cycle 2 (t in (1, 2] s, converged and common to all grids). h_eff = N_nodes^(-1/3).
Prints the pairwise observed order p = log(E_i/E_{i+1}) / log(h_i/h_{i+1}) and the log-log
regression over all grids.

Pass criterion used in the paper: the error DECREASES monotonically under refinement.
Reference values (patched solver, 4 ranks): 28.2 % -> 15.7 % -> 11.3 % on g1/g2/g3 (N = 9322 /
27634 / 88289 nodes), pairwise p = 1.61 then 0.86, global p = 1.22. A formal order ~2 is NOT
obtained: the Stokes layer (delta ~ 1.1 mm) is thinner than the isotropic element on all three
grids, so the spatial order is reported as open (mesh-limited), not claimed.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import benchtools as bt
from params import R0_CM, LT_CM, RHO, NU, OMEGA, P_AMP, DT, SLAB_HALF_CM, CORE_RMAX, REFINE_WINDOW
from womersley import Womersley


def core_error(run_dir, dt=DT):
    import pyvista as pv
    files = bt.find_results(run_dir)
    if not files:
        return None
    w = Womersley(R0_CM, LT_CM, RHO, NU, OMEGA, P_AMP)
    lo, hi = REFINE_WINDOW
    num = den = 0.0
    nsel, N = 0, None
    for fp in files:
        t = bt.step_of(fp) * dt
        if not (lo < t <= hi + 1e-9):
            continue
        g = pv.read(fp)
        N = g.n_points
        P = bt.result_points_cm(g)
        uz_all = np.asarray(g.point_data["Velocity"])[:, 2]
        m = np.abs(P[:, 2] - LT_CM / 2) < SLAB_HALF_CM
        r = np.hypot(P[m, 0], P[m, 1])
        sel = r < CORE_RMAX * R0_CM
        ua = w.u(r[sel], t)
        num += float(np.sum(r[sel] * (uz_all[m][sel] - ua) ** 2))
        den += float(np.sum(r[sel] * ua ** 2))
        nsel += 1
    if den == 0.0:
        return None
    return dict(E=np.sqrt(num / den), N=N, nsel=nsel)


def main(specs):
    if not specs:
        raise SystemExit(__doc__)
    res = []
    for spec in specs:
        label, rd = spec.split(":", 1)
        r = core_error(rd)
        if r is None:
            print(f"  {label:4s}: no results in the window {REFINE_WINDOW} under {rd}")
            continue
        r.update(label=label, h=r["N"] ** (-1.0 / 3.0))
        res.append(r)
        print(f"  {label:4s}: N={r['N']:6d} nodes  h_eff={r['h']:.4e}  core L2={r['E']:.2%}  ({r['nsel']} snapshots)")
    if len(res) < 2:
        return 1
    for a, b in zip(res[:-1], res[1:]):
        p = np.log(a["E"] / b["E"]) / np.log(a["h"] / b["h"])
        print(f"  {a['label']} -> {b['label']}: p = {p:.2f}  (E {a['E']:.2%} -> {b['E']:.2%})")
    p_glob = np.polyfit(np.log([r["h"] for r in res]), np.log([r["E"] for r in res]), 1)[0]
    mono = all(b["E"] < a["E"] for a, b in zip(res[:-1], res[1:]))
    print(f"\n[B3r] global observed order (log-log, {len(res)} grids): p = {p_glob:.2f} "
          f"(reference 1.22; not claimed, mesh-limited)")
    print(f"[B3r] {'PASS' if mono else 'FAIL'}: core L2 error {'decreases' if mono else 'does NOT decrease'} "
          f"monotonically under refinement")
    return 0 if mono else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
