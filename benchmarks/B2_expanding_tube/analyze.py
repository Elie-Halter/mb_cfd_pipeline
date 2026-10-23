#!/usr/bin/env python3
"""
B2 -- expanding-tube analysis: moving-wall no-slip, induced flow rate, discrete mass balance,
and amplitude/phase of the wall velocity.

Usage: python3 analyze.py <run_dir> --case <case_dir> [--dt 0.005] [--label patched|nogcl]

For every snapshot results_###.vtu (output points are already deformed, mm -> cm):
  - Q_z0, Q_z1 = flux of u through the caps, Q_wall = flux through the wall (P1 exact,
    outward normals);
  - analytic dV/dt = 2 pi R L dR/dt with R(t) = R0 s(t).
Verdicts on the established regime t >= 0.5 s (the fluid starts from rest):
  (a) no-slip: max_wall |u - w| against the wall velocity actually imposed (slope of the
      piecewise-linear displacement file), on the converged second cycle   -> < 1e-4 cm/s
  (b) induced flow: max|Q_z0 + Q_z1 + dV/dt| / max|dV/dt|                    -> < 2 %
  (c) discrete mass balance through the WHOLE closed boundary:
      max|Q_z0 + Q_z1 + Q_wall| / max|dV/dt|                                 -> < 2 %
      (the discriminating DGCL observable: 3.3e-6 patched vs 6.66e-2 without patch 0014)
  (d) amplitude and phase of the mean radial wall velocity vs R0 ds/dt over [1.0, 1.5] s
      (fundamental fitted by least squares)                                  -> +-5 %, < 5 deg
Reference values (patched solver, 4 ranks, dt = 5 ms): (a) 2.68e-6 cm/s, (b) 1.64 %,
(c) 3.3e-6, (d) 1.000 / +0.6 deg. Without patch 0014: (b) 7.36 %, (c) 6.66e-2, (a) and (d)
unchanged.
"""
import argparse
import csv
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import benchtools as bt
from params import (R0, LT, PERIOD, DT, N_SAMPLES, SETTLE, NOSLIP_FROM, PASS_NOSLIP_ABS,
                    PASS_FLOW_REL, PASS_MASS_REL, scale_fn, scale_dot)

R0_CM, LT_CM = R0 * bt.MM2CM, LT * bt.MM2CM
OMEGA = 2 * np.pi / PERIOD


def load_face(case, name):
    import pyvista as pv
    s = pv.read(os.path.join(case, "mesh", "mesh-surfaces", f"{name}.vtp"))
    return np.asarray(s.point_data["GlobalNodeID"], dtype=np.int64), s.faces.reshape(-1, 4)[:, 1:]


def wall_flux(pts, u, gid_w, tri_w):
    """Flux through the wall, each triangle oriented along its outward RADIAL direction."""
    tri = np.take(gid_w - 1, tri_w)
    p0, p1, p2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    an = 0.5 * np.cross(p1 - p0, p2 - p0)
    rad = (p0 + p1 + p2) / 3.0
    rad[:, 2] = 0.0
    s = np.sign(np.einsum("ij,ij->i", an, rad)); s[s == 0] = 1.0
    um = (u[tri[:, 0]] + u[tri[:, 1]] + u[tri[:, 2]]) / 3.0
    return float(np.einsum("ij,ij->", um, an * s[:, None]))


def series(run_dir, case, dt):
    """Per-snapshot quantities of one run (shared with temporal_order.py)."""
    import pyvista as pv
    files = bt.find_results(run_dir)
    if not files:
        raise SystemExit(f"[B2] no results_*.vtu under {run_dir}")
    gid_z0, tri_z0 = load_face(case, "z0")
    gid_z1, tri_z1 = load_face(case, "z1")
    gid_w, tri_w = load_face(case, "wall")
    Xref = np.asarray(pv.read(os.path.join(case, "mesh", "mesh-complete.mesh.vtu")).points) * bt.MM2CM
    ts_file = np.linspace(0.0, PERIOD, N_SAMPLES + 1)
    rows = []
    for fp in files:
        g = pv.read(fp)
        step = bt.step_of(fp)
        t = step * dt
        u = np.asarray(g.point_data["Velocity"])[:, :3]
        pts = bt.result_points_cm(g)
        q0 = bt.tri_flux(pts, np.take(gid_z0 - 1, tri_z0), u, (0, 0, -1))
        q1 = bt.tri_flux(pts, np.take(gid_z1 - 1, tri_z1), u, (0, 0, 1))
        qw = wall_flux(pts, u, gid_w, tri_w)
        s, sd = float(scale_fn(t)), float(scale_dot(t))
        dVa = 2 * np.pi * (R0_CM * s) * LT_CM * (R0_CM * sd)
        # no-slip against the wall velocity actually imposed (slope of the linear file samples)
        tm = t % PERIOD
        k = min(int(np.floor(tm / (PERIOD / N_SAMPLES) - 1e-9)), N_SAMPLES - 1)
        sd_disc = float(scale_fn(ts_file[k + 1]) - scale_fn(ts_file[k])) / (ts_file[k + 1] - ts_file[k])
        uw = u[gid_w - 1]
        vw = np.zeros_like(uw)
        vw[:, 0] = sd_disc * Xref[gid_w - 1, 0]
        vw[:, 1] = sd_disc * Xref[gid_w - 1, 1]
        err_d = float(np.linalg.norm(uw - vw, axis=1).max())
        # mean radial velocity at the wall
        pw = pts[gid_w - 1]
        r = np.hypot(pw[:, 0], pw[:, 1])
        ur_wall = float(np.mean((uw[:, 0] * pw[:, 0] + uw[:, 1] * pw[:, 1]) / r))
        rows.append(dict(step=step, t_s=t, Q_z0=q0, Q_z1=q1, Q_caps=q0 + q1, Q_wall=qw,
                         minus_dVa_dt=-dVa, noslip_err=err_d, ur_wall=ur_wall))
        print(f"  step {step:4d} t={t:5.3f}  sumQ_caps={q0 + q1:+8.4f} Q_wall={qw:+8.4f} "
              f"vs -dV/dt={-dVa:+8.4f} | closed sum={q0 + q1 + qw:+.2e} | no-slip {err_d:.1e} cm/s")
    return rows


def verdicts(rows):
    tt = np.array([r["t_s"] for r in rows])
    Qc = np.array([r["Q_caps"] for r in rows])
    Qw = np.array([r["Q_wall"] for r in rows])
    mdV = np.array([r["minus_dVa_dt"] for r in rows])
    scale = np.abs(mdV).max()
    est = tt >= SETTLE
    conv = tt >= NOSLIP_FROM
    res = dict(
        noslip=float(np.max([r["noslip_err"] for r in rows if r["t_s"] >= NOSLIP_FROM])) if conv.any() else np.nan,
        flow=float(np.abs(Qc - mdV)[est].max() / scale),
        mass=float(np.abs(Qc + Qw)[est].max() / scale),
        mass_rms=float(np.sqrt(np.mean((Qc + Qw)[est] ** 2)) / scale),
    )
    win = (tt >= 1.0 - 1e-9) & (tt <= 1.5 + 1e-9)
    if win.sum() >= 4:
        As, ps = bt.fit_sincos(tt[win], np.array([r["ur_wall"] for r in rows])[win], OMEGA)
        Aa, pa = bt.fit_sincos(tt[win], R0_CM * scale_dot(tt[win]), OMEGA)
        res["ur_amp"] = As / Aa
        res["ur_phase_deg"] = float(np.degrees(((pa - ps) + np.pi) % (2 * np.pi) - np.pi))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--case", required=True, help="case directory written by gen_case.py")
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--label", default="patched")
    ap.add_argument("--csv", help="output CSV (default <run_dir>/b2_metrics_<label>.csv)")
    a = ap.parse_args()
    rows = series(a.run_dir, os.path.abspath(a.case), a.dt)
    out = a.csv or os.path.join(a.run_dir, f"b2_metrics_{a.label}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    v = verdicts(rows)
    ok_a = v["noslip"] < PASS_NOSLIP_ABS
    ok_b = v["flow"] < PASS_FLOW_REL
    ok_c = v["mass"] < PASS_MASS_REL
    print(f"\n[B2] {a.label} -> {out}  (verdicts on t >= {SETTLE} s)")
    print(f"[B2] (a) moving-wall no-slip (t >= {NOSLIP_FROM} s): {v['noslip']:.3e} cm/s "
          f"({'PASS' if ok_a else 'FAIL'} < {PASS_NOSLIP_ABS:g}; reference 2.68e-6)")
    print(f"[B2] (b) induced cap flow vs analytic: {v['flow']:.3%} "
          f"({'PASS' if ok_b else 'FAIL'} < {PASS_FLOW_REL:.0%}; reference 1.64 % patched, 7.36 % without 0014)")
    print(f"[B2] (c) closed-boundary mass balance / max|dV/dt|: {v['mass']:.3e} "
          f"({'PASS' if ok_c else 'FAIL'} < {PASS_MASS_REL:.0%}; reference 3.3e-6 patched, 6.66e-2 without 0014)")
    ok = ok_a and ok_b and ok_c
    if "ur_amp" in v:
        ok_d = abs(v["ur_amp"] - 1) <= 0.05 and abs(v["ur_phase_deg"]) <= 5.0
        print(f"[B2] (d) wall radial velocity: amplitude {v['ur_amp']:.4f}, phase {v['ur_phase_deg']:+.2f} deg "
              f"({'PASS' if ok_d else 'FAIL'} +-5 %, < 5 deg; reference 1.000 / +0.6 deg)")
        ok = ok and ok_d
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
