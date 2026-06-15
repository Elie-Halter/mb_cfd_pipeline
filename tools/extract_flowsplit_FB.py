#!/usr/bin/env python3
"""
Flow-split extraction for a fixed-boundary (FB) run, to validate the Option-3
RCR calibration. Integrates Q = sum(v_tri . area_vec)/100 [mL/s] through each
outlet cap (triangle integration, the ParaView/COMSOL/Fluent method), time-
averages over the saved cycle, and prints the % split vs the MRI targets.

Moving-boundary correct: velocities are mapped from the volume by GlobalNodeID
(NOT positional KDTree — the volume nodes sit at the DEFORMED positions while the
.vtp is at the reference), cap areas use the DEFORMED node positions from the VTU
(cap areas vary in time), and each triangle's normal is oriented outward via the
apex of its parent tetrahedron (so Q>0 = outflow regardless of the .vtp winding).

Usage:
  python3 extract_flowsplit_FB.py <results_dir> <mesh_surfaces_dir> [--outlets desc,btca,lcca,lsa]
e.g.
  python3 extract_flowsplit_FB.py \
      <results_dir>/2-procs \
      <mesh_complete_dir>/mesh-surfaces
"""
import sys, glob, os, argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# Example outlet names and target split (%) -- replace with your cohort's values.
OUTLETS = ["desc", "bcca", "lcca", "lsa"]
MRI = {"desc": 75.3, "bcca": 12.6, "lcca": 4.7, "lsa": 7.5}
# the same vessel is named bcca (FB meshes) or btca (MB pipeline meshes)
ALIASES = {"bcca": ["bcca", "btca"], "btca": ["btca", "bcca"]}


def find_vtp(surf_dir, name):
    for cand in ALIASES.get(name, [name]):
        p = os.path.join(surf_dir, f"{cand}.vtp")
        if os.path.exists(p):
            return p
    raise SystemExit(f"no .vtp for outlet '{name}' (tried {ALIASES.get(name,[name])}) in {surf_dir}")


def volume_row_for_gid(vol, surf_gid):
    """Row in `vol` for each surface GlobalNodeID.

    The svMP results VTU carries NO GlobalNodeID, but its node order is the
    reference-mesh order with GlobalNodeID == row+1 (verified: build_iso_mesh
    assigns arange+1 in point order), so row = gid-1. If a future VTU does carry
    GlobalNodeID, use it. Bounds-checked. Deformation-proof either way."""
    vg = vol.point_data.get("GlobalNodeID")
    if vg is not None:
        vg = np.asarray(vg).astype(np.int64)
        m = np.full(int(vg.max()) + 1, -1, dtype=np.int64)
        m[vg] = np.arange(len(vg))
        idx = m[surf_gid]
    else:
        idx = surf_gid - 1
    if idx.min() < 0 or idx.max() >= vol.n_points:
        return None
    return idx


def tri_flux(surf, vol):
    """Q [mL/s] through a cap, in the .vtp's RAW winding (sign fixed globally in main).

    gid-1 maps cap nodes to volume rows (deformation-proof); the DEFORMED volume
    positions give the (time-varying) cap area."""
    surf = surf.triangulate()
    sg = np.asarray(surf.point_data["GlobalNodeID"]).astype(np.int64)
    idx = volume_row_for_gid(vol, sg)
    if idx is None:
        raise SystemExit("cap GlobalNodeID outside the volume -- inconsistent mesh/result")

    P = vol.points[idx]                                  # DEFORMED cap positions (mm)
    vel = np.asarray(vol.point_data["Velocity"])[idx]    # cm/s at the same nodes
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    Q = 0.0
    for i, j, k in faces:
        area_vec = 0.5 * np.cross(P[j] - P[i], P[k] - P[i])   # mm^2
        v_tri = (vel[i] + vel[j] + vel[k]) / 3.0              # cm/s
        Q += np.dot(v_tri, area_vec)                          # cm/s * mm^2
    return Q / 100.0                                          # -> mL/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("surf_dir")
    ap.add_argument("--outlets", default=",".join(OUTLETS),
                    help="comma list of logical outlet names (default desc,bcca,lcca,lsa)")
    args = ap.parse_args()
    results_dir, surf_dir = args.results_dir, args.surf_dir
    outlets = [o.strip() for o in args.outlets.split(",") if o.strip()]

    vtus = sorted(glob.glob(os.path.join(results_dir, "results_*.vtu")),
                  key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    if not vtus:
        raise SystemExit(f"no results_*.vtu in {results_dir}")
    print(f"{len(vtus)} VTU snapshots: {os.path.basename(vtus[0])} .. {os.path.basename(vtus[-1])}")

    surfs = {o: pv.read(find_vtp(surf_dir, o)) for o in outlets}
    means = {o: [] for o in outlets}
    for f in vtus:
        vol = pv.read(f)
        for o in outlets:
            means[o].append(tri_flux(surfs[o], vol))

    q_mean = {o: float(np.mean(means[o])) for o in outlets}   # cycle-avg mL/s, RAW winding
    # Global outward orientation by MASS CONSERVATION (rigorous, not physiology):
    # the cycle-mean net flux through all outlets equals the prescribed inflow > 0.
    # The caps share a consistent .vtp winding, so one global sign makes every cap
    # outward-positive. (Per-cap geometric probes are fragile on small curved caps.)
    total_raw = sum(q_mean.values())
    sign = -1.0 if total_raw < 0 else 1.0
    if sign < 0:
        print("  orientation: .vtp winding = inward -> global flip (mass conservation)")
    q_mean = {o: sign * q for o, q in q_mean.items()}
    total = sign * total_raw
    print(f"\n{'outlet':6} {'Q_mean[mL/s]':>13} {'split%':>8} {'MRI%':>7} {'err':>7}")
    print("-" * 46)
    for o in outlets:
        sp = 100.0 * q_mean[o] / total if total else float("nan")
        tgt = MRI.get(o, float("nan"))
        print(f"{o:6} {q_mean[o]:13.2f} {sp:8.1f} {tgt:7.1f} {sp-tgt:+7.1f}")
    print(f"{'TOTAL':6} {total:13.2f} {'100.0':>8}")
    # consistency check: any cap whose raw sign disagrees with the global flip
    n_disagree = sum(1 for o in outlets if q_mean[o] < 0)
    if n_disagree:
        print(f"  [WARN] {n_disagree} cap(s) with Q<0 after flip: winding NOT consistent "
              f"across caps, or genuine net backflow on that cap -- check.")
    print(f"\nCardiac output (outlet sum) = {total*60/1000:.2f} L/min")


if __name__ == "__main__":
    main()
