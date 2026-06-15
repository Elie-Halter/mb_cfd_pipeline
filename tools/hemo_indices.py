#!/usr/bin/env python3
"""
Hemodynamic indices over the last cycle.

- TAWSS = (1/T) integral_0^T |WSS| dt                         [dyne/cm^2]
- OSI   = 0.5 * (1 - |integral WSS dt| / integral |WSS| dt)   [-]
- Helicity: density h = u . omega (omega = curl u) and local normalised helicity
  LNH = (u . omega)/(|u||omega|) in [-1,1]                    [-] (flow chirality)

We aggregate over the last cycle's VTUs (same nodes in FB; in MB without remesh the
topology is preserved -> direct aggregation). Outputs: a wall VTP (TAWSS, OSI) + a
volume VTU (mean helicity), plus summary stats.

Usage (the cycle bounds below are an example -- one cycle = 974 steps):
  python3 hemo_indices.py <results_dir> --wall <wall.vtp> \
        --cycle-start 974 --cycle-end 1948 [--dt 0.001] [--out-prefix idx_]
"""
import sys, os, glob, argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def cycle_vtus(results_dir, s0, s1):
    out = []
    for f in glob.glob(os.path.join(results_dir, "results_*.vtu")):
        step = int("".join(filter(str.isdigit, os.path.basename(f))) or -1)
        if s0 <= step <= s1:
            out.append((step, f))
    return [f for _, f in sorted(out)]


def surface_rows_in_volume(surf, vol):
    """Row index in `vol` for each `surf` point — EXACT, deformation-proof.

    The svMP results VTU carries NO GlobalNodeID but its node order is the
    reference-mesh order with GlobalNodeID == row+1, so row = (surf gid)-1. This
    is required for moving boundary: the volume points are at the DEFORMED
    positions while the .vtp is at the reference, so a positional KDTree mismatches
    precisely where motion is large. KDTree is used only if the wall .vtp itself
    lacks GlobalNodeID (then the run must be fixed-boundary)."""
    sg = surf.point_data.get("GlobalNodeID")
    if sg is not None:
        idx = np.asarray(sg).astype(np.int64) - 1
        if idx.min() >= 0 and idx.max() < vol.n_points:
            return idx, "gid-1"
    tree = cKDTree(vol.points)
    d, idx = tree.query(surf.points, k=1)
    print(f"  [WARN] wall->volume mapping via KDTree (no GlobalNodeID on the wall) "
          f"-- dist med {np.median(d):.3g}: WRONG if the boundary moves.")
    return idx, "kdtree"


def wss_indices(vtus, wall, dt):
    """TAWSS & OSI on the wall nodes (WSS mapped from the volume by GlobalNodeID)."""
    wpts = wall.points
    sum_mag = np.zeros(len(wpts))      # integral |WSS| dt
    sum_vec = np.zeros((len(wpts), 3)) # integral WSS dt
    T = 0.0
    mode_seen = None
    n_used = n_skip = 0
    for f in vtus:
        m = pv.read(f)
        if "WSS" not in m.point_data:
            n_skip += 1
            continue
        idx, mode = surface_rows_in_volume(wall, m)
        if mode != mode_seen:
            print(f"  wall->volume mapping: {mode}")
            mode_seen = mode
        wss = np.asarray(m.point_data["WSS"])[idx]
        sum_mag += np.linalg.norm(wss, axis=1) * dt
        sum_vec += wss * dt
        T += dt
        n_used += 1
    if n_skip:
        print(f"  [WARN] {n_skip}/{len(vtus)} VTU without WSS field -- skipped (TAWSS/OSI on {n_used}).")
    if n_used == 0:
        raise SystemExit("no VTU with WSS field -- nothing to integrate")
    tawss = sum_mag / max(T, 1e-12)
    osi = 0.5 * (1.0 - np.linalg.norm(sum_vec, axis=1) / np.maximum(sum_mag, 1e-12))
    return tawss, osi


def helicity(vtus):
    """Cycle-averaged helicity (density u.omega and LNH), volume field."""
    base = pv.read(vtus[0])
    h_sum = np.zeros(base.n_points)
    lnh_sum = np.zeros(base.n_points)
    n = 0
    for f in vtus:
        m = pv.read(f)
        if "Velocity" not in m.point_data:
            continue
        d = m.compute_derivative(scalars="Velocity", vorticity=True)
        u = np.asarray(m.point_data["Velocity"])
        w = np.asarray(d.point_data["vorticity"])
        h = np.sum(u * w, axis=1)                       # helicity density
        denom = np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1)
        lnh = np.divide(h, denom, out=np.zeros_like(h), where=denom > 1e-9)
        h_sum += h; lnh_sum += lnh; n += 1
    if n < len(vtus):
        print(f"  [WARN] {len(vtus)-n}/{len(vtus)} VTU without Velocity field -- skipped (helicity on {n}).")
    if n == 0:
        raise SystemExit("no VTU with Velocity field")
    return base, h_sum / n, lnh_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--wall", required=True)
    ap.add_argument("--cycle-start", type=int, required=True)
    ap.add_argument("--cycle-end", type=int, required=True)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--out-prefix", default="idx_")
    args = ap.parse_args()

    vtus = cycle_vtus(args.results_dir, args.cycle_start, args.cycle_end)
    if not vtus:
        raise SystemExit(f"no VTU in [{args.cycle_start},{args.cycle_end}]")
    print(f"{len(vtus)} cycle snapshots: "
          f"{os.path.basename(vtus[0])} .. {os.path.basename(vtus[-1])}")

    wall = pv.read(args.wall)
    tawss, osi = wss_indices(vtus, wall, args.dt)
    wall["TAWSS"] = tawss; wall["OSI"] = osi
    wall_out = args.out_prefix + "wall_TAWSS_OSI.vtp"
    wall.save(wall_out)

    base, hmean, lnh = helicity(vtus)
    base["Helicity"] = hmean; base["LNH"] = lnh
    vol_out = args.out_prefix + "helicity.vtu"
    base.save(vol_out)

    print("\n--- summary (last cycle) ---")
    print(f"TAWSS  [dyne/cm^2] : mean {tawss.mean():.3f}  med {np.median(tawss):.3f}  max {tawss.max():.3f}")
    print(f"OSI    [-]         : mean {osi.mean():.4f}  max {osi.max():.4f}  "
          f"%(OSI>0.2) {100*np.mean(osi>0.2):.1f}")
    print(f"Helicity density   : mean {hmean.mean():.3e}")
    print(f"LNH    [-]         : mean {lnh.mean():+.4f}  |LNH| mean {np.abs(lnh).mean():.4f}")
    print(f"\n-> {wall_out}\n-> {vol_out}")


if __name__ == "__main__":
    main()
