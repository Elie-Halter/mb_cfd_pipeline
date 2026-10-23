#!/usr/bin/env python3
"""
smooth_wall_field.py -- step 3 of the image-registration wall chain: repair the propagated
wall field before it is prescribed.

A field propagated by image registration carries two defect families that the solver does not
forgive:
  1. sub-millimetre spatial roughness (registration noise, IDW-filled spots) -> O(1) strain
     between neighbouring wall nodes -> near-wall elements invert within a few time steps,
     however well the mesh equation is solved;
  2. genuine folds of the prescribed wall surface (triangle normals flipping) at some instants
     of the interpolated cycle.

Repair, applied to the displacement relative to the reference phase (which stays exactly zero):
  (a) global Taubin smoothing (lambda/mu, amplitude-preserving) on the wall-surface graph, every
      phase;
  (b) local Laplacian passes on the k-ring neighbourhood of every triangle that still folds
      anywhere in the interpolated cycle, repeated until no fold is left;
  (c) local Laplacian passes around every edge whose strain |d_a - d_b| / |x_a - x_b| exceeds a
      threshold at any instant, repeated until none does.
The report gives the fold count, the worst normal ratio and the maximum edge strain after each
stage, the change of the field and the change of the enclosed volume per phase.

Usage:
  python3 chain_image/smooth_wall_field.py --in wall_positions.npz --wall-vtp mesh-surfaces/wall.vtp \
      --period 0.9 --out wall_positions_smooth.npz [--ref-index 10]
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wallfield as wf


def report(tag, m, thr):
    print(f"[smooth] {tag}: folded triangles {len(m['folded'])}, worst normal ratio "
          f"{m['worst_ratio']:.3f} (t={m['worst_t']:.3f} s), max edge strain {m['edge_strain'].max():.2f} "
          f"({int((m['edge_strain'] > thr).sum())} edges > {thr})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="wall-field .npz")
    ap.add_argument("--wall-vtp", required=True, help="wall face of the solver mesh (topology)")
    ap.add_argument("--period", type=float, required=True, help="cardiac cycle (s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-index", type=int, help="reference phase (default: stored in the .npz)")
    ap.add_argument("--taubin-pairs", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=-0.53)
    ap.add_argument("--fold-rings", type=int, default=3)
    ap.add_argument("--fold-passes", type=int, default=6)
    ap.add_argument("--strain-max", type=float, default=0.8, help="edge-strain threshold")
    ap.add_argument("--strain-rings", type=int, default=2)
    ap.add_argument("--strain-passes", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.002, help="cycle scan step for folds (s)")
    ap.add_argument("--strain-dt", type=float, default=0.004, help="cycle scan step for edge strain (s)")
    a = ap.parse_args()

    pos, gid, ref_stored = wf.load_field(a.inp)
    ref = a.ref_index if a.ref_index is not None else ref_stored
    if ref is None:
        sys.exit("reference phase unknown: pass --ref-index")
    G = wf.WallGraph(a.wall_vtp, gid)
    base = pos[ref]
    D0 = pos - base
    D = D0.copy()
    nph = len(pos)
    t_ref = ref / nph * a.period                     # scans start at the reference phase
    scan = lambda Df, dt=a.dt: wf.cycle_metrics(G, base, Df, a.period, dt=dt, t0=t_ref)   # noqa: E731

    report("input", scan(D), a.strain_max)
    D = np.array([G.taubin(D[p], a.taubin_pairs, a.lam, a.mu) for p in range(nph)])
    D[ref] = 0.0
    m = scan(D)
    report("after global Taubin", m, a.strain_max)

    for it in range(a.fold_passes):
        if not m["folded"]:
            break
        mask = G.rings(np.unique(G.tri[m["folded"]]), a.fold_rings).astype(float)
        D = np.array([G.smooth(D[p], 15, a.lam, mask) for p in range(nph)])
        D[ref] = 0.0
        m = scan(D)
        report(f"fold pass {it + 1} ({int(mask.sum())} nodes)", m, a.strain_max)

    m = scan(D, a.strain_dt)
    for it in range(a.strain_passes):
        hot = np.where(m["edge_strain"] > a.strain_max)[0]
        if len(hot) == 0:
            break
        mask = G.rings(np.unique(G.edges[hot]), a.strain_rings).astype(float)
        for p in range(nph):
            if p != ref:
                D[p] = G.smooth(D[p], 12, a.lam, mask)
        m = scan(D, a.strain_dt)
        report(f"strain pass {it + 1} ({int(mask.sum())} nodes)", m, a.strain_max)
    m = scan(D)
    report("final", m, a.strain_max)

    delta = np.linalg.norm(D - D0, axis=2)
    print(f"[smooth] field change: max {delta.max():.3f} mm, p99 {np.quantile(delta, 0.99):.3f} mm, "
          f"mean {delta.mean():.4f} mm")
    dv = [G.enclosed_volume_ml(base + D[p]) - G.enclosed_volume_ml(base + D0[p]) for p in range(nph)]
    print(f"[smooth] max |enclosed-volume change| over the phases: {np.abs(dv).max():.2f} mL "
          "(open wall surface: relative indicator only)")
    if m["folded"]:
        print(f"[smooth] WARNING: {len(m['folded'])} triangles still fold")
    wf.save_field(a.out, base + D, gid, ref_phase=ref)
    print(f"[smooth] wrote {a.out}")


if __name__ == "__main__":
    main()
