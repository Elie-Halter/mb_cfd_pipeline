#!/usr/bin/env python3
"""
phase_subset_wall.py -- rebuild a prescribed wall motion from a chosen SUBSET of its phases.

How many segmented phases does a moving-wall run need, and which ones? Starting from a wall
field with all phases (e.g. 20 phases from chain_image/), this tool keeps only the listed
keyframes, interpolates them in time with a shape-preserving periodic PCHIP (two keyframes
replicated on each side of the period), and resamples the result at every original phase time.
The output has the same layout as the input, so it goes through the rest of the chain
(chain_image/taper_wall_field.py, write_wall_displacement.py) unchanged, and the run can be
compared with the full-field run.

The report gives, against the full field: mean / p99 / max nodal position error over all phases,
the mean error as a fraction of the mean motion amplitude, and the error at the reference phase
(non-zero when the reference phase is not a keyframe: write_wall_displacement.py then blends the
mesh into the field during its pre-roll).

Phases are evenly spaced over the cycle (phase k at k/N of the period), so the result does not
depend on the period value.

Usage:
  python3 tools/phase_subset_wall.py wall_positions.npz --keep 10 13 5 --out subset.npz [--ref-index 10]
  python3 tools/phase_subset_wall.py wall_positions.npz --keep 0 4 8 12 16 --out even5.npz

Input/output: .npz with positions (n_phases, n_wall_nodes, 3) mm, gid, optional ref_phase.
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chain_image"))
import wallfield as wf


def subset_field(pos, keep, wrap="pad"):
    """Positions at every phase, interpolated from the keyframes `keep` only."""
    n = len(pos)
    keep = sorted(set(int(k) for k in keep))
    if len(keep) < 2:
        raise ValueError("need at least two keyframes")
    if min(keep) < 0 or max(keep) >= n:
        raise ValueError(f"keyframes must be in 0..{n - 1}")
    tau = wf.phase_times(n, 1.0)
    f = wf.periodic_pchip(tau[keep], pos[keep], 1.0, wrap=wrap)
    return f(tau)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field", help="wall-field .npz with all phases")
    ap.add_argument("--keep", nargs="+", type=int, required=True, help="phase indices to keep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-index", type=int, help="reference (mesh) phase; default: stored in the .npz")
    a = ap.parse_args()
    pos, gid, ref_stored = wf.load_field(a.field)
    ref = a.ref_index if a.ref_index is not None else ref_stored
    rs = subset_field(pos, a.keep)
    err = np.linalg.norm(rs - pos, axis=2)
    if ref is not None:
        swing = np.linalg.norm(pos - pos[ref], axis=2).max(axis=0)
        print(f"full field: per-node motion amplitude mean {swing.mean():.2f} mm, max {swing.max():.2f} mm")
    keep = sorted(set(a.keep))
    msg = (f"keep {keep}: error vs full field mean {err.mean():.3f} mm, p99 {np.percentile(err, 99):.3f} mm, "
           f"max {err.max():.3f} mm")
    if ref is not None:
        msg += (f" ({100 * err.mean() / swing.mean():.1f} % of the mean amplitude); "
                f"reference phase {ref}: max {err[ref].max():.3f} mm"
                + ("" if ref in keep else " (not a keyframe)"))
    print(msg)
    wf.save_field(a.out, rs, gid, ref_phase=ref)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
