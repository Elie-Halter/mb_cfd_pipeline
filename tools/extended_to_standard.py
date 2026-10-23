#!/usr/bin/env python3
"""
extended_to_standard.py -- turn the EXTENDED displacement file written by morph/ (every node of
the domain, offline rest-shape morph) into the STANDARD wall-motion file used in production.

morph/pipeline.py and morph/run_pinned_morph.py write an EXTENDED file ("n nT", an "EXTENDED"
line, the GID list, then per time "t" and "gid x y z"). With that keyword the patched solver
imposes the morph on EVERY node (patch 0013, not used in production). Production takes only the
wall positions from it and lets the solver's mesh equation move the interior: the same blocks
without the keyword and GID list form a STANDARD file, which the solver maps by GID onto the
face it is attached to and ignores everywhere else. With --face, only the nodes of that face
are kept (smaller file, same effect).

Usage:
  python3 tools/extended_to_standard.py displacement_all_nodes.txt displacement_wall.txt \
      [--face mesh-surfaces/wall.vtp]

The mesh equation then needs the wall as Prescribed_displacement (this file) and the caps as
Dir 0 -- see CONVENTIONS.md section 4.
"""
import argparse
import sys
import numpy as np


def convert(src, dst, keep_gids=None):
    with open(src) as f, open(dst, "w") as g:
        n, nt = map(int, f.readline().split())
        if f.readline().strip() != "EXTENDED":
            raise SystemExit(f"{src}: not an EXTENDED file (no keyword on line 2)")
        gids = np.array([int(f.readline()) for _ in range(n)])
        keep = np.ones(n, bool) if keep_gids is None else np.isin(gids, keep_gids)
        if keep_gids is not None and keep.sum() != len(np.unique(keep_gids)):
            raise SystemExit(f"{len(np.unique(keep_gids)) - keep.sum()} face nodes missing from {src}")
        g.write(f"{int(keep.sum())} {nt}\n")
        for _ in range(nt):
            g.write(f.readline())
            lines = [f.readline() for _ in range(n)]
            g.writelines(l for l, k in zip(lines, keep) if k)
    return int(keep.sum()), nt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--face", help="keep only the nodes of this face (.vtp with GlobalNodeID)")
    a = ap.parse_args()
    keep = None
    if a.face:
        import pyvista as pv
        keep = np.asarray(pv.read(a.face).point_data["GlobalNodeID"]).astype(int)
    n, nt = convert(a.src, a.dst, keep)
    print(f"wrote {a.dst}: STANDARD, {n} nodes x {nt} samples")


if __name__ == "__main__":
    sys.exit(main())
