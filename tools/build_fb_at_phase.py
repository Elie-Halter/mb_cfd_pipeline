#!/usr/bin/env python3
"""Build a rigid-wall mesh frozen at an arbitrary phase of a prescribed cycle.

A fixed-boundary (FB) run has to freeze the lumen at *some* instant of the cardiac cycle,
and the choice is rarely stated: the reference phase of the segmentation, the temporal mean
and end-diastole are all defensible, and they are not equivalent. This tool builds the rigid
mesh at any phase of an existing prescribed-motion field, so that the choice can be varied
and its effect measured instead of assumed.

The output keeps the GlobalNodeID numbering and the connectivity of the input mesh, so FB
runs built at different phases stay node-to-node comparable with each other and with the
moving-boundary run. If the prescribed field pins the caps -- as `morph/pin_caps.py` does --
then cap areas, the RCR parameters and the imposed flow rate are identical across phases, and
the frozen wall is the only thing that changes.

Two input formats, matching the two ways a displacement field is produced upstream:

  --wall-npz      an .npz with `positions` (n_phases, n_wall_nodes, 3) and `gid`
                  (n_wall_nodes,), i.e. wall nodes only. The interior is reconstructed by
                  harmonic extension, as in morph/pin_caps.py.
  --extended-txt  the solver's EXTENDED prescribed-displacement file, which already carries
                  every node of the domain. No reconstruction needed.

Usage:
  # what does the cycle look like, and which phase is the most contracted?
  python3 tools/build_fb_at_phase.py --mesh ref.vtu --surfaces mesh-surfaces \
      --wall-npz wall_positions.npz --ref-index 17 --scan

  # build the mesh at that phase
  python3 tools/build_fb_at_phase.py --mesh ref.vtu --surfaces mesh-surfaces \
      --wall-npz wall_positions.npz --ref-index 17 --phase 12 --out-dir mesh-complete_ph12

Then build the solver input for the new mesh with tools/make_patient_xml.py.
"""
import argparse
import glob
import os
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "morph"))
import pin_caps  # noqa: E402


def signed_volumes(points, tets):
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def read_extended(path, scale):
    """Yield (index, time, positions) for every frame of an EXTENDED file.

    Format: a header line `n_nodes n_times`, then per frame a time line followed by
    n_nodes lines of `global_node_id x y z`. Rows are returned ordered by id.
    """
    with open(path) as fh:
        n_nodes, n_times = (int(v) for v in fh.readline().split())
        for index in range(n_times):
            time = float(fh.readline())
            raw = np.fromstring("".join(islice(fh, n_nodes)), sep=" ").reshape(n_nodes, 4)
            order = raw[:, 0].astype(int) - 1
            pos = np.empty((n_nodes, 3))
            pos[order] = raw[:, 1:] * scale
            yield index, time, pos


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, help="reference mesh-complete .vtu")
    p.add_argument("--surfaces", required=True, help="directory of boundary .vtp files")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--wall-npz", help="wall-only positions (.npz with positions, gid)")
    src.add_argument("--extended-txt", help="EXTENDED whole-domain prescribed-displacement file")
    p.add_argument("--ref-index", type=int, default=0,
                   help="index of the phase the reference mesh sits at (default 0)")
    p.add_argument("--phase", type=int, help="phase/frame index to freeze at")
    p.add_argument("--scan", action="store_true",
                   help="report lumen volume for every phase and exit, writing nothing")
    p.add_argument("--input-scale", type=float, default=1.0,
                   help="multiplier on input coordinates, e.g. 10 for cm into a mm mesh")
    p.add_argument("--out-dir", help="output directory (with --phase)")
    a = p.parse_args()
    if not a.scan and (a.phase is None or not a.out_dir):
        p.error("--phase and --out-dir are required unless --scan is given")

    mesh = pv.read(a.mesh)
    tets = mesh.cells.reshape(-1, 5)[:, 1:]
    mesh_gid = np.asarray(mesh.point_data["GlobalNodeID"]).astype(int) - 1
    orientation = np.sign(np.median(signed_volumes(np.asarray(mesh.points), tets)))

    if a.wall_npz:
        npz = np.load(a.wall_npz)
        wall_pos = npz["positions"].astype(float) * a.input_scale
        wall_gid = npz["gid"].astype(int)
        X0, surf_ids, _, _, _ = pin_caps.load_geom(a.mesh, a.surfaces)
        to_local = -np.ones(len(X0), np.int64)
        to_local[surf_ids] = np.arange(len(surf_ids))
        wall_local = to_local[wall_gid - 1]
        n_phases = wall_pos.shape[0]

        def positions(k):
            boundary = np.zeros((len(surf_ids), 3))
            boundary[wall_local] = wall_pos[k] - wall_pos[a.ref_index]
            return pin_caps.harmonic_extend(X0, tets, surf_ids, boundary)

        times = [None] * n_phases
    else:
        frames = {i: (t, x) for i, t, x in read_extended(a.extended_txt, a.input_scale)}
        n_phases = len(frames)
        times = [frames[i][0] for i in range(n_phases)]

        def positions(k):
            return frames[k][1]

    # the reference phase must reproduce the input mesh, which also catches a unit mismatch
    deviation = np.linalg.norm(positions(a.ref_index)[mesh_gid] - np.asarray(mesh.points), axis=1).max()
    if deviation > 1e-3:
        sys.exit(f"the field at index {a.ref_index} is {deviation:.3g} away from the mesh; "
                 f"wrong --ref-index, or units differ (try --input-scale)")

    if a.scan:
        print(f"{'phase':>6} {'time':>10} {'volume':>12} {'inverted':>9}")
        volumes = []
        for k in range(n_phases):
            v = orientation * signed_volumes(positions(k)[mesh_gid], tets)
            volumes.append(abs(v.sum()))
            t = "-" if times[k] is None else f"{times[k]:.4f}"
            print(f"{k:6d} {t:>10} {volumes[-1]:12.1f} {(v <= 0).sum():9d}", flush=True)
        volumes = np.array(volumes)
        lo, hi = int(volumes.argmin()), int(volumes.argmax())
        span = volumes[hi] - volumes[lo]
        print(f"\nsmallest at phase {lo} ({volumes[lo]:.1f}), largest at phase {hi} ({volumes[hi]:.1f}), "
              f"span {100 * span / volumes[lo]:.1f}% of the smallest")
        print(f"the reference phase {a.ref_index} sits at "
              f"{100 * (volumes[a.ref_index] - volumes[lo]) / span:.1f}% of that span")
        return

    pos = positions(a.phase)
    v = orientation * signed_volumes(pos[mesh_gid], tets)
    print(f"phase {a.phase}: volume {abs(v.sum()):.1f}, inverted elements {(v <= 0).sum()}")
    if (v <= 0).any():
        sys.exit("inverted elements at this phase: pick another phase, or fix the field upstream")

    os.makedirs(os.path.join(a.out_dir, "mesh-surfaces"), exist_ok=True)
    out = mesh.copy()
    out.points = pos[mesh_gid]
    out.save(os.path.join(a.out_dir, "mesh-complete.mesh.vtu"))
    moved = []
    for path in sorted(glob.glob(os.path.join(a.surfaces, "*.vtp"))):
        s = pv.read(path)
        gid = np.asarray(s.point_data["GlobalNodeID"]).astype(int) - 1
        s2 = s.copy()
        s2.points = pos[gid]
        s2.save(os.path.join(a.out_dir, "mesh-surfaces", os.path.basename(path)))
        shift = np.linalg.norm(pos[gid] - np.asarray(s.points), axis=1).max()
        moved.append(shift)
        print(f"  {os.path.basename(path):<16} max node shift {shift:.4f}")
    if min(moved) > 1e-6:
        print("warning: every boundary moved, so no face is pinned. Cap areas, the RCR "
              "parameters and the flow split will differ from the reference mesh, and FB runs "
              "built at different phases will not be comparable.")
    print(f"\nwrote {a.out_dir}; build the solver input for it with tools/make_patient_xml.py")


if __name__ == "__main__":
    main()
