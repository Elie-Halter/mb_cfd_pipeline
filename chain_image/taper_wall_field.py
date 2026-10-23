#!/usr/bin/env python3
"""
taper_wall_field.py -- step 4 of the image-registration wall chain: cap pinning + geodesic taper.

The caps are Dirichlet-0 in the solver's mesh equation. If the prescribed wall moves the wall
nodes on the cap rings, the elements between a moving ring and a fixed cap are crushed within
~20 ms of cycle. The wall displacement (relative to the MESH geometry) is therefore multiplied by
    w = smoothstep(g / L),
g = geodesic distance along the boundary surface to the nearest cap node: w = 0 on the rings,
w = 1 beyond L (default 12 mm). This is the same construction as morph/pin_caps.py in the
surface-registration chain.

A cheap fold proxy is printed for a few taper lengths: the boundary displacement at the phase of
largest mean motion is extended harmonically into the volume and inverted tetrahedra are
counted (the solver's mesh equation is not harmonic, so this is a screening, not a certificate).

Usage:
  python3 chain_image/taper_wall_field.py --in wall_positions_smooth.npz \
      --mesh mesh-complete.mesh.vtu --surfaces mesh-surfaces --caps inlet out1 out2 \
      --length 12 --out wall_positions_taper.npz [--sweep 0 8 12 16]
"""
import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "morph"))
import wallfield as wf
import pin_caps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="wall-field .npz")
    ap.add_argument("--mesh", required=True, help="mesh-complete.mesh.vtu (mm)")
    ap.add_argument("--surfaces", required=True, help="mesh-surfaces directory")
    ap.add_argument("--caps", nargs="+", required=True, help="cap face names (without .vtp)")
    ap.add_argument("--length", type=float, default=12.0, help="taper length L (mm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", nargs="*", type=float, default=None,
                    help="also print the fold proxy for these taper lengths (mm)")
    a = ap.parse_args()

    import pyvista as pv
    X0, vid, _, cap_local, gdist = pin_caps.load_geom(a.mesh, a.surfaces, caps=a.caps)
    tet = pv.read(a.mesh).cells_dict[10].astype(np.int64)
    print(f"[taper] boundary nodes {len(vid)}, cap nodes {int(cap_local.sum())}, "
          f"max geodesic distance {gdist[np.isfinite(gdist)].max():.1f} mm")
    pos, gid, ref = wf.load_field(a.inp)
    ref_xyz = X0[gid - 1]                                    # mesh geometry of the wall nodes
    if ref is not None:
        print(f"[taper] |field(reference phase) - mesh| max = {np.abs(pos[ref] - ref_xyz).max():.3f} mm")
    v2l = -np.ones(len(X0), np.int64); v2l[vid] = np.arange(len(vid))
    loc = v2l[gid - 1]
    if (loc < 0).any():
        sys.exit("wall-field nodes not on the mesh boundary")
    D = pos - ref_xyz                                        # relative to the MESH
    pmax = int(np.argmax(np.linalg.norm(D, axis=2).mean(axis=1)))

    if a.sweep is not None:
        sgn = 1.0 if np.median(pin_caps._signed_vol(X0, tet)) > 0 else -1.0
        print(f"[taper] fold proxy at phase {pmax} (largest mean |d|):")
        print("%8s %12s %9s %12s" % ("L (mm)", "ring |d| max", "inv_tets", "min V (mm3)"))
        for L in [None] + list(a.sweep):
            w = np.ones(len(gid)) if L is None else pin_caps.taper_weights(gdist, cap_local, L)[loc]
            bnd = np.zeros((len(vid), 3)); bnd[loc] = w[:, None] * D[pmax]
            v = sgn * pin_caps._signed_vol(pin_caps.harmonic_extend(X0, tet, vid, bnd), tet)
            ring = np.linalg.norm(bnd[cap_local], axis=1).max()
            print("%8s %12.2e %9d %12.2e" % ("none" if L is None else f"{L:g}", ring, int((v <= 0).sum()), v.min()))

    w = pin_caps.taper_weights(gdist, cap_local, a.length)[loc]
    Dt = w[None, :, None] * D
    wf.save_field(a.out, ref_xyz + Dt, gid, ref_phase=ref)
    nD, nDt = np.linalg.norm(D, axis=2), np.linalg.norm(Dt, axis=2)
    band = w < 0.999
    print(f"[taper] L = {a.length:g} mm: {int(band.sum())}/{len(w)} wall nodes attenuated "
          f"({int((w == 0).sum())} pinned); max |d| {nD.max():.2f} -> {nDt.max():.2f} mm; "
          f"mean |d| in the band {nD[:, band].mean():.2f} -> {nDt[:, band].mean():.2f} mm")
    print(f"[taper] wrote {a.out}")


if __name__ == "__main__":
    main()
