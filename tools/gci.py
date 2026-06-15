#!/usr/bin/env python3
"""
Grid Convergence Index (GCI) -- mesh-independence verification (Roache 1994;
Celik et al. 2008, ASME J. Fluids Eng.). The standard expected for reporting the
convergence of a WSS metric on >=3 grids.

Method (3 grids, fine->medium->coarse, scalar metric phi):
  h_i  = (V_total / N_cells_i)^(1/3)            (representative cell size, Celik)
  r21 = h2/h1,  r32 = h3/h2                       (>1)
  eps21 = phi2-phi1,  eps32 = phi3-phi2;  s = sign(eps32/eps21)
  apparent order p: fixed point
     p = |ln|eps32/eps21| + q(p)| / ln(r21),  q(p) = ln((r21^p - s)/(r32^p - s))
  phi_ext21 = (r21^p*phi1 - phi2)/(r21^p - 1)
  e_a21 = |(phi1-phi2)/phi1|;  GCI21 = Fs*e_a21/(r21^p - 1)   (Fs=1.25 for >=3 grids)
  asymptotic range: GCI32 / (r21^p * GCI21) ~ 1

Usage (explicit values):
  python3 gci.py --h 0.18 0.24 0.32 --phi 26.4 25.9 24.8
Usage (from meshes + TAWSS walls):  h computed from each .vtu, phi = area-averaged TAWSS
  python3 gci.py --mesh M1.vtu M2.vtu M3.vtu --wall M1_idx_wall_TAWSS_OSI.vtp M2_*.vtp M3_*.vtp \
                 [--field TAWSS]
Argument order: ALWAYS fine, medium, coarse.
"""
import argparse, sys
import numpy as np


def repr_h(mesh_path):
    import pyvista as pv
    m = pv.read(mesh_path)
    V = float(np.sum(np.abs(m.compute_cell_sizes(length=False, area=False, volume=True)
                            .cell_data["Volume"])))
    n = m.n_cells
    return (V / n) ** (1.0 / 3.0), n, V


def area_avg_field(wall_vtp, field):
    import pyvista as pv
    w = pv.read(wall_vtp).compute_cell_sizes(length=False, area=True, volume=False)
    if field not in w.point_data:
        raise SystemExit(f"field '{field}' missing from {wall_vtp} (run hemo_indices first)")
    # convert nodal field -> cells, weight by area
    wc = w.point_data_to_cell_data()
    a = np.abs(np.asarray(w.cell_data["Area"]))
    f = np.asarray(wc.cell_data[field])
    return float(np.sum(f * a) / np.sum(a))


def apparent_order(phi, r21, r32, tol=1e-10, itmax=1000):
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    ratio = e32 / e21
    if not np.isfinite(ratio) or ratio <= 0:
        # non-monotone convergence: p undefined, flag it
        return None, e21, e32
    s = np.sign(ratio)
    p = 2.0
    for _ in range(itmax):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    return p, e21, e32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", nargs=3, type=float, help="cell sizes fine medium coarse")
    ap.add_argument("--phi", nargs=3, type=float, help="scalar metric (fine medium coarse)")
    ap.add_argument("--mesh", nargs=3, help="3 meshes .vtu (fine medium coarse) -> h auto")
    ap.add_argument("--wall", nargs=3, help="3 walls .vtp with the field -> phi = area-average")
    ap.add_argument("--field", default="TAWSS")
    ap.add_argument("--fs", type=float, default=1.25, help="safety factor (1.25 for >=3 grids)")
    a = ap.parse_args()

    if a.mesh:
        H = [repr_h(m) for m in a.mesh]
        h = [x[0] for x in H]
        print("mesh       N_cells     V_total      h_repr")
        for m, (hi, n, V) in zip(a.mesh, H):
            print(f"  {m.split('/')[-1]:18} {n:9d} {V:11.3g} {hi:10.5f}")
    elif a.h:
        h = list(a.h)
    else:
        raise SystemExit("provide --h or --mesh")

    if a.wall:
        phi = [area_avg_field(w, a.field) for w in a.wall]
        print(f"phi = area-averaged {a.field}: fine {phi[0]:.4f}  med {phi[1]:.4f}  coarse {phi[2]:.4f}")
    elif a.phi:
        phi = list(a.phi)
    else:
        raise SystemExit("provide --phi or --wall")

    if not (h[0] < h[1] < h[2]):
        print(f"[WARN] h must increase fine<med<coarse; got {h} -- check the argument order.")
    r21, r32 = h[1] / h[0], h[2] / h[1]
    p, e21, e32 = apparent_order(phi, r21, r32)

    print("\n--- GCI (Celik et al. 2008) ---")
    print(f"r21 = {r21:.4f}   r32 = {r32:.4f}")
    if p is None:
        print("WARN: NON-MONOTONE convergence (eps32/eps21 <= 0): GCI undefined. "
              "Refine further or choose another metric.")
        return
    phi_ext = (r21 ** p * phi[0] - phi[1]) / (r21 ** p - 1.0)
    ea21 = abs((phi[0] - phi[1]) / phi[0])
    ea32 = abs((phi[1] - phi[2]) / phi[1])
    eext21 = abs((phi_ext - phi[0]) / phi_ext)
    gci21 = a.fs * ea21 / (r21 ** p - 1.0)
    gci32 = a.fs * ea32 / (r32 ** p - 1.0)
    asymp = gci32 / (r21 ** p * gci21)
    print(f"apparent order p            = {p:.3f}")
    print(f"extrapolated value phi_ext21 = {phi_ext:.4f}")
    print(f"relative error e_a21         = {100*ea21:.3f} %")
    print(f"extrapolated error e_ext21   = {100*eext21:.3f} %")
    print(f"GCI_fine (21)               = {100*gci21:.3f} %   <- report this")
    print(f"GCI (32)                    = {100*gci32:.3f} %")
    print(f"asymptotic range GCI32/(r21^p*GCI21) = {asymp:.3f}  (~1 = converged)")


if __name__ == "__main__":
    main()
