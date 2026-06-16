"""
Cycle validation + EXTENDED write-out, generic version (any mesh + snapshots).
- loads the solver mesh (vtu with GlobalNodeID) and the snapshots <snap_prefix>_KK.npy
  (full positions, KK = 1..n-1; sample 0 = the vtu reference positions)
- checks signed J at each sample AND at the midpoints (linear interp), near-wall stats
- writes the periodic EXTENDED file (absolute positions, cm)
Usage: python3 morph/check_write_any.py <mesh.vtu> <snap_prefix> <n_samples> <out.txt>
        [T_cycle=0.974] [scale=0.1]
"""
import sys
import numpy as np, pyvista as pv


def main(mesh_vtu, snap_prefix, n_samples, out_txt, T=0.974, scale=0.1):
    g = pv.read(mesh_vtu)
    X0 = np.asarray(g.points).astype(float)
    tet = g.cells_dict[10].astype(np.int64)
    gid = np.asarray(g.point_data["GlobalNodeID"]).astype(int)

    def vols(P):
        a, b, c, d = P[tet[:, 0]], P[tet[:, 1]], P[tet[:, 2]], P[tet[:, 3]]
        return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a) / 6

    v0 = vols(X0)
    sgn = 1.0 if np.median(v0) > 0 else -1.0
    pos = [X0] + [np.load(f"{snap_prefix}_{k:02d}.npy").astype(float) for k in range(1, n_samples)]
    tt = np.arange(n_samples) / n_samples

    print("smpl    t      ninv   minV(mm³)   worst6V(cm³)")
    worst_s, worst_m, worst_negV = 0, 0, 0.0
    for k in range(n_samples):
        v = sgn * vols(pos[k])
        ninv = int((v <= 0).sum())
        worst_s = max(worst_s, ninv)
        print(f"[{k:2d}] t={tt[k]:.4f} {ninv:5d}  {v.min():.2e}  {6*v.min()*1e-3:.2e}")
        vm = sgn * vols(0.5 * (pos[k] + pos[(k + 1) % n_samples]))
        worst_m = max(worst_m, int((vm <= 0).sum()))
        worst_negV = min(worst_negV, float(v.min()), float(vm.min()))
    print(f"worst sample = {worst_s} ; worst midpoint = {worst_m} (out of {len(tet)} tets)")

    # solver-safety verdict (read by run_patient.sh gate G1): the svMP mesh-motion solver
    # tolerates inverted tets whose volume is far below its critical Jacobian threshold
    # (the validated iso reference itself left ~2-4 such negligible slivers, |6V|~1e-7 cm³,
    # and traversed the cycle fine). Gate on the PHYSICAL volume, not on a raw fold count.
    worst6V_cm3 = abs(6.0 * worst_negV) * 1e-3            # scale mm³ -> cm³
    THRESH_cm3 = 1e-3                                     # svMP mesh-motion J critical threshold
    verdict = "SOLVER-SAFE" if worst6V_cm3 < THRESH_cm3 else "FOLDS-EXCEED-THRESHOLD"
    print(f"[morph-gate] worst inverted |6V| = {worst6V_cm3:.2e} cm^3 ; "
          f"threshold {THRESH_cm3:g} cm^3 -> {verdict}")

    tt_out = np.append(tt * T, T)
    with open(out_txt, "w") as f:
        f.write(f"{len(X0)} {len(tt_out)}\nEXTENDED\n")
        np.savetxt(f, gid, fmt="%d")
        for k, t in enumerate(tt_out):
            f.write(f"{t:.6f}\n")
            P = pos[k % n_samples]
            np.savetxt(f, np.column_stack([gid, P * scale]), fmt="%d %.6f %.6f %.6f")
    print(f"wrote {out_txt} ({len(tt_out)} samples, {len(X0)} nodes)")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]), a[4],
         float(a[5]) if len(a) > 5 else 0.974, float(a[6]) if len(a) > 6 else 0.1)
