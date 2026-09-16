#!/usr/bin/env python3
"""run_provenance.py — QUELLE config a produit ce run ? et un restart est-il FIDÈLE ?

Motivation (erreur payée sur ce projet) : une campagne Δt/2 de ~14 h a failli tourner avec la
MAUVAISE config — `MB_Parabolic.xml` au lieu de `MB_WALLONLY_PROD.xml` (déplacement pré-retiming
invalidé, Max_iter 8 vs 15, petsc-jacobi vs fsils, caps libres vs épinglés). Rien dans le dossier
du run ne disait quel XML l'avait produit. Seul un **contrôle de réplication nulle** l'a révélé.

Trois modes :
  compare   A.vtu B.vtu        -> verdict ROUND-OFF / DIFFÉRENT (le test qui fait foi)
  fingerprint RUN.vtu          -> empreinte du run (n_pts, champs, |D|max, <|v|>) pour matcher
  candidates DIR RUN.vtu       -> liste les XML du DIR avec leurs réglages discriminants,
                                  et signale ceux incompatibles avec l'empreinte du run

Usage :
  run_provenance.py compare  MB_NULLREP/4-procs/results_1520.vtu  MB_PROD/4-procs/results_1520.vtu
  run_provenance.py fingerprint MB_PROD/4-procs/results_1650.vtu
  run_provenance.py candidates ~/4826_sim/cap_pinning MB_PROD/4-procs/results_1650.vtu
"""
import sys, os, glob, re
import numpy as np

TOL_ROUNDOFF = 1e-5      # relatif ; au-dessus => ce n'est PAS une réplication


def _vel(m):
    for k in ("Velocity", "velocity"):
        if k in m.point_data:
            return np.asarray(m[k])
    raise KeyError(f"pas de champ Velocity ; disponibles={list(m.point_data.keys())}")


def cmd_compare(a_path, b_path):
    import pyvista as pv
    a, b = pv.read(a_path), pv.read(b_path)
    if a.n_points != b.n_points:
        print(f"VERDICT: INCOMPARABLE — n_points {a.n_points} vs {b.n_points}"); return 2
    dc = float(np.abs(a.points - b.points).max())
    va, vb = _vel(a), _vel(b)
    dv = np.linalg.norm(va - vb, axis=1)
    vmax = float(np.linalg.norm(vb, axis=1).max())
    rel = float(dv.max() / vmax) if vmax else float("inf")
    print(f"maxΔcoords      = {dc:.3e}")
    print(f"maxΔ|v|         = {dv.max():.3e}   (relatif {rel:.2e})")
    if "Displacement" in a.point_data and "Displacement" in b.point_data:
        dd = float(np.abs(np.asarray(a["Displacement"]) - np.asarray(b["Displacement"])).max())
        print(f"maxΔDisplacement= {dd:.3e}")
        if dd > 1e-6:
            print("  ⚠ le DÉPLACEMENT diffère -> cinématique/temps/config de paroi différents")
    ok = (rel < TOL_ROUNDOFF) and (dc < 1e-6)
    print(f"\nVERDICT: {'ROUND-OFF — réplication FIDÈLE' if ok else 'DIFFÉRENT — ce ne sont PAS les mêmes conditions'}")
    if not ok:
        print("  → si tu testais un restart : la config utilisée n'est pas celle du run d'origine,")
        print("    OU le restart n'est pas fidèle. Ne lance AUCUNE campagne avant d'avoir résolu ça.")
    return 0 if ok else 1


def cmd_fingerprint(path):
    import pyvista as pv
    m = pv.read(path)
    v = _vel(m); mag = np.linalg.norm(v, axis=1)
    print(f"fichier      : {path}")
    print(f"n_points     : {m.n_points}")
    print(f"champs       : {list(m.point_data.keys())}")
    print(f"<|v|>        : {mag.mean():.4f}   max {mag.max():.3f}")
    if "Displacement" in m.point_data:
        d = np.linalg.norm(np.asarray(m["Displacement"]), axis=1)
        print(f"|D| paroi    : mean {d.mean():.3f}  max {d.max():.3f}   <-- DISCRIMINANT de la cinématique")
        print("               (compare-le à l'amplitude du fichier de déplacement de chaque XML candidat)")
    else:
        print("|D|          : absent -> run RIGIDE (pas d'équation mesh)")
    return 0


_FIELDS = {
    "displacement": r"<Prescribed_displacement_file_path>\s*([^<\s]+)",
    "mesh":         r"<Mesh_file_path>\s*([^<\s]+)",
    "dt":           r"<Time_step_size>\s*([0-9.eE+-]+)",
    "nsteps":       r"<Number_of_time_steps>\s*([0-9]+)",
    "max_iter":     r"<Max_iterations>\s*([0-9]+)",
    "precond":      r"<Preconditioner>\s*([A-Za-z0-9_-]+)",
}


def cmd_candidates(dirpath, run_vtu):
    import pyvista as pv
    m = pv.read(run_vtu)
    has_disp = "Displacement" in m.point_data
    dmax = float(np.linalg.norm(np.asarray(m["Displacement"]), axis=1).max()) if has_disp else None
    print(f"empreinte du run : n_pts={m.n_points}  {'|D|max=%.3f' % dmax if has_disp else 'RIGIDE (pas de Displacement)'}\n")
    rows = []
    for x in sorted(glob.glob(os.path.join(dirpath, "*.xml"))):
        t = open(x, errors="ignore").read()
        g = {k: (re.search(p, t).group(1) if re.search(p, t) else "-") for k, p in _FIELDS.items()}
        mobile = g["displacement"] != "-"
        flag = ""
        if has_disp and not mobile:
            flag = "✗ run MOBILE mais XML sans déplacement"
        elif (not has_disp) and mobile:
            flag = "✗ run RIGIDE mais XML avec déplacement"
        rows.append((os.path.basename(x), g, flag))
    w = max(len(r[0]) for r in rows) if rows else 20
    print(f"{'XML'.ljust(w)}  dt        max_it  precond   déplacement")
    for name, g, flag in rows:
        disp = os.path.basename(g["displacement"]) if g["displacement"] != "-" else "-"
        parent = os.path.basename(os.path.dirname(g["displacement"])) if g["displacement"] != "-" else ""
        print(f"{name.ljust(w)}  {g['dt']:<9} {g['max_iter']:<6} {g['precond']:<9} {parent}/{disp} {flag}")
    print("\n⚠ CE TABLEAU NE PROUVE RIEN. Seule une RÉPLICATION NULLE fait foi :")
    print("   1) redémarrer avec le XML candidat au MÊME pas de temps depuis un .bin du run,")
    print("   2) run_provenance.py compare <nouveau>.vtu <stocké>.vtu  -> doit être ROUND-OFF.")
    return 0


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    cmd = sys.argv[1]
    if cmd == "compare" and len(sys.argv) == 4:      return cmd_compare(sys.argv[2], sys.argv[3])
    if cmd == "fingerprint" and len(sys.argv) == 3:  return cmd_fingerprint(sys.argv[2])
    if cmd == "candidates" and len(sys.argv) == 4:   return cmd_candidates(sys.argv[2], sys.argv[3])
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main())
