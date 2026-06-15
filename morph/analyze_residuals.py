"""
Analysis of the morph's residual inversions:
 - identity/stability of inverted elements across the samples
 - number of boundary nodes, quality of the supporting boundary triangle (registration)
 - would they be COUNTED by the solver? ("true collapse" criterion: 6V < -1e-3 in cm³;
   below that = sliver skipped from the assembly)
 - per-free-node LP certificate: is the intersection of the half-spaces (incident tets, other
   nodes fixed) empty? (proven local infeasibility -> requires a REFERENCE connectivity fix)
"""
import numpy as np, pyvista as pv
from scipy.optimize import linprog

ROOT = "."

g = pv.read(f"{ROOT}/mesh.vtu")
X0 = np.asarray(g.points).astype(float)
tet = g.cells_dict[10].astype(np.int64)
z = np.load(f"{ROOT}/work/morph_snapshots.npz")
pos, tt, vid = z["pos"].astype(float), z["t"], z["vid"]
bnd = np.zeros(len(X0), bool); bnd[vid] = True


def vols(P):
    a, b, c, d = P[tet[:, 0]], P[tet[:, 1]], P[tet[:, 2]], P[tet[:, 3]]
    return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a) / 6


allbad = {}
for k in range(len(pos)):
    v = vols(pos[k])
    for e in np.where(v <= 0)[0]:
        allbad.setdefault(int(e), []).append((k, float(v[e])))

print(f"distinct inverted elements over the whole cycle: {len(allbad)}")
counted = 0
for e, occ in sorted(allbad.items()):
    nb = int(bnd[tet[e]].sum())
    worstV = min(o[1] for o in occ)                    # mm³
    detJ_cm3 = 6 * worstV * 1e-3
    is_counted = detJ_cm3 < -1e-3
    counted += is_counted
    print(f"  tet {e}: present in {len(occ)}/{len(pos)} samples, n_boundary={nb}, "
          f"worstV={worstV:.2e} mm³ (6V={detJ_cm3:.2e} cm³) "
          f"{'COUNTED' if is_counted else 'skipped (sliver)'}")
print(f"\nelements that would be COUNTED by the solver (true collapse): {counted}")

# LP certificate at each element's worst sample (free tet nodes, one by one)
print("\n=== LP certificates (free node v: ∩ half-spaces of the incident tets) ===")
from collections import defaultdict
incident = defaultdict(list)
for i, t4 in enumerate(tet):
    for n in t4:
        incident[int(n)].append(i)

for e, occ in sorted(allbad.items()):
    k = min(occ, key=lambda o: o[1])[0]
    P = pos[k]
    freeN = [int(n) for n in tet[e] if not bnd[n]]
    verdicts = []
    for n in freeN:
        # demi-espaces: pour chaque tet incident, V>eps en fonction de x_n (lin.) autres fixes
        Acons, bcons = [], []
        feas_now = True
        for ti in incident[n]:
            t4 = tet[ti].tolist()
            loc = t4.index(n)
            others = [t4[(loc + 1) % 4], t4[(loc + 2) % 4], t4[(loc + 3) % 4]]
            # V = sign * det(...): volume = ((b-a)x(c-a))·(d-a)/6 with the tet ordering
            # reconstruct: V(x_n) = (n_vec · x_n + c)/6, n_vec = volume gradient
            a, b, c, d = [P[m] for m in tet[ti]]
            pts = [a, b, c, d]
            grad = np.zeros(3)
            # gradient by finite difference: volume is linear in x_n
            base = np.einsum('i,i->', np.cross(pts[1] - pts[0], pts[2] - pts[0]), pts[3] - pts[0]) / 6
            for ax in range(3):
                q = [p.copy() for p in pts]
                q[loc][ax] += 1.0
                vq = np.einsum('i,i->', np.cross(q[1] - q[0], q[2] - q[0]), q[3] - q[0]) / 6
                grad[ax] = vq - base
            cst = base - grad @ P[n]
            Acons.append(-grad); bcons.append(cst)   # -grad·x <= cst  <=>  grad·x + cst >= 0
        res = linprog(np.zeros(3), A_ub=np.array(Acons), b_ub=np.array(bcons) - 1e-12,
                      bounds=[(None, None)] * 3, method="highs")
        verdicts.append("FEASIBLE" if res.status == 0 else "INFEASIBLE")
    print(f"  tet {e} (sample {k}): free nodes {len(freeN)} -> {verdicts}")
