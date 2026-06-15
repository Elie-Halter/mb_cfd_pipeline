"""
Diagnostic (a): why node-move untangling plateaus.
1. Tets with 3/4 boundary nodes in the reference mesh (pinned if the whole boundary is prescribed).
2. Per phase: folds of the nearest-point correspondence (deformed wall triangles whose normal
   opposes the target STL normal at the matched point) + near-degenerate triangles.
3. RBF morph (reproduces the Escobar untangler init): inversions, and link to folds / pinning.
Output: <out_dir>/diag_pinned.txt
"""
import numpy as np, pyvista as pv, vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree

ROOT = "."
PHASES = ["phase_1_capped.stl", "phase_2_capped.stl", "phase_3_capped.stl", "phase_0_capped.stl"]

g = pv.read(f"{ROOT}/mesh.vtu")
X0 = np.asarray(g.points)
tet = g.cells_dict[10].astype(np.int64)
gid = np.asarray(g.point_data["GlobalNodeID"]).astype(int)
order = np.argsort(gid)

def idx_of(name):
    s = pv.read(f"{ROOT}/mesh-surfaces/{name}.vtp")
    sg = np.asarray(s.point_data["GlobalNodeID"]).astype(int)
    return order[np.searchsorted(gid[order], sg)], s

wall_i, wall = idx_of("wall")
cap_i = np.concatenate([idx_of(n)[0] for n in ["asc", "desc", "btca", "lcca", "lsa"]])
bnd = np.zeros(len(X0), bool); bnd[wall_i] = True; bnd[cap_i] = True
iswall = np.zeros(len(X0), bool); iswall[wall_i] = True

def signed_vol(P):
    a, b, c, d = P[tet[:, 0]], P[tet[:, 1]], P[tet[:, 2]], P[tet[:, 3]]
    return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a) / 6

v0 = signed_vol(X0)
print(f"ref: minV={v0.min():.4f}, inverted={int((v0<0).sum())}")

nb = bnd[tet].sum(1)
nw = iswall[tet].sum(1)
out = []
out.append(f"=== 1. structural pinning (reference mesh, {len(tet)} tets) ===")
out.append(f"tets with 4 boundary nodes (J fully determined by the prescription): {int((nb==4).sum())}")
out.append(f"tets with 3 boundary nodes: {int((nb==3).sum())}")
out.append(f"tets with 4 WALL nodes: {int((nw==4).sum())} ; with 3 WALL nodes: {int((nw==3).sum())}")

# wall triangles (connectivity from the surface vtp, remapped to volume indices)
wf = wall.faces.reshape(-1, 4)[:, 1:]
wgid = np.asarray(wall.point_data["GlobalNodeID"]).astype(int)
w2v = order[np.searchsorted(gid[order], wgid)]
wtri = w2v[wf]                       # (ntri,3) volume indices
def tri_normals(P, tri):
    n = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-30)[:, None], a / 2

n_ref, a_ref = tri_normals(X0, wtri)

def stl_with_normals(name):
    s = pv.read(f"{ROOT}/phases/{name}").triangulate().clean()
    s = s.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    loc = vtk.vtkCellLocator(); loc.SetDataSet(s); loc.BuildLocator()
    return s, loc, np.asarray(s.cell_data["Normals"])

def closest(loc, pts):
    q = np.empty_like(pts); cid = np.empty(len(pts), np.int64)
    c = vtk.mutable(0); sub = vtk.mutable(0); d = vtk.mutable(0.0)
    p = [0.0, 0.0, 0.0]
    gc = vtk.vtkGenericCell()
    for k in range(len(pts)):
        loc.FindClosestPoint(pts[k], p, gc, c, sub, d)
        q[k] = p; cid[k] = c.get()
    return q, cid

# consistent reference orientation: wall normals vs the reference STL
s4, loc4, cn4 = stl_with_normals("phase_ref_capped.stl")
cent0 = X0[wtri].mean(1)
_, cid0 = closest(loc4, cent0)
flip = (np.einsum('ij,ij->i', n_ref, cn4[cid0]) < 0)
sgn = np.where(flip, -1.0, 1.0)      # n_ref aligned with the STL orientation

out.append("\n=== 2. NEAREST-POINT correspondence folds, per phase (wall only) ===")
p4pts = np.asarray(s4.points)
tree4 = cKDTree(p4pts)
_, i4 = tree4.query(X0[wall_i]); ref_match = p4pts[i4]
disp_np = {}
for name in PHASES:
    s, loc, cn = stl_with_normals(name)
    spts = np.asarray(s.points)
    _, j = cKDTree(spts).query(ref_match)
    d = spts[j] - ref_match
    disp_np[name] = d
    P = X0.copy(); P[wall_i] += d
    n_d, a_d = tri_normals(P, wtri)
    cent = P[wtri].mean(1)
    _, cid = closest(loc, cent)
    dot = np.einsum('ij,ij->i', n_d * sgn[:, None], cn[cid])
    folded = int((dot < 0).sum()); near = int(((dot >= 0) & (dot < 0.2)).sum())
    degen = int((a_d < 0.05 * a_ref).sum())
    # tets inverted if ONLY the wall moves (the interior follows perfectly = lower bound)
    vw = signed_vol(P)
    inv_w = (vw < 0)
    out.append(f"{name:28s}: |d|max={np.linalg.norm(d,axis=1).max():5.2f}  "
               f"folded triangles={folded:4d}  near-folded(<0.2)={near:4d}  degenerate={degen:4d}  "
               f"inverted tets (wall-only moved)={int(inv_w.sum()):5d} "
               f"of which nw>=3: {int((inv_w & (nw>=3)).sum())}")

out.append("\n=== 3. RBF morph (untangler init) phase p8: which ones are inverted? ===")
name = "phase_1_capped.stl"
total = disp_np[name]
rs = np.random.RandomState(0); sw = rs.choice(len(wall_i), 1000, replace=False)
def Km(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
C = np.vstack([X0[wall_i][sw], X0[cap_i][:300]])
rhs = np.vstack([total[sw], np.zeros((300, 3))])
Wt = np.linalg.solve(Km(C, C) + 0.1 * np.eye(len(C)), rhs)
X = X0 + Km(X0, C) @ Wt
X[wall_i] = X0[wall_i] + total           # wall exactly prescribed
X[cap_i] = X0[cap_i]
v = signed_vol(X)
inv = v < 0
out.append(f"inverted={int(inv.sum())} (wall exactly prescribed, interior RBF 1300 centers)")
for k in range(5):
    out.append(f"  inverted with {k} boundary node(s): {int((inv & (nb==k)).sum())}")
# link to the folds: inverted tets touching a folded wall triangle
P = X0.copy(); P[wall_i] += total
n_d, _ = tri_normals(P, wtri)
s, loc, cn = stl_with_normals(name)
cent = P[wtri].mean(1); _, cid = closest(loc, cent)
fold_tri = (np.einsum('ij,ij->i', n_d * sgn[:, None], cn[cid]) < 0.1)
fold_nodes = set(wtri[fold_tri].ravel().tolist())
touch = np.array([len(fold_nodes.intersection(t)) > 0 for t in tet[inv]])
out.append(f"inverted tets touching a folded/near-folded triangle node: {int(touch.sum())}/{int(inv.sum())}")

txt = "\n".join(out)
open(f"{ROOT}/work/diag_pinned.txt", "w").write(txt + "\n")
print(txt)
