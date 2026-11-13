#!/usr/bin/env python3
"""gen_mesh_gmsh.py -- all-tetrahedral mesh of the aortic lumen with Gmsh, in the svMultiPhysics mesh-complete format.

Input directory: wall_open.stl (lumen wall, open at the five vessel ends) and cuts.json (centre of each vessel end:
asc, desc, btca, lcca, lsa). The open ends are closed by plane caps, the volume is meshed with Gmsh
(Frontal-Delaunay 2D, Delaunay 3D, size between MESH_MIN and MESH_MAX mm, refined with the wall curvature), and the
mesh is written as mesh-complete.mesh.vtu + mesh-surfaces/{wall,asc,desc,btca,lcca,lsa}.vtp with GlobalNodeID /
GlobalElementID. Defaults (MESH_MIN 0.8, MESH_MAX 1.4, MESH_CURV 20) give the medium mesh used in the paper; the
coarse and fine meshes of the convergence study change MESH_MIN / MESH_MAX.
Usage: python3 gen_mesh_gmsh.py GEO_DIR OUT_DIR
"""
import gmsh, vtk, numpy as np, json, sys
from vtk.util import numpy_support as ns

GEO = sys.argv[1]
cuts = json.load(open(f"{GEO}/cuts.json"))

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.merge(f"{GEO}/wall_open.stl")
gmsh.model.mesh.classifySurfaces(float(__import__("os").environ.get("CLASS_ANGLE","40"))*np.pi/180, True, True, 180*np.pi/180)
gmsh.model.mesh.createGeometry()

# rim curves = boundary curves adjacent to exactly one surface
curves = gmsh.model.getEntities(1)
rims = []
for d,t in curves:
    up,_ = gmsh.model.getAdjacencies(1,t)
    if len(up) == 1:
        rims.append(t)
print("rim curves:", rims)
# group rim curves into closed loops via shared endpoints
ends = {t: set(abs(p[1]) for p in gmsh.model.getBoundary([(1,t)], oriented=False)) for t in rims}
loops, pool = [], set(rims)
while pool:
    t0 = pool.pop(); grp = [t0]; pts = set(ends[t0])
    changed = True
    while changed:
        changed = False
        for t in list(pool):
            if ends[t] & pts:
                grp.append(t); pts |= ends[t]; pool.discard(t); changed = True
    loops.append(grp)
print("loops:", loops)
cap_tags = {}
for grp in loops:
    cl = gmsh.model.geo.addCurveLoop(grp, reorient=True)
    ps = gmsh.model.geo.addPlaneSurface([cl])
    allc = []
    for t in grp:
        ntags, coords, _ = gmsh.model.mesh.getNodes(1, t, includeBoundary=True)
        allc.append(np.array(coords).reshape(-1,3))
    c = np.vstack(allc).mean(0)
    which = min(cuts, key=lambda k: np.linalg.norm(np.array(cuts[k]["center"])-c))
    cap_tags[ps] = which
    print("cap", ps, "->", which, np.round(c,1))
assert sorted(cap_tags.values()) == sorted(cuts.keys()), cap_tags
gmsh.model.geo.synchronize()
allsurf = [t for d,t in gmsh.model.getEntities(2)]
sl = gmsh.model.geo.addSurfaceLoop(allsurf)
vol = gmsh.model.geo.addVolume([sl])
gmsh.model.geo.synchronize()

import os as _os
H_MIN = float(_os.environ.get("MESH_MIN", "0.8"))
H_MAX = float(_os.environ.get("MESH_MAX", "1.4"))
gmsh.option.setNumber("Mesh.MeshSizeMin", H_MIN)
gmsh.option.setNumber("Mesh.MeshSizeMax", H_MAX)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", float(_os.environ.get("MESH_CURV", "20")))
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", float(_os.environ.get("MESH_ALGO3D", "1")))
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.model.mesh.generate(3)

# ---- extract ----
ntags, ncoords, _ = gmsh.model.mesh.getNodes()
ntags = np.array(ntags, dtype=np.int64); coords = np.array(ncoords).reshape(-1,3)
order = np.argsort(ntags)
ntags, coords = ntags[order], coords[order]
remap = np.zeros(ntags.max()+1, dtype=np.int64); remap[ntags] = np.arange(len(ntags))

etypes, etags, enodes = gmsh.model.mesh.getElements(3)
tets = remap[np.array(enodes[0], dtype=np.int64).reshape(-1,4)]
print("nodes:", len(ntags), "tets:", len(tets))

faces = {}
for d,t in gmsh.model.getEntities(2):
    ft, fe, fn = gmsh.model.mesh.getElements(2, t)
    if not fe: continue
    tris = remap[np.array(fn[0], dtype=np.int64).reshape(-1,3)]
    name = cap_tags.get(t, "wall")
    faces.setdefault(name, []).append(tris)
faces = {k: np.vstack(v) for k,v in faces.items()}
for k,v in faces.items(): print("face", k, len(v), "tris")
gmsh.finalize()

# ---- write svMP mesh-complete ----
import os
OUT = sys.argv[2]
os.makedirs(f"{OUT}/mesh-surfaces", exist_ok=True)

vp = vtk.vtkPoints(); vp.SetData(ns.numpy_to_vtk(coords, deep=True))
ug = vtk.vtkUnstructuredGrid(); ug.SetPoints(vp)
ca = vtk.vtkCellArray()
conn = np.hstack([np.full((len(tets),1),4,dtype=np.int64), tets]).ravel()
ca.SetCells(len(tets), ns.numpy_to_vtkIdTypeArray(conn, deep=True))
ug.SetCells(vtk.VTK_TETRA, ca)
gid = ns.numpy_to_vtk(np.arange(1,len(ntags)+1,dtype=np.int32), deep=True); gid.SetName("GlobalNodeID")
ug.GetPointData().AddArray(gid)
eid = ns.numpy_to_vtk(np.arange(1,len(tets)+1,dtype=np.int32), deep=True); eid.SetName("GlobalElementID")
ug.GetCellData().AddArray(eid)
rid = ns.numpy_to_vtk(np.ones(len(tets),dtype=np.int32), deep=True); rid.SetName("ModelRegionID")
ug.GetCellData().AddArray(rid)
w = vtk.vtkXMLUnstructuredGridWriter(); w.SetFileName(f"{OUT}/mesh-complete.mesh.vtu"); w.SetInputData(ug); w.SetDataModeToBinary(); w.SetCompressorTypeToNone(); w.Write()

FACE_ID = {"wall":1,"asc":2,"desc":3,"btca":4,"lcca":5,"lsa":6}
# global element ids for surface tris: continue after tets? svMP uses GlobalElementID of the ADJACENT tet normally (SimVascular convention: exterior faces carry parent element id). Build tet-face lookup.
from collections import defaultdict
face2tet = {}
for i,tt in enumerate(tets):
    for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)):
        key = tuple(sorted((tt[f[0]],tt[f[1]],tt[f[2]])))
        face2tet[key] = i+1
for name,tris in faces.items():
    used = np.unique(tris)
    lmap = np.zeros(used.max()+1,dtype=np.int64); lmap[used]=np.arange(len(used))
    p = vtk.vtkPoints(); p.SetData(ns.numpy_to_vtk(coords[used], deep=True))
    pdta = vtk.vtkPolyData(); pdta.SetPoints(p)
    ca2 = vtk.vtkCellArray()
    lt = lmap[tris]
    conn2 = np.hstack([np.full((len(lt),1),3,dtype=np.int64), lt]).ravel()
    ca2.SetCells(len(lt), ns.numpy_to_vtkIdTypeArray(conn2, deep=True))
    pdta.SetPolys(ca2)
    g2 = ns.numpy_to_vtk((used+1).astype(np.int32), deep=True); g2.SetName("GlobalNodeID")
    pdta.GetPointData().AddArray(g2)
    ge = np.array([face2tet.get(tuple(sorted(t)), 0) for t in tris], dtype=np.int32)
    if (ge==0).any(): print("WARN: %s has %d tris with no parent tet" % (name,(ge==0).sum()))
    a = ns.numpy_to_vtk(ge, deep=True); a.SetName("GlobalElementID"); pdta.GetCellData().AddArray(a)
    fi = ns.numpy_to_vtk(np.full(len(lt),FACE_ID[name],dtype=np.int32), deep=True); fi.SetName("ModelFaceID")
    pdta.GetCellData().AddArray(fi)
    ww = vtk.vtkXMLPolyDataWriter(); ww.SetFileName(f"{OUT}/mesh-surfaces/{name}.vtp"); ww.SetInputData(pdta); ww.SetDataModeToBinary(); ww.SetCompressorTypeToNone(); ww.Write()
    print("wrote", name, len(lt), "tris,", len(used), "pts")
print("DONE")
