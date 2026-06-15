"""Measure the near-wall normal resolution of a tet mesh (first normal step off the wall)."""
import sys
import numpy as np, pyvista as pv, meshio
from scipy.spatial import cKDTree

ROOT = "."


def stats(path):
    m = meshio.read(path)
    X = m.points.astype(float)
    tet = m.cells_dict["tetra"].astype(np.int64)
    wall = pv.read(f"{ROOT}/mesh-surfaces/wall.vtp")
    wall = wall.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    wtree = cKDTree(np.asarray(wall.points))
    wn = np.asarray(wall.point_data["Normals"])
    d, j = wtree.query(X)
    e = np.vstack([tet[:, [0, 1]], tet[:, [0, 2]], tet[:, [0, 3]],
                   tet[:, [1, 2]], tet[:, [1, 3]], tet[:, [2, 3]]])
    e.sort(1); e = np.unique(e, axis=0)
    ev = X[e[:, 1]] - X[e[:, 0]]
    el = np.linalg.norm(ev, axis=1)
    nm = wn[j[e[:, 0]]]
    cosn = np.abs(np.einsum('ij,ij->i', ev, nm)) / np.maximum(el, 1e-12)
    # first normal step: wall nodes (d~0) -> near-normal edges leaving the wall
    onw = d[e[:, 0]] < 1e-6
    onw2 = d[e[:, 1]] < 1e-6
    first = (onw ^ onw2) & (cosn > 0.6)
    band = (0.5 * (d[e[:, 0]] + d[e[:, 1]]) < 1.0) & (cosn > 0.7)
    tang = (0.5 * (d[e[:, 0]] + d[e[:, 1]]) < 1.0) & (cosn < 0.3)
    print(f"{path.split('/')[-1]:24s} {len(X):7d} pts {len(tet):8d} tets | "
          f"first normal step med={np.median(el[first]):.3f} p10={np.percentile(el[first],10):.3f} | "
          f"band<1: normal med={np.median(el[band]):.3f} tangent med={np.median(el[tang]):.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stats(p)
