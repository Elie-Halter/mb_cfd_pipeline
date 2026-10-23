#!/usr/bin/env python3
"""
register_phases.py -- step 1 of the image-registration wall chain.

One phase of a time-resolved acquisition (e.g. the 4D-flow magnitude/anatomy images
reconstructed per cardiac phase) carries a lumen segmentation; every other phase is registered
to it by a B-spline deformable registration (elastix, via itk-elastix):

    fixed  = phase k image,   moving = reference-phase image
    -> transform T_k mapping phase-k coordinates to reference coordinates.

The transform parameter files are what step 2 (invert_propagate.py) inverts to carry the
reference wall nodes to every phase. As a by-product the reference segmentation is warped onto
every phase (nearest neighbour) and its volume reported; manual segmentations of other phases,
if available, are compared with the warped ones (Dice, mean / 95th-percentile / max surface
distance).

All phases are cropped to the same region: the bounding box of the reference segmentation plus
a margin. The crop is recorded in crop.json and reused by step 2.

Usage:
  python3 chain_image/register_phases.py --phases phase_*.nii.gz --ref-index 10 \
      --ref-seg ref_segmentation.nrrd --out reg/ [--validate 4:seg_phase4.nrrd ...]

Inputs: one 3D image per phase, in cycle order (any format SimpleITK reads); the reference
segmentation (non-zero = lumen), on the reference-phase grid or resampled onto it.
Outputs (in --out): tp_phase<kk>.txt (elastix transform parameters, one per non-reference
phase), segs/seg_phase<kk>.nii.gz, crop.json, results.json. Re-running skips finished phases.
Dependencies: itk-elastix, SimpleITK (pip install itk-elastix SimpleITK).
"""
import argparse
import glob
import json
import os
import sys
import time
import numpy as np


def read_seg(path, ref_img):
    import SimpleITK as sitk
    s = sitk.ReadImage(path)
    if s.GetNumberOfComponentsPerPixel() > 1:
        s = sitk.VectorIndexSelectionCast(s, 0)
    s = sitk.Cast(s != 0, sitk.sitkUInt8)
    if (s.GetSize() != ref_img.GetSize()
            or not np.allclose(s.GetSpacing(), ref_img.GetSpacing(), atol=1e-3)
            or not np.allclose(s.GetOrigin(), ref_img.GetOrigin(), atol=0.6)):
        s = sitk.Resample(s, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    else:
        s.CopyInformation(ref_img)
    return s


def crop_box(seg, ref_img, margin_mm):
    import SimpleITK as sitk
    f = sitk.LabelShapeStatisticsImageFilter(); f.Execute(seg)
    bb = f.GetBoundingBox(1)                       # x, y, z, sx, sy, sz (index units)
    sp = ref_img.GetSpacing()
    mg = [int(round(margin_mm / s)) for s in sp]
    lo = [max(0, bb[i] - mg[i]) for i in range(3)]
    hi = [min(ref_img.GetSize()[i], bb[i] + bb[3 + i] + mg[i]) for i in range(3)]
    return lo, [hi[i] - lo[i] for i in range(3)]


def to_itk(simg):
    import SimpleITK as sitk
    import itk
    img = itk.GetImageFromArray(sitk.GetArrayFromImage(simg))
    img.SetSpacing(simg.GetSpacing()); img.SetOrigin(simg.GetOrigin())
    return img


def dice(a, b):
    import SimpleITK as sitk
    a = sitk.GetArrayViewFromImage(a) > 0; b = sitk.GetArrayViewFromImage(b) > 0
    return float(2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum()))


def surface_distances(a, b):
    import SimpleITK as sitk
    da = sitk.SignedMaurerDistanceMap(a, squaredDistance=False, useImageSpacing=True)
    db = sitk.SignedMaurerDistanceMap(b, squaredDistance=False, useImageSpacing=True)
    ca, cb = sitk.LabelContour(a), sitk.LabelContour(b)
    d = np.abs(np.concatenate([sitk.GetArrayViewFromImage(db)[sitk.GetArrayViewFromImage(ca) > 0],
                               sitk.GetArrayViewFromImage(da)[sitk.GetArrayViewFromImage(cb) > 0]]))
    return dict(assd_mm=float(d.mean()), hd95_mm=float(np.percentile(d, 95)), hd_mm=float(d.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", nargs="+", required=True,
                    help="phase images in cycle order (a quoted glob is expanded and sorted)")
    ap.add_argument("--ref-index", type=int, required=True, help="index of the segmented phase")
    ap.add_argument("--ref-seg", required=True, help="segmentation of the reference phase")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-mm", type=float, default=25.0, help="crop margin around the lumen")
    ap.add_argument("--grid-spacing", type=float, default=12.0,
                    help="final B-spline control-point spacing (mm)")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--samples", type=int, default=4000, help="spatial samples per iteration")
    ap.add_argument("--validate", action="append", default=[], metavar="K:SEG",
                    help="manual segmentation of phase K to compare with the warped reference")
    a = ap.parse_args()

    import SimpleITK as sitk
    import itk

    files = []
    for p in a.phases:
        files += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    n = len(files)
    if not 0 <= a.ref_index < n:
        sys.exit(f"--ref-index {a.ref_index} outside 0..{n - 1}")
    os.makedirs(os.path.join(a.out, "segs"), exist_ok=True)
    phases = [sitk.ReadImage(f, sitk.sitkFloat32) for f in files]
    ref_img = phases[a.ref_index]
    if not np.allclose(np.asarray(ref_img.GetDirection()), np.eye(3).ravel(), atol=1e-6):
        print("[register] WARNING: non-identity image direction; invert_propagate.py assumes "
              "physical = origin + spacing * index (resample the phases to an axis-aligned grid first)")
    seg_ref = read_seg(a.ref_seg, ref_img)
    lo, sz = crop_box(seg_ref, ref_img, a.margin_mm)
    crop = lambda img: sitk.RegionOfInterest(img, sz, lo)     # noqa: E731
    json.dump(dict(lo=lo, size=sz, ref_index=a.ref_index, n_phases=n,
                   ref_image=os.path.abspath(files[a.ref_index]), margin_mm=a.margin_mm),
              open(os.path.join(a.out, "crop.json"), "w"), indent=1)
    print(f"[register] {n} phases, reference {a.ref_index}, crop index {lo} size {sz}", flush=True)

    pobj = itk.ParameterObject.New()
    pm = pobj.GetDefaultParameterMap("bspline", 3, a.grid_spacing)
    pm["MaximumNumberOfIterations"] = [str(a.iterations)]
    pm["NumberOfSpatialSamples"] = [str(a.samples)]
    pm["FinalBSplineInterpolationOrder"] = ["3"]
    pobj.AddParameterMap(pm)

    moving = to_itk(crop(ref_img))
    seg_crop = crop(seg_ref)
    seg_itk = to_itk(sitk.Cast(seg_crop, sitk.sitkFloat32))
    vox_ml = float(np.prod(ref_img.GetSpacing())) / 1000.0
    results = {}
    for k in range(n):
        seg_path = os.path.join(a.out, "segs", f"seg_phase{k:02d}.nii.gz")
        tp_path = os.path.join(a.out, f"tp_phase{k:02d}.txt")
        t0 = time.time()
        if os.path.exists(seg_path) and (k == a.ref_index or os.path.exists(tp_path)):
            warped = sitk.ReadImage(seg_path)
        elif k == a.ref_index:
            warped = seg_crop
            sitk.WriteImage(warped, seg_path, True)
        else:
            fixed = to_itk(crop(phases[k]))
            _, tparams = itk.elastix_registration_method(fixed, moving, parameter_object=pobj,
                                                          log_to_console=False)
            # nearest-neighbour resampling for the label map (does not change the transform)
            tparams.SetParameter(0, "FinalBSplineInterpolationOrder", "0")
            tparams.SetParameter(0, "ResampleInterpolator", "FinalNearestNeighborInterpolator")
            tparams.WriteParameterFile(tp_path)
            w = itk.transformix_filter(seg_itk, tparams)
            warped = sitk.GetImageFromArray((itk.GetArrayFromImage(w) > 0.5).astype(np.uint8))
            warped.CopyInformation(seg_crop)
            sitk.WriteImage(warped, seg_path, True)
        vol = float((sitk.GetArrayViewFromImage(warped) > 0).sum()) * vox_ml
        results[k] = dict(volume_ml=round(vol, 2), seconds=round(time.time() - t0, 1))
        print(f"[register] phase {k:02d}: warped-lumen volume {vol:.1f} mL ({results[k]['seconds']} s)", flush=True)

    validation = {}
    for spec in a.validate:
        k, path = spec.split(":", 1)
        k = int(k)
        manual = crop(read_seg(path, ref_img))
        warped = sitk.ReadImage(os.path.join(a.out, "segs", f"seg_phase{k:02d}.nii.gz"))
        warped.CopyInformation(manual)
        validation[k] = dict(dice=dice(warped, manual), **surface_distances(warped, manual),
                             dice_static_reference=dice(seg_crop, manual))
        print(f"[register] phase {k:02d} vs manual: {validation[k]}")
    json.dump(dict(phases=results, validation=validation),
              open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
