#!/usr/bin/env python3
"""
dicom_to_phases.py -- step 0 (optional) of the image-registration wall chain: split a
time-resolved DICOM series (one file per slice and trigger time, e.g. the anatomy/magnitude
series of a 4D-flow acquisition) into one 3D volume per cardiac phase.

Files are grouped by TriggerTime, sorted along the slice axis by ImagePositionPatient[2], and
written as phase<kk>.nii.gz in trigger-time order. Assumes axial slices with an axis-aligned
orientation (the registration and inversion steps work in the physical frame
origin + spacing * index); the slice spacing is taken from the slice positions.

Usage:
  python3 chain_image/dicom_to_phases.py <dicom_dir> <out_dir> [--glob '*.dcm']
Dependencies: pydicom, SimpleITK.
"""
import argparse
import collections
import glob
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dicom_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--glob", default="*.dcm")
    a = ap.parse_args()
    import pydicom
    import SimpleITK as sitk

    files = sorted(glob.glob(os.path.join(a.dicom_dir, a.glob)))
    if not files:
        raise SystemExit(f"no file matches {a.glob} in {a.dicom_dir}")
    groups = collections.defaultdict(list)
    for f in files:
        d = pydicom.dcmread(f, stop_before_pixels=True,
                            specific_tags=["TriggerTime", "ImagePositionPatient", "ImageOrientationPatient"])
        groups[float(d.TriggerTime)].append((float(d.ImagePositionPatient[2]), f))
    times = sorted(groups)
    counts = {len(v) for v in groups.values()}
    print(f"[dicom] {len(times)} phases, slices per phase: {sorted(counts)}")
    if len(counts) != 1:
        raise SystemExit("phases have different slice counts")
    ref = pydicom.dcmread(groups[times[0]][0][1], stop_before_pixels=True)
    iop = np.asarray(getattr(ref, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    if not np.allclose(np.abs(iop), [1, 0, 0, 0, 1, 0], atol=1e-3):
        print(f"[dicom] WARNING: non-axial orientation {iop}; the output assumes an identity direction")
    os.makedirs(a.out_dir, exist_ok=True)
    for i, tt in enumerate(times):
        slices = sorted(groups[tt])
        vol = np.stack([pydicom.dcmread(f).pixel_array for _, f in slices]).astype(np.int16)
        img = sitk.GetImageFromArray(vol)
        zs = [s[0] for s in slices]
        dz = (zs[-1] - zs[0]) / (len(zs) - 1)
        img.SetSpacing((float(ref.PixelSpacing[1]), float(ref.PixelSpacing[0]), dz))
        ipp = pydicom.dcmread(slices[0][1], stop_before_pixels=True).ImagePositionPatient
        img.SetOrigin((float(ipp[0]), float(ipp[1]), float(ipp[2])))
        sitk.WriteImage(img, os.path.join(a.out_dir, f"phase{i:02d}.nii.gz"), True)
        print(f"[dicom] phase {i:02d}: trigger {tt:.1f} ms, z0 {zs[0]:.2f} mm, dz {dz:.4f} mm")


if __name__ == "__main__":
    main()
