import os
import nibabel as nib
import numpy as np
import cc3d


input_dir = r"C:\Users\user\Desktop\1130_3_result"
output_dir = os.path.join(input_dir, "post_processed")
dst_img = r"C:\Users\user\Desktop\1130_3_result\1130_3_upload"


os.makedirs(output_dir, exist_ok=True)
os.makedirs(dst_img, exist_ok=True)


for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".nii.gz"):
        continue

    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    print(f"{fname}")

    nii = nib.load(in_path)
    data = nii.get_fdata().astype(np.int32)

    cleaned_data = np.zeros_like(data)

    for label in np.unique(data):
        if label == 0:
            continue

        mask = (data == label)
        labeled, n = cc3d.connected_components(mask, connectivity=26, return_N=True)

        if n > 0:
            sizes = np.bincount(labeled.flatten())
            sizes[0] = 0
            largest_label = np.argmax(sizes)
            cleaned_data[labeled == largest_label] = label


    cleaned_nii = nib.Nifti1Image(cleaned_data.astype(np.uint8), nii.affine, nii.header)
    nib.save(cleaned_nii, out_path)

    print(f": {fname} -> {out_path}")


import os, shutil

os.makedirs(dst_img, exist_ok=True)

for i, fname in enumerate(sorted(os.listdir(output_dir))):
    src_path = os.path.join(output_dir, fname)

    if not os.path.isfile(src_path) or not fname.endswith(".nii.gz"):
        print(f"Skipping non-NIfTI item: {fname}")
        continue

    i = i + 51
    case_id = f"patient{i:04d}"
    dst_path = os.path.join(dst_img, f"{case_id}.nii.gz")

    shutil.copy(src_path, dst_path)
    print(f"Copied {fname} → {dst_path}")


