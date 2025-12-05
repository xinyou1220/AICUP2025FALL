import nibabel as nib
import numpy as np
import os

seg_dir   = r"D:\1_2025_fall_ai_cup\train_dataset\dilate_label"
ori_dir   = r"D:\1_2025_fall_ai_cup\train_dataset\train_dataset"
label_dir = r"D:\1_2025_fall_ai_cup\train_dataset\train_label"

output_data_dir  = r"D:\1_2025_fall_ai_cup\train_dataset\for_xc_test\pre_dataset"
output_label_dir = r"D:\1_2025_fall_ai_cup\train_dataset\for_xc_test\pre_label"

os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

num_cases = 50
pad = 0


for i in range(0, num_cases + 1):
    case_id = f"patient{i:04d}"
    print(f"process: {case_id}")

    seg_path   = os.path.join(seg_dir,   f"{i}_dilate.nii.gz")
    ori_path   = os.path.join(ori_dir,   f"{case_id}.nii.gz")
    label_path = os.path.join(label_dir, f"{case_id}_gt.nii.gz")

    seg_data   = nib.load(seg_path).get_fdata()
    ori_img    = nib.load(ori_path)
    ori_data   = ori_img.get_fdata()
    label_img  = nib.load(label_path)
    label_data = label_img.get_fdata()

    coords = np.array(np.nonzero(seg_data))

    minz, miny, minx = coords.min(axis=1)
    maxz, maxy_, maxx_ = coords.max(axis=1)

    minz = max(minz - pad, 0)
    miny = max(miny - pad, 0)
    minx = max(minx - pad, 0)
    maxz = min(maxz + pad, ori_data.shape[0] - 1)
    maxy_ = min(maxy_ + pad, ori_data.shape[1] - 1)
    maxx_ = min(maxx_ + pad, ori_data.shape[2] - 1)

    cropped_ori   = ori_data[minz:maxz+1, miny:maxy_+1, minx:maxx_+1]
    cropped_label = label_data[minz:maxz+1, miny:maxy_+1, minx:maxx_+1]

    # z, y, x = cropped_ori.shape
    # square_size = max(y, x)
    # pad_y = max(0, (square_size - y) // 2)
    # pad_x = max(0, (square_size - x) // 2)

    # cropped_ori   = np.pad(cropped_ori,   ((0, 0), (pad_y, square_size - y - pad_y), (pad_x, square_size - x - pad_x)), mode='constant')
    # cropped_label = np.pad(cropped_label, ((0, 0), (pad_y, square_size - y - pad_y), (pad_x, square_size - x - pad_x)), mode='constant')

    out_data_path  = os.path.join(output_data_dir,  f"pre_{case_id}.nii.gz")
    out_label_path = os.path.join(output_label_dir, f"{case_id}_gt.nii.gz")

    nib.save(nib.Nifti1Image(cropped_ori, ori_img.affine, ori_img.header), out_data_path)
    nib.save(nib.Nifti1Image(cropped_label, label_img.affine, label_img.header), out_label_path)

