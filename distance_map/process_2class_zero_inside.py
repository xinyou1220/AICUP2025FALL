#!/usr/bin/env python3
"""
將 nnUNet 格式的 label 轉換為 distance map (Signed Distance Field)
為標籤 1 和標籤 2 分別生成獨立的 distance map
修改版：輪廓內部設為 0，外部仍為正數
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from tqdm import tqdm

def compute_sdf(segmentation):
    """
    計算 Signed Distance Field (SDF)
    修改版：內部設為 0，外部為正值
    
    Parameters:
    -----------
    segmentation : numpy.ndarray
        二值化的分割遮罩 (0 或 1)
    
    Returns:
    --------
    sdf : numpy.ndarray
        Signed distance field
        0：在物體內部（包含邊界）
        正值：在物體外部的距離
    """
    # 確保是二值化遮罩
    binary_mask = (segmentation > 0).astype(np.uint8)
    
    # 計算物體外部的距離 (正值)
    distance_outside = distance_transform_edt(1 - binary_mask)
    
    # 內部全部設為 0
    sdf = np.where(binary_mask > 0, 0, distance_outside)
    
    return sdf.astype(np.float32)


def normalize_sdf_clipped(sdf, clip_range=10, feature_range=(-1, 1)):
    """
    Clipped Normalization
    先裁剪到 [0, clip_range]，再歸一化到 [-1, 1]
    
    優點：
    - 聚焦在邊界區域（最重要的形狀資訊）
    - 不受遠離邊界的極值影響
    - 最常用於深度學習的形狀先驗
    
    參數：
    - clip_range: 裁剪範圍（像素），預設 10
    """
    # 裁剪到指定範圍 (現在只有正值，所以裁剪到 [0, clip_range])
    sdf_clipped = np.clip(sdf, 0, clip_range)
    
    # 歸一化到目標範圍
    min_val, max_val = feature_range
    sdf_normalized = sdf_clipped / clip_range  # 先歸一化到 [0, 1]
    
    # 轉換到 [-1, 1]
    sdf_normalized = sdf_normalized * 2 - 1  # [0, 1] -> [-1, 1]
    
    # 如果需要其他範圍
    if feature_range != (-1, 1):
        sdf_normalized = (sdf_normalized + 1) / 2  # 轉到 [0, 1]
        sdf_normalized = sdf_normalized * (max_val - min_val) + min_val
    
    return sdf_normalized.astype(np.float32)


def process_single_file(input_path, output_dir, clip_range=10):
    """
    處理單一 NIfTI 檔案，為標籤 1 和標籤 2 分別生成 distance map
    
    Parameters:
    -----------
    input_path : Path
        輸入的 label 檔案路徑
    output_dir : Path
        輸出目錄
    clip_range : int
        裁剪範圍
    """
    try:
        # 載入 label
        nii = nib.load(input_path)
        label_data = nii.get_fdata()
        
        # 獲取檔名（不含副檔名）
        stem = input_path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]
        
        # 檢查是否有標籤 1
        has_label1 = False
        if 1 in np.unique(label_data):
            has_label1 = True
            binary_mask_1 = (label_data == 1).astype(np.uint8)
            sdf_1 = compute_sdf(binary_mask_1)
            sdf_label1 = normalize_sdf_clipped(sdf_1, clip_range)
        else:
            # 如果沒有標籤 1，初始化為最大正值（表示離邊界最遠）
            sdf_label1 = np.ones_like(label_data, dtype=np.float32)
        
        # 檢查是否有標籤 2
        has_label2 = False
        if 2 in np.unique(label_data):
            has_label2 = True
            binary_mask_2 = (label_data == 2).astype(np.uint8)
            sdf_2 = compute_sdf(binary_mask_2)
            sdf_label2 = normalize_sdf_clipped(sdf_2, clip_range)
        else:
            # 如果沒有標籤 2，初始化為最大正值（表示離邊界最遠）
            sdf_label2 = np.ones_like(label_data, dtype=np.float32)
        
        if not has_label1 and not has_label2:
            print(f"警告: {input_path.name} 沒有標籤 1 或標籤 2，跳過處理")
            return None
        
        # 堆疊成兩個通道 (channel, depth, height, width)
        sdf_combined = np.stack([sdf_label1, sdf_label2], axis=0)
        
        # 生成輸出檔名
        output_file = output_dir / f"{stem}_sdf.npy"
        
        # 儲存為 .npy 檔案
        np.save(output_file, sdf_combined)
        
        # 返回統計資訊
        stats = {
            'file': input_path.name,
            'shape': sdf_combined.shape,
            'has_label1': has_label1,
            'has_label2': has_label2,
            'label1_min': float(sdf_label1.min()) if has_label1 else None,
            'label1_max': float(sdf_label1.max()) if has_label1 else None,
            'label2_min': float(sdf_label2.min()) if has_label2 else None,
            'label2_max': float(sdf_label2.max()) if has_label2 else None,
        }
        
        return stats
        
    except Exception as e:
        print(f"錯誤處理 {input_path.name}: {str(e)}")
        return None


def main():
    # 設定路徑
    LABEL_DIR = r"C:\Users\user\Desktop\labelsTr"
    OUTPUT_DIR = r"C:\Users\user\Desktop\labelsTr\sdfTr3"
    
    # 轉換為 Path 物件
    label_dir = Path(LABEL_DIR)
    output_dir = Path(OUTPUT_DIR)
    
    # 建立輸出資料夾
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Distance Map 生成程式 - 多標籤獨立版本（修改版）")
    print("=" * 70)
    print(f"\nLabel 資料夾: {label_dir}")
    print(f"輸出資料夾: {output_dir}\n")
    
    # 設定裁剪範圍
    clip_range = 10
    print(f"使用方法: Clipped Normalization（修改版）")
    print(f"裁剪範圍: 0 到 {clip_range} pixels")
    print(f"輸出範圍: [-1, 1]")
    print(f"輸出格式: 兩個通道 [標籤1, 標籤2]")
    print(f"特性: 輪廓內部=0，外部=正值\n")
    
    # 尋找所有 .nii.gz 檔案
    label_files = sorted(label_dir.glob("*.nii.gz"))
    
    if len(label_files) == 0:
        print(f"錯誤: 在 {label_dir} 中找不到 .nii.gz 檔案")
        return
    
    print(f"找到 {len(label_files)} 個檔案")
    print("開始處理...\n")
    
    # 處理每個檔案
    success_count = 0
    all_stats = []
    
    for label_file in tqdm(label_files, desc="處理進度"):
        stats = process_single_file(label_file, output_dir, clip_range)
        
        if stats:
            success_count += 1
            all_stats.append(stats)
    
    # 顯示統計資訊
    print(f"\n{'=' * 70}")
    print(f"完成! 成功處理 {success_count}/{len(label_files)} 個檔案")
    print(f"{'=' * 70}\n")
    
    if all_stats:
        label1_count = sum(1 for s in all_stats if s['has_label1'])
        label2_count = sum(1 for s in all_stats if s['has_label2'])
        
        print("整體統計：")
        print(f"  含有標籤 1 的檔案: {label1_count}/{len(all_stats)}")
        print(f"  含有標籤 2 的檔案: {label2_count}/{len(all_stats)}")
        
        # 標籤 1 統計
        if label1_count > 0:
            label1_mins = [s['label1_min'] for s in all_stats if s['has_label1']]
            label1_maxs = [s['label1_max'] for s in all_stats if s['has_label1']]
            print(f"\n  標籤 1 數值範圍: [{min(label1_mins):.4f}, {max(label1_maxs):.4f}]")
        
        # 標籤 2 統計
        if label2_count > 0:
            label2_mins = [s['label2_min'] for s in all_stats if s['has_label2']]
            label2_maxs = [s['label2_max'] for s in all_stats if s['has_label2']]
            print(f"  標籤 2 數值範圍: [{min(label2_mins):.4f}, {max(label2_maxs):.4f}]")
    
    print(f"\nDistance maps 已儲存至: {output_dir}")
    
    # 顯示使用建議
    print("\n" + "=" * 70)
    print("使用說明：")
    print("=" * 70)
    print("✓ 每個 .npy 檔案包含兩個通道的 distance map")
    print("  - 通道 0: 標籤 1 的 distance map")
    print("  - 通道 1: 標籤 2 的 distance map")
    print("  - 形狀: (2, depth, height, width)")
    print("\n✓ 使用方式:")
    print("  sdf = np.load('file_sdf.npy')")
    print("  sdf_label1 = sdf[0]  # 標籤 1 的 distance map")
    print("  sdf_label2 = sdf[1]  # 標籤 2 的 distance map")
    print("\n✓ 數值範圍: [-1, 1]（修改版）")
    print("  - -1 表示在物體內部（原本的 0）")
    print("  - 1 表示離邊界最遠（clip_range 以外）")
    print("  - 介於兩者之間表示距離邊界的遠近")


if __name__ == "__main__":
    # 檢查必要套件
    try:
        import nibabel
        import scipy
        from tqdm import tqdm
    except ImportError as e:
        print(f"缺少必要套件: {e}")
        print("\n請執行以下指令安裝:")
        print("pip install nibabel scipy tqdm")
        exit(1)
    
    main()
