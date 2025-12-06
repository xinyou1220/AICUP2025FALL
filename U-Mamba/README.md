# AICUP2025FALL

本專案為 AICUP2025FALL 競賽專案，基於 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 框架構建，實現了多種深度學習方法用於生物醫學圖像分割任務。

## 包含的方法

本專案實現和使用了多種方法，包括但不限於：

- **[U-Mamba](https://wanglab.ai/u-mamba.html)**: 使用 Mamba 模塊增強長距離依賴建模的生物醫學圖像分割方法
- **SAM-based Methods**: 基於 Segment Anything Model 的方法
- **Distance-based Methods**: 基於距離的方法

> 📖 **詳細的 Trainer 使用指南**: 請參閱 [README.md](README.md) 以了解 Trainer 架構、如何創建自定義 Trainer 以及訓練流程的詳細說明。

## Installation 

Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n umamba python=3.10 -y` and `conda activate umamba `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. Download code: `git clone https://github.com/bowang-lab/U-Mamba`
5. `cd U-Mamba/umamba` and run `pip install -e .`


sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```

![network](https://github.com/bowang-lab/U-Mamba/blob/main/assets/U-Mamba-network.png)



https://github.com/bowang-lab/U-Mamba/assets/19947331/1ac552d6-4ffd-4909-ba31-7b48644fd104




## Model Training

本專案基於 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 框架構建。如果您想在自己的數據集上訓練模型，請遵循此 [指南](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) 來準備數據集。 

### Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### 訓練範例

#### U-Mamba 方法

**2D 模型**:
- Train 2D `U-Mamba_Bot` model:
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot
```

- Train 2D `U-Mamba_Enc` model:
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaEnc
```

**3D 模型**:
- Train 3D `U-Mamba_Bot` model:
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaBot
```

- Train 3D `U-Mamba_Enc` model:
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaEnc
```

#### 其他方法

更多訓練範例請參閱 [README.md](README.md) 中的「使用範例」章節。


## Inference

### U-Mamba 方法推理範例

- Predict testing cases with `U-Mamba_Bot` model:
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta
```

- Predict testing cases with `U-Mamba_Enc` model:
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaEnc --disable_tta
```

> `CONFIGURATION` 可以是 `2d` 和 `3d_fullres`，分別對應 2D 和 3D 模型。

### 其他方法

其他方法的推理命令類似，只需將 `-tr` 參數替換為對應的 Trainer 名稱即可。可用的 Trainer 列表請參閱 [TRAINER_README.md](TRAINER_README.md)。

## Remarks

### 1. 路徑設置

默認數據目錄設置在專案根目錄下的 `data` 文件夾。如果您想使用其他目錄，可以在 `umamba/nnunetv2/path.py` 中調整以下路徑：

```python
# 設置其他數據路徑的範例
base = '/home/user_name/Documents/AICUP2025FALL/data'
nnUNet_raw = join(base, 'nnUNet_raw') # 或改為 os.environ.get('nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # 或改為 os.environ.get('nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results') # 或改為 os.environ.get('nnUNet_results')
```

### 2. U-Mamba 相關注意事項

- **AMP 問題**: 使用自動混合精度 (AMP) 時，Mamba 模塊可能導致 NaN 值。我們提供了無 AMP 版本的 Trainer (`nnUNetTrainerUMambaEncNoAMP`)，可以在訓練時使用以避免此問題。

### 3. 其他方法的使用

不同方法可能有不同的配置要求，請參閱 [TRAINER_README.md](TRAINER_README.md) 了解各方法的詳細說明。

## 參考文獻

### U-Mamba

```
@article{U-Mamba,
    title={U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation},
    author={Ma, Jun and Li, Feifei and Wang, Bo},
    journal={arXiv preprint arXiv:2401.04722},
    year={2024}
}
```

### nnU-Net

```
@article{Isensee2021nnUNet,
    title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
    author={Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
    journal={Nature methods},
    volume={18},
    number={2},
    pages={203--211},
    year={2021}
}
```


## Acknowledgements

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [Mamba](https://github.com/state-spaces/mamba) for making their valuable code publicly available.

