# AICUP2025FALL 完整使用指南

本文檔詳細介紹 AICUP2025FALL 專案的完整工作流程，包括數據預處理、距離圖生成、模型訓練和後處理。本專案基於 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 框架構建，提供了靈活的訓練器系統，支援多種深度學習方法，包括但不限於：

- **U-Mamba**: 使用 Mamba 模塊增強長距離依賴建模
- **SAM-based Methods**: 基於 Segment Anything Model 的方法
- **Distance-based Methods**: 基於距離的方法

## 完整工作流程

```
原始數據
    ↓
[1. 預處理 (Preprocess)]
    ├── TotalSegmentator 分割 (totalseg.py)
    └── ROI 裁剪 (roi.py)
    ↓
[2. 距離圖生成 (Distance Map)]
    ├── 標準版本 (process_2class.py)
    └── 零內部版本 (process_2class_zero_inside.py)
    ↓
[3. 模型訓練 (Training)]
    └── 使用 nnUNet Trainer
    ↓
[4. 後處理 (Postprocess)]
    └── 連通組件分析 (postprocesscode.py)
    ↓
最終結果
```

## 目錄

- [1. 預處理 (Preprocess)](#1-預處理-preprocess)
  - [1.1 TotalSegmentator 分割](#11-totalsegmentator-分割)
  - [1.2 ROI 裁剪](#12-roi-裁剪)
- [2. 距離圖生成 (Distance Map)](#2-距離圖生成-distance-map)
  - [2.1 標準距離圖生成](#21-標準距離圖生成)
  - [2.2 零內部距離圖生成](#22-零內部距離圖生成)
- [3. 模型訓練 (Training)](#3-模型訓練-training)
  - [Trainer 架構概述](#trainer-架構概述)
  - [核心 Trainer 類別](#核心-trainer-類別)
  - [Trainer 週期](#trainer-週期)
  - [創建自定義 Trainer](#創建自定義-trainer)
  - [可用的 Trainer 變體](#可用的-trainer-變體)
  - [訓練流程詳解](#訓練流程詳解)
- [4. 後處理 (Postprocess)](#4-後處理-postprocess)
- [常見問題與解決方案](#常見問題與解決方案)

---

## 1. 預處理 (Preprocess)

預處理階段用於準備訓練數據，包括 ROI（感興趣區域）裁剪和使用 TotalSegmentator 進行自動分割。

### 1.1 TotalSegmentator 分割

`preprocess/totalseg.py` 使用 TotalSegmentator 進行自動分割，提取特定 ROI（如心臟）。

#### 功能說明

- 使用 TotalSegmentator 進行全身分割
- 提取特定 ROI（如 heart）
- 生成分割遮罩

#### 使用方法

1. **安裝 TotalSegmentator**：

```bash
pip install TotalSegmentator
```

2. **修改路徑配置**：

```python
inp = r"C:\Users\coffee\test.nii.gz"  # 輸入圖像路徑
out = r"C:\Users\coffee\testmask.nii.gz"  # 輸出遮罩路徑
```

3. **設定參數**：

```python
totalsegmentator(
    input=inp,
    output=out,
    task="total",          # 全身任務
    roi_subset=["heart"],  # 只取 heart（可修改為其他 ROI）
    fast=False,            # 高品質模式（False）或快速模式（True）
    preview=False          # 是否生成預覽圖
)
```

4. **執行腳本**：

```bash
python preprocess/totalseg.py
```

#### 可用的 ROI 選項

TotalSegmentator 支援多種 ROI，常見選項包括：
- `["heart"]`: 心臟
- `["lung"]`: 肺部
- `["liver"]`: 肝臟
- `["kidney"]`: 腎臟
- 更多選項請參考 [TotalSegmentator 文檔](https://github.com/wasserth/TotalSegmentator)

#### 注意事項

- TotalSegmentator 需要較長的處理時間，建議使用 GPU 加速
- `fast=True` 模式速度較快但精度較低
- 確保輸入圖像格式為 NIfTI（.nii.gz）

---
### 1.2 ROI 裁剪

`preprocess/roi.py` 用於根據擴張後的標籤（dilate label）對原始圖像和標籤進行 ROI 裁剪。

#### 功能說明

- 讀取擴張後的標籤文件（dilate label）
- 根據標籤的非零區域計算邊界框
- 對原始圖像和標籤進行裁剪
- 可選的邊界填充（padding）

#### 使用方法

1. **修改路徑配置**：

```python
seg_dir   = r"D:\1_2025_fall_ai_cup\train_dataset\dilate_label"  # 擴張標籤目錄
ori_dir   = r"D:\1_2025_fall_ai_cup\train_dataset\train_dataset"  # 原始圖像目錄
label_dir = r"D:\1_2025_fall_ai_cup\train_dataset\train_label"   # 原始標籤目錄

output_data_dir  = r"D:\1_2025_fall_ai_cup\train_dataset\for_xc_test\pre_dataset"   # 輸出圖像目錄
output_label_dir = r"D:\1_2025_fall_ai_cup\train_dataset\for_xc_test\pre_label"      # 輸出標籤目錄
```

2. **設定參數**：

```python
num_cases = 50  # 處理的案例數量
pad = 0         # 邊界填充像素數
```

3. **執行腳本**：

```bash
python preprocess/roi.py
```

#### 輸出格式

- 輸出圖像：`pre_{case_id}.nii.gz`
- 輸出標籤：`{case_id}_gt.nii.gz`

#### 注意事項

- 確保所有輸入目錄存在且包含對應的文件
- 文件命名格式需符合：`{i}_dilate.nii.gz`、`patient{i:04d}.nii.gz`、`patient{i:04d}_gt.nii.gz`
- 輸出目錄會自動創建

---


## 2. 距離圖生成 (Distance Map)

距離圖生成階段將標籤轉換為 Signed Distance Field (SDF)，用於提供形狀先驗信息。

### 2.1 標準距離圖生成

`distance_map/process_2class.py` 為標籤 1 和標籤 2 分別生成獨立的 distance map。

#### 功能說明

- 計算 Signed Distance Field (SDF)
- 為標籤 1 和標籤 2 分別生成獨立的 distance map
- 使用 Clipped Normalization 進行歸一化
- 輸出範圍：[-1, 1]
  - 負值：在物體內部
  - 正值：在物體外部
  - 0：在邊界上

#### 使用方法

1. **修改路徑配置**：

```python
LABEL_DIR = r"C:\Users\user\Desktop\labelsTr"           # 輸入標籤目錄
OUTPUT_DIR = r"C:\Users\user\Desktop\labelsTr\sdfTr2"   # 輸出目錄
```

2. **設定參數**：

```python
clip_range = 10  # 裁剪範圍（像素），預設 10
```

3. **執行腳本**：

```bash
python distance_map/process_2class.py
```

#### 輸出格式

- 輸出文件：`{stem}_sdf.npy`
- 數據格式：NumPy 數組
- 形狀：`(2, depth, height, width)`
  - 通道 0：標籤 1 的 distance map
  - 通道 1：標籤 2 的 distance map

#### 使用範例

```python
import numpy as np

# 載入 distance map
sdf = np.load('file_sdf.npy')

# 提取標籤 1 和標籤 2 的 distance map
sdf_label1 = sdf[0]  # 標籤 1 的 distance map
sdf_label2 = sdf[1]  # 標籤 2 的 distance map
```

#### 注意事項

- 需要安裝：`nibabel`、`scipy`、`tqdm`
- 輸入文件需為 `.nii.gz` 格式
- 如果標籤中沒有標籤 1 或標籤 2，會初始化為最大正值

---

### 2.2 零內部距離圖生成

`distance_map/process_2class_zero_inside.py` 是標準版本的修改版，將輪廓內部設為 0，外部仍為正數。

#### 功能說明

- 計算 Signed Distance Field (SDF)
- **修改特性**：輪廓內部設為 0，外部為正值
- 使用 Clipped Normalization 進行歸一化
- 輸出範圍：[-1, 1]
  - -1：在物體內部（原本的 0）
  - 1：離邊界最遠（clip_range 以外）
  - 介於兩者之間：距離邊界的遠近

#### 使用方法

1. **修改路徑配置**：

```python
LABEL_DIR = r"C:\Users\user\Desktop\labelsTr"           # 輸入標籤目錄
OUTPUT_DIR = r"C:\Users\user\Desktop\labelsTr\sdfTr3"   # 輸出目錄
```

2. **設定參數**：

```python
clip_range = 10  # 裁剪範圍（像素），預設 10
```

3. **執行腳本**：

```bash
python distance_map/process_2class_zero_inside.py
```

#### 輸出格式

- 輸出文件：`{stem}_sdf.npy`
- 數據格式：NumPy 數組
- 形狀：`(2, depth, height, width)`
  - 通道 0：標籤 1 的 distance map
  - 通道 1：標籤 2 的 distance map

#### 與標準版本的差異

| 特性 | 標準版本 | 零內部版本 |
|------|---------|-----------|
| 內部值 | 負值（距離邊界的距離） | -1（固定值） |
| 外部值 | 正值（距離邊界的距離） | 正值（距離邊界的距離） |
| 邊界 | 0 | 接近 -1 |

#### 注意事項

- 需要安裝：`nibabel`、`scipy`、`tqdm`
- 輸入文件需為 `.nii.gz` 格式
- 適用於需要明確區分內部/外部區域的應用場景

---

## 3. 模型訓練 (Training)

### Trainer 架構概述

### 基礎架構

AICUP2025FALL 專案的訓練系統圍繞 `nnUNetTrainer` 類別構建，這是所有訓練器的基類。它提供了完整的訓練流程管理，包括：

- **模型初始化與配置**
- **數據加載與預處理**
- **訓練循環管理**
- **驗證與評估**
- **檢查點保存與恢復**
- **分佈式訓練支持 (DDP)**

### 核心組件

```
nnUNetTrainer
├── 網絡架構 (Network Architecture)
├── 優化器配置 (Optimizer Configuration)
├── 損失函數 (Loss Function)
├── 數據加載器 (Data Loaders)
├── 學習率調度器 (Learning Rate Scheduler)
└── 日誌記錄系統 (Logging System)
```

---

## 核心 Trainer 類別

### nnUNetTrainer

`nnUNetTrainer` 是所有訓練器的基類，位於 `umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`。

#### 初始化參數

```python
class nnUNetTrainer(object):
    def __init__(self, 
                 plans: dict,                    # 實驗計劃配置
                 configuration: str,              # 配置名稱 (如 '2d', '3d_fullres')
                 fold: int,                       # 交叉驗證折數
                 dataset_json: dict,              # 數據集 JSON 配置
                 unpack_dataset: bool = True,     # 是否解壓數據集
                 device: torch.device = torch.device('cuda'))  # 計算設備
```

#### 關鍵屬性

| 屬性 | 說明 | 默認值 |
|------|------|--------|
| `initial_lr` | 初始學習率 | `5e-4` |
| `weight_decay` | 權重衰減係數 | `3e-6` |
| `num_epochs` | 訓練輪數 | `1000` |
| `batch_size` | 批次大小 | 從配置管理器獲取 |
| `enable_deep_supervision` | 是否啟用深度監督 | `True` |
| `oversample_foreground_percent` | 前景過採樣比例 | `0.33` |
| `num_iterations_per_epoch` | 每個 epoch 的迭代次數 | `250` |
| `num_val_iterations_per_epoch` | 每個 epoch 的驗證迭代次數 | `50` |

#### 核心方法

##### 1. `initialize()`

初始化網絡、優化器、損失函數等組件。

```python
def initialize(self):
    # 確定輸入通道數
    self.num_input_channels = determine_num_input_channels(...)
    
    # 構建網絡架構
    self.network = self.build_network_architecture(...)
    
    # 配置優化器和學習率調度器
    self.optimizer, self.lr_scheduler = self.configure_optimizers()
    
    # 構建損失函數
    self.loss = self._build_loss()
```

##### 2. `build_network_architecture()`

**靜態方法**，用於構建網絡架構。子類應該重寫此方法以實現自定義架構。

```python
@staticmethod
def build_network_architecture(plans_manager: PlansManager,
                               dataset_json,
                               configuration_manager: ConfigurationManager,
                               num_input_channels,
                               enable_deep_supervision: bool = True) -> nn.Module:
    """
    構建網絡架構
    
    參數:
        plans_manager: 計劃管理器
        dataset_json: 數據集配置
        configuration_manager: 配置管理器
        num_input_channels: 輸入通道數
        enable_deep_supervision: 是否啟用深度監督
    
    返回:
        nn.Module: 構建好的網絡模型
    """
    return get_network_from_plans(...)
```

##### 3. `train_step()`

執行單個訓練步驟。

```python
def train_step(self, batch: dict) -> dict:
    """
    訓練步驟
    
    參數:
        batch: 包含 'data' 和 'target' 的批次字典
    
    返回:
        dict: 包含 'loss' 的字典
    """
    data = batch['data'].to(self.device)
    target = batch['target'].to(self.device)
    
    self.optimizer.zero_grad()
    
    # 使用自動混合精度 (AMP)
    with autocast(self.device.type, enabled=True):
        output = self.network(data)
        loss = self.loss(output, target)
    
    # 反向傳播
    if self.grad_scaler is not None:
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
    else:
        loss.backward()
        self.optimizer.step()
    
    return {'loss': loss.detach().cpu().numpy()}
```

##### 4. `validation_step()`

執行單個驗證步驟。

```python
def validation_step(self, batch: dict) -> dict:
    """
    驗證步驟
    
    返回:
        dict: 包含 'loss', 'tp_hard', 'fp_hard', 'fn_hard' 的字典
    """
    # 計算損失和 Dice 係數相關指標
    ...
    return {'loss': ..., 'tp_hard': ..., 'fp_hard': ..., 'fn_hard': ...}
```

##### 5. `configure_optimizers()`

配置優化器和學習率調度器。

```python
def configure_optimizers(self):
    """
    配置優化器和學習率調度器
    
    返回:
        tuple: (optimizer, lr_scheduler)
    """
    optimizer = torch.optim.SGD(
        self.network.parameters(), 
        self.initial_lr, 
        weight_decay=self.weight_decay,
        momentum=0.99, 
        nesterov=True
    )
    lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
    return optimizer, lr_scheduler
```

##### 6. `run_training()`

主訓練循環。

```python
def run_training(self):
    """
    執行完整的訓練流程
    """
    self.on_train_start()
    
    for epoch in range(self.current_epoch, self.num_epochs):
        self.on_epoch_start()
        
        # 訓練階段
        self.on_train_epoch_start()
        train_outputs = []
        for batch_id in range(self.num_iterations_per_epoch):
            train_outputs.append(self.train_step(next(self.dataloader_train)))
        self.on_train_epoch_end(train_outputs)
        
        # 驗證階段
        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(self.num_val_iterations_per_epoch):
                val_outputs.append(self.validation_step(next(self.dataloader_val)))
            self.on_validation_epoch_end(val_outputs)
        
        self.on_epoch_end()
    
    self.on_train_end()
```

---

## Trainer 週期

### 訓練流程圖

```
初始化 (__init__)
    ↓
on_train_start()
    ├── initialize()          # 初始化網絡、優化器等
    ├── get_dataloaders()     # 獲取數據加載器
    └── print_plans()         # 打印配置信息
    ↓
for epoch in range(num_epochs):
    ├── on_epoch_start()
    │
    ├── 訓練階段
    │   ├── on_train_epoch_start()
    │   ├── train_step() × num_iterations_per_epoch
    │   └── on_train_epoch_end()
    │
    ├── 驗證階段
    │   ├── on_validation_epoch_start()
    │   ├── validation_step() × num_val_iterations_per_epoch
    │   └── on_validation_epoch_end()
    │
    └── on_epoch_end()
        ├── 保存檢查點
        └── 記錄指標
    ↓
on_train_end()
    └── 保存最終檢查點
```

### 回調方法說明

| 方法 | 調用時機 | 主要功能 |
|------|----------|----------|
| `on_train_start()` | 訓練開始前 | 初始化組件、準備數據加載器 |
| `on_train_epoch_start()` | 每個 epoch 開始 | 設置網絡為訓練模式、更新學習率 |
| `on_train_epoch_end()` | 每個 epoch 訓練結束 | 聚合訓練損失、記錄日誌 |
| `on_validation_epoch_start()` | 每個 epoch 驗證開始 | 設置網絡為評估模式 |
| `on_validation_epoch_end()` | 每個 epoch 驗證結束 | 計算 Dice 係數、記錄指標 |
| `on_epoch_end()` | 每個 epoch 完全結束 | 保存檢查點、更新最佳模型 |
| `on_train_end()` | 訓練完全結束 | 保存最終檢查點、清理資源 |

---

## 創建自定義 Trainer

### 基本範例：U-Mamba Encoder Trainer

以下是一個創建自定義 U-Mamba Encoder Trainer 的範例（U-Mamba 是本專案中實現的方法之一）：

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import nn

class nnUNetTrainerUMambaEnc(nnUNetTrainer):
    """
    U-Mamba Encoder Trainer
    
    在編碼器部分使用 Mamba 模塊來增強長距離依賴建模能力
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # 自定義超參數
        self.initial_lr = 1e-4
        self.weight_decay = 1e-4
        
        # 注意：如果使用 Mamba 模塊，可能需要禁用 AMP
        # self.grad_scaler = None  # 取消註釋以禁用 AMP
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        構建 U-Mamba Encoder 網絡架構
        
        這裡應該實現您的 U-Mamba Encoder 架構
        """
        from your_module import UMambaEncoder  # 假設的模塊
        
        label_manager = plans_manager.get_label_manager(dataset_json)
        
        model = UMambaEncoder(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            patch_size=configuration_manager.patch_size,
            enable_deep_supervision=enable_deep_supervision
        )
        
        return model
    
    def configure_optimizers(self):
        """
        可選：自定義優化器配置
        """
        from torch.optim import AdamW
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        
        lr_scheduler = PolyLRScheduler(
            optimizer,
            self.initial_lr,
            self.num_epochs,
            exponent=1.0
        )
        
        return optimizer, lr_scheduler
    
    def train_step(self, batch: dict) -> dict:
        """
        可選：自定義訓練步驟
        
        例如，如果禁用 AMP，可以這樣實現：
        """
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # 不使用 AMP（如果 grad_scaler 為 None）
        output = self.network(data)
        loss = self.loss(output, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        return {'loss': loss.detach().cpu().numpy()}
```

### 使用無 AMP 的版本

如果 Mamba 模塊在使用 AMP 時出現 NaN 問題，可以使用無 AMP 版本：

```python
class nnUNetTrainerUMambaEncNoAMP(nnUNetTrainerUMambaEnc):
    """
    U-Mamba Encoder Trainer (無 AMP 版本)
    
    禁用自動混合精度以避免 Mamba 模塊中的 NaN 問題
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # 禁用 AMP
        self.grad_scaler = None
```

### 繼承自其他 Trainer

您也可以繼承自其他現有的 Trainer：

```python
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision

class nnUNetTrainerUMambaBot(nnUNetTrainerNoDeepSupervision):
    """
    U-Mamba Bottleneck Trainer
    
    在 Bottleneck 部分使用 Mamba 模塊，並禁用深度監督
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # 自定義配置
        self.initial_lr = 1e-4
        self.weight_decay = 1e-4
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        """
        構建 U-Mamba Bottleneck 網絡架構
        """
        # 實現您的架構
        ...
```

---

## 可用的 Trainer 變體

### 網絡架構變體

位於 `umamba/nnunetv2/training/nnUNetTrainer/variants/network_architecture/`：

- **`nnUNetTrainerNoDeepSupervision`**: 禁用深度監督的基類
- **`nnUNetTrainerBN`**: 使用 BatchNorm 的變體

### 損失函數變體

位於 `umamba/nnunetv2/training/nnUNetTrainer/variants/loss/`：

- **`nnUNetTrainerCELoss`**: 僅使用交叉熵損失
- **`nnUNetTrainerDiceLoss`**: 僅使用 Dice 損失
- **`nnUNetTrainerTopkLoss`**: 使用 TopK 損失

### 優化器變體

位於 `umamba/nnunetv2/training/nnUNetTrainer/variants/optimizer/`：

- **`nnUNetTrainerAdam`**: 使用 Adam 優化器
- **`nnUNetTrainerAdan`**: 使用 Adan 優化器

### 數據增強變體

位於 `umamba/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/`：

- **`nnUNetTrainerNoDA`**: 無數據增強
- **`nnUNetTrainerNoMirroring`**: 無鏡像增強
- **`nnUNetTrainerDA5`**: 增強版數據增強

### 學習率調度變體

位於 `umamba/nnunetv2/training/nnUNetTrainer/variants/lr_schedule/`：

- **`nnUNetTrainerCosAnneal`**: 使用餘弦退火調度器

### AICUP2025FALL 專案中的其他 Trainer

本專案包含多種實現的方法，對應的 Trainer 包括：

- **`nnUNetTrainerUNETR`**: UNETR 架構的 Trainer
- **`nnUNetTrainerSwinUNETR`**: Swin UNETR 架構的 Trainer
- **`nnUNetTrainerSegResNet`**: SegResNet 架構的 Trainer
- **`nnUNetTrainerNNSAM`**: 基於 SAM (Segment Anything Model) 的 Trainer
- **`nnUNetTrainerNNSAMdis`**: SAM + Distance-based 的 Trainer
- **`nnUNetTrainerNNSAMshape`**: SAM + Shape Prior 的 Trainer
- **`nnUNetTrainerNNdis`**: Distance-based 的 Trainer
---

## 訓練流程詳解

### 1. 數據加載

Trainer 使用 `get_dataloaders()` 方法獲取訓練和驗證數據加載器：

```python
def get_dataloaders(self):
    """
    獲取訓練和驗證數據加載器
    
    返回:
        tuple: (dataloader_train, dataloader_val)
    """
    # 獲取訓練和驗證案例標識符
    tr_keys, val_keys = self.do_split()
    
    # 創建數據集
    dataset_tr = nnUNetDataset(...)
    dataset_val = nnUNetDataset(...)
    
    # 創建數據加載器
    if len(self.configuration_manager.patch_size) == 2:
        dataloader_train = nnUNetDataLoader2D(...)
        dataloader_val = nnUNetDataLoader2D(...)
    else:
        dataloader_train = nnUNetDataLoader3D(...)
        dataloader_val = nnUNetDataLoader3D(...)
    
    return dataloader_train, dataloader_val
```

### 2. 損失函數構建

`_build_loss()` 方法根據標籤類型構建適當的損失函數：

```python
def _build_loss(self):
    """
    構建損失函數
    
    根據標籤類型選擇：
    - 區域訓練：DC_and_BCE_loss (Dice + Binary Cross Entropy)
    - 常規訓練：DC_and_CE_loss (Dice + Cross Entropy)
    """
    if self.label_manager.has_regions:
        loss = DC_and_BCE_loss(...)
    else:
        loss = DC_and_CE_loss(...)
    
    # 如果啟用深度監督，包裝損失函數
    if self.enable_deep_supervision:
        loss = DeepSupervisionWrapper(loss, weights)
    
    return loss
```

### 3. 檢查點管理

#### 保存檢查點

```python
def save_checkpoint(self, filename: str):
    """
    保存訓練檢查點
    
    包含：
    - 網絡權重
    - 優化器狀態
    - 梯度縮放器狀態（如果使用 AMP）
    - 日誌信息
    - 當前 epoch
    - 初始化參數
    """
    checkpoint = {
        'network_weights': self.network.state_dict(),
        'optimizer_state': self.optimizer.state_dict(),
        'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler else None,
        'logging': self.logger.get_checkpoint(),
        '_best_ema': self._best_ema,
        'current_epoch': self.current_epoch + 1,
        'init_args': self.my_init_kwargs,
        'trainer_name': self.__class__.__name__,
        'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
    }
    torch.save(checkpoint, filename)
```

#### 加載檢查點

```python
def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]):
    """
    加載訓練檢查點
    
    用於：
    - 繼續訓練 (--c 參數)
    - 僅運行驗證 (--val 參數)
    - 加載預訓練權重
    """
    checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
    
    # 加載網絡權重
    self.network.load_state_dict(checkpoint['network_weights'])
    
    # 加載優化器狀態
    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # 恢復訓練狀態
    self.current_epoch = checkpoint['current_epoch']
    self.logger.load_checkpoint(checkpoint['logging'])
    self._best_ema = checkpoint['_best_ema']
```

### 4. 驗證與評估

`perform_actual_validation()` 方法執行完整的驗證流程：

```python
def perform_actual_validation(self, save_probabilities: bool = False):
    """
    執行實際驗證
    
    使用滑動窗口預測對驗證集進行預測，
    並計算 Dice 係數等指標
    """
    # 禁用深度監督
    self.set_deep_supervision_enabled(False)
    self.network.eval()
    
    # 創建預測器
    predictor = nnUNetPredictor(...)
    predictor.manual_initialization(...)
    
    # 對每個驗證案例進行預測
    for case_id in validation_cases:
        data, seg, properties = dataset_val.load_case(case_id)
        prediction = predictor.predict_sliding_window_return_logits(data)
        # 導出預測結果
        export_prediction_from_logits(...)
    
    # 計算指標
    metrics = compute_metrics_on_folder(...)
```

---

## 常見問題與解決方案

### 1. AMP 導致 NaN 問題

**問題**: 使用自動混合精度 (AMP) 時，Mamba 模塊可能產生 NaN 值。

**解決方案**: 禁用 AMP

```python
class nnUNetTrainerUMambaEncNoAMP(nnUNetTrainerUMambaEnc):
    def __init__(self, ...):
        super().__init__(...)
        self.grad_scaler = None  # 禁用 AMP
```

### 2. 內存不足

**問題**: 訓練時 GPU 內存不足。

**解決方案**:
- 減小批次大小（在 plans.json 中修改）
- 減小 patch size
- 使用梯度累積
- 使用更少的數據增強

### 3. 訓練速度慢

**問題**: 訓練速度較慢。

**解決方案**:
- 啟用 `torch.compile()`（設置環境變量 `nnUNet_compile=true`）
- 使用多進程數據加載（調整 `num_processes`）
- 使用 DDP 進行多 GPU 訓練
- 減少數據增強強度

### 4. 檢查點文件過大

**問題**: 檢查點文件佔用大量磁盤空間。

**解決方案**:
- 僅保存最佳檢查點（修改 `save_every`）
- 使用 `--disable_checkpointing` 僅在訓練結束時保存
- 定期清理舊的檢查點

### 5. 自定義網絡架構加載失敗

**問題**: 推理時無法加載自定義網絡架構。

**解決方案**:
- 確保 `build_network_architecture()` 是靜態方法
- 確保 Trainer 類別名稱與訓練時一致
- 檢查網絡架構的參數是否匹配

---

## 使用範例

### 訓練範例

#### U-Mamba 方法

訓練 2D U-Mamba Encoder 模型：
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaEnc
```

訓練 3D U-Mamba Bottleneck 模型：
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaBot
```

#### SAM-based 方法

訓練 SAM-based 模型：
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerNNSAM
```

訓練 SAM + Distance 模型：
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerNNSAMdis
```

訓練 SAM + Shape Prior 模型：
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerNNSAMshape
```

#### Distance-based 方法

訓練 Distance-based 模型：
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerNNdis
```

#### ViT-based 方法

訓練 ViT Concat 模型：
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerViTConcat
```

### 繼續訓練

```bash
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainerUMambaEnc --c
```

### 僅運行驗證

```bash
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainerUMambaEnc --val
```

### 使用最佳檢查點進行驗證

```bash
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainerUMambaEnc --val --val_best
```

---

## 4. 後處理 (Postprocess)

後處理階段用於清理預測結果，移除小的連通組件，只保留每個標籤的最大連通組件。

### 功能說明

`postprocess/postprocesscode.py` 執行以下操作：

1. **連通組件分析**：對每個標籤進行 3D 連通組件分析
2. **保留最大組件**：只保留每個標籤的最大連通組件
3. **文件重命名**：將處理後的文件重命名為標準格式

#### 使用方法

1. **修改路徑配置**：

```python
input_dir = r"C:\Users\user\Desktop\1130_3_result"                    # 輸入目錄（包含預測結果）
output_dir = os.path.join(input_dir, "post_processed")                # 後處理輸出目錄
dst_img = r"C:\Users\user\Desktop\1130_3_result\1130_3_upload"       # 最終輸出目錄
```

2. **執行腳本**：

```bash
python postprocess/postprocesscode.py
```

#### 處理流程

1. **讀取預測結果**：從 `input_dir` 讀取所有 `.nii.gz` 文件
2. **連通組件分析**：
   - 對每個標籤（1, 2, ...）分別進行處理
   - 使用 26 連通性進行 3D 連通組件分析
   - 計算每個連通組件的大小
   - 只保留最大的連通組件
3. **保存結果**：將清理後的結果保存到 `output_dir`
4. **文件重命名**：將文件重命名為 `patient{id:04d}.nii.gz` 格式並複製到 `dst_img`

#### 輸出格式

- 後處理結果：`{original_filename}.nii.gz`（保存在 `output_dir`）
- 最終輸出：`patient{id:04d}.nii.gz`（保存在 `dst_img`）

#### 參數說明

- **連通性**：使用 26 連通性（3D 中的完全連通）
- **起始編號**：文件重命名時從 51 開始（`i = i + 51`），可根據需要修改

#### 注意事項

- 需要安裝：`nibabel`、`numpy`、`cc3d`
- 輸入文件需為 `.nii.gz` 格式
- 確保輸出目錄有足夠的寫入權限
- 文件重命名邏輯可能需要根據實際需求調整

#### 安裝依賴

```bash
pip install nibabel numpy cc3d
```

---

## 參考資料

- [nnU-Net 官方文檔](https://github.com/MIC-DKFZ/nnUNet)
- [Mamba 官方倉庫](https://github.com/state-spaces/mamba)
- [MedSAM 官方倉庫](https://github.com/bowang-lab/MedSAM)
- [U-Mamba 論文](https://arxiv.org/abs/2401.04722)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [cc3d](https://github.com/seung-lab/connected-components-3d)

---


### 命名規範

建議使用以下命名規範：
- `nnUNetTrainer[方法名稱]`: 基本方法 Trainer
- `nnUNetTrainer[方法名稱][變體]`: 方法的變體（如 NoAMP、dis、shape 等）

例如：
- `nnUNetTrainerUMambaEnc`: U-Mamba Encoder
- `nnUNetTrainerUMambaEncNoAMP`: U-Mamba Encoder（無 AMP）
- `nnUNetTrainerNNSAMdis`: SAM + Distance
- `nnUNetTrainerNNSAMshape`: SAM + Shape Prior

---

**專案名稱**: AICUP2025FALL  


