import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
from scipy.ndimage import distance_transform_edt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def compute_sdf_numpy(label, C):
    """
    Compute signed distance function for each class.
    Args:
        label: [D,H,W] numpy array with class labels
        C: number of classes
    Returns:
        sdf: [C,D,H,W] numpy array with signed distance maps
    """
    D, H, W = label.shape
    sdf = np.zeros((C, D, H, W), dtype=np.float32)

    for c in range(1, C):  # Skip background class
        mask = (label == c)

        if mask.sum() == 0:
            continue

        # Distance from outside the mask
        dist_out = distance_transform_edt(~mask)
        # Distance from inside the mask
        dist_in = distance_transform_edt(mask)
        # Signed distance: positive outside, negative inside
        sdf[c] = dist_out - dist_in

    return sdf


class nnUNetTrainerNNdis(nnUNetTrainer):
    """
    nnUNet Trainer with Shape-aware Distance Map Loss.
    This trainer uses pre-computed SDF maps to add shape constraints to the segmentation loss.
    """

    def __init__(self,
                 plans,
                 configuration,
                 fold,
                 dataset_json,
                 unpack_dataset=True,
                 device=torch.device("cuda")):

        # Directory containing pre-computed SDF maps
        self.sdf_dir = "/mnt/disk3/xinyou/aicup2025fall/sdfTr2"

        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # Training hyperparameters
        self.initial_lr = 1e-3
        self.weight_decay = 1e-4
        self.identifier = "nnUNetTrainer_SDF"

        # Load pre-computed SDF maps into memory
        self.sdf_cache = self._load_sdf_maps()

        # Store original loss function
        self.original_loss = self.loss

        # Replace with combined loss
        self.loss = self.combined_loss

        print(f"[nnUNetTrainerWithSDF] Initialized with SDF loss")
        print(f"[nnUNetTrainerWithSDF] SDF directory: {self.sdf_dir}")
        print(f"[nnUNetTrainerWithSDF] Number of cached SDFs: {len(self.sdf_cache)}")


    def _load_sdf_maps(self):
        """
        Load all pre-computed SDF maps from disk into memory cache.
        Returns:
            cache: Dictionary mapping case_id to SDF array
        """
        cache = {}
        if not os.path.exists(self.sdf_dir):
            raise FileNotFoundError(f"SDF directory not found: {self.sdf_dir}")
        
        for fname in os.listdir(self.sdf_dir):
            if fname.endswith("_sdf.npy"):
                case_id = fname.replace("_sdf.npy", "")
                sdf_path = os.path.join(self.sdf_dir, fname)
                cache[case_id] = np.load(sdf_path)
        
        print(f"[SDF] Loaded {len(cache)} SDF files from {self.sdf_dir}")
        return cache


    def sdf_shape_loss(self, pred_logits, target, case_ids):
        """
        Compute shape-aware loss using signed distance functions.
        Args:
            pred_logits: Model output logits [B, C, D, H, W]
            target: Ground truth labels [B, D, H, W]
            case_ids: List of case identifiers for the batch
        Returns:
            mean SDF loss across the batch
        """
        device = pred_logits.device
        B, C, D, H, W = pred_logits.shape
        
        # Get predicted labels
        prob = torch.softmax(pred_logits, dim=1)
        pred_label = prob.argmax(dim=1).detach().cpu().numpy()
        gt_label = target.detach().cpu().numpy()

        losses = []
        for i in range(B):
            cid = case_ids[i]
            
            # Load pre-computed ground truth SDF
            if cid not in self.sdf_cache:
                print(f"[WARNING] Case {cid} not found in SDF cache, skipping")
                continue
            
            sdf_gt = self.sdf_cache[cid]  # [C, D, H, W]
            sdf_gt = torch.tensor(sdf_gt, device=device, dtype=torch.float32)

            # Compute SDF from prediction
            sdf_pred = compute_sdf_numpy(pred_label[i], C)
            sdf_pred = torch.tensor(sdf_pred, device=device, dtype=torch.float32)

            # L1 loss between predicted and ground truth SDFs
            loss = torch.mean(torch.abs(sdf_pred - sdf_gt))
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return sum(losses) / len(losses)


    def combined_loss(self, output, target):
        """
        Combined segmentation and shape-aware loss.
        Args:
            output: Model output (can be list for deep supervision)
            target: Ground truth labels
        Returns:
            combined loss value
        """
        # Standard segmentation loss (Dice + CE)
        seg_loss = self.original_loss(output, target)

        # Extract logits
        if isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output

        # Get case IDs for current batch
        # Note: This assumes you have access to case identifiers
        # You may need to modify this based on your dataloader implementation
        try:
            case_ids = self.get_batch_case_ids()
        except:
            # Fallback: skip SDF loss if case IDs are not available
            print("[WARNING] Could not retrieve case IDs, using only segmentation loss")
            return seg_loss

        # Compute SDF loss
        sdf_loss = self.sdf_shape_loss(logits, target, case_ids)

        # Weight for SDF loss (tune this hyperparameter)
        sdf_weight = 0.005

        combined = seg_loss + sdf_weight * sdf_loss
        
        return combined


    def get_batch_case_ids(self):
        """
        Helper function to retrieve case IDs for current batch.
        This needs to be implemented based on your dataloader structure.
        """
        # TODO: Implement based on your actual dataloader
        # This is a placeholder - you'll need to modify based on how
        # nnUNet provides case identifiers
        raise NotImplementedError("get_batch_case_ids needs to be implemented")


    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json: dict,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build standard nnUNet architecture (no SAM/ViT modifications).
        """
        model = get_network_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision
        )

        print(f"[nnUNetTrainerWithSDF] Using standard nnUNet architecture")
        print(f"[nnUNetTrainerWithSDF] Patch size: {configuration_manager.patch_size}")

        return model


    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer = Adam(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-4
        )
        
        lr_scheduler = PolyLRScheduler(
            optimizer,
            self.initial_lr,
            self.num_epochs,
            exponent=0.9
        )
        
        return optimizer, lr_scheduler