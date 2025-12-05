import torch
import os
import torch.nn as nn
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt as MedNeXt_Orig
from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper
import torch.nn.functional as F
from nnunet_mednext.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet_mednext.training.loss_functions.dice_loss import SoftDiceLoss


class MedNeXt(MedNeXt_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.num_classes = kwargs['n_classes']
        # self.do_ds = False   # only needed for mirrored TTA

    @staticmethod
    def reshape_outputs(net, desired_output_shape):
        if isinstance(net, tuple):
            assert len(net) == len(desired_output_shape)
            reshaped = [i.view((i.shape[0], i.shape[1], -1)) for i in net]
            for s, d in zip(reshaped, desired_output_shape):
                assert s.shape[2] == d, ('Desired output shape does not match network output shape. '
                                         'Make sure the network architecture is correct and that the '
                                         'desired output shape at stage i corresponds to the output of '
                                         'the network at stage i.')
        else:
            reshaped = net.view((net.shape[0], net.shape[1], -1))
            assert reshaped.shape[2] == desired_output_shape[0], \
                ('Desired output shape does not match network output shape. '
                 'Make sure the network architecture is correct and that the '
                 'desired output shape at stage i corresponds to the output of '
                 'the network at stage i.')
        return reshaped

    @staticmethod
    def reshape_targets(target, desired_out_shape):
        if isinstance(desired_out_shape, list):
            reshaped = []
            for d in desired_out_shape:
                reshaped.append(target.view((target.shape[0], 1, -1)))
                assert reshaped[-1].shape[2] == d, \
                    ('Desired output shape does not match target shape. '
                     'Make sure the desired channels are correct and that '
                     'the desired output shape at stage i corresponds to the '
                     'output of the network at stage i.')
        else:
            reshaped = target.view((target.shape[0], 1, -1))
            assert reshaped.shape[2] == desired_out_shape[0], \
                ('Desired output shape does not match target shape. '
                 'Make sure the desired channels are correct and that '
                 'the desired output shape at stage i corresponds to the '
                 'output of the network at stage i.')
        return reshaped


# -----------------------------
#   Dice + IoU 損失函數
# -----------------------------
class SoftIoULoss(nn.Module):
    """
    Soft IoU (Jaccard) loss that works with network probability outputs.
    """
    def __init__(self, apply_nonlin=None, smooth=1e-5, do_bg=False):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, net_output, target):
        """
        net_output: (B, C, ...)
        target:     (B, ...) int labels or (B, 1, ...) or (B, C, ...) one-hot
        """
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # ensure target has shape (B, C, ...)
        if target.ndim == net_output.ndim - 1:
            # (B, D, H, W) -> (B, 1, D, H, W)
            target = target.unsqueeze(1)

        if target.shape[1] == 1 and net_output.shape[1] > 1:
            # (B,1,D,H,W) -> one-hot (B,C,D,H,W)
            target = F.one_hot(
                target.long().squeeze(1),
                num_classes=net_output.shape[1]
            )
            target = target.permute(0, 4, 1, 2, 3).float()
        elif target.shape[1] == net_output.shape[1]:
            target = target.float()
        else:
            # fallback
            target = F.one_hot(
                target.long(),
                num_classes=net_output.shape[1]
            )
            target = target.permute(0, 4, 1, 2, 3).float()

        if (not self.do_bg) and net_output.shape[1] > 1:
            net_output = net_output[:, 1:]
            target = target[:, 1:]

        axes = tuple(range(2, net_output.ndim))
        intersection = (net_output * target).sum(dim=axes)
        union = net_output.sum(dim=axes) + target.sum(dim=axes) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


class DicePlusIoULoss(nn.Module):
    """
    Combined Soft Dice + Soft IoU loss.
    """
    def __init__(self, apply_nonlin=None,
                 dice_weight=0.5, iou_weight=0.5,
                 smooth=1e-5, do_bg=False):
        super().__init__()
        self.dice = SoftDiceLoss(
            apply_nonlin=apply_nonlin,
            smooth=smooth,
            do_bg=do_bg
        )
        self.iou = SoftIoULoss(
            apply_nonlin=apply_nonlin,
            smooth=smooth,
            do_bg=do_bg
        )
        self.dw = dice_weight
        self.iw = iou_weight

    def forward(self, net_output, target):
        dice_loss = self.dice(net_output, target)
        iou_loss = self.iou(net_output, target)
        return self.dw * dice_loss + self.iw * iou_loss


# --------------------------------------------------
#   原本 MedNeXt 基底 Trainer（保持不動）
# --------------------------------------------------
class nnUNetTrainerV2_Optim_and_LR(nnUNetTrainerV2):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-3

    def process_plans(self, plans):
        super().process_plans(plans)
        # Please don't do this for nnunet. This is only for MedNeXt for all the DS to be used
        num_of_outputs_in_mednext = 5
        self.net_num_pool_op_kernel_sizes = [[2,2,2] for i in range(num_of_outputs_in_mednext+1)]    
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None



class nnUNetTrainerV2_MedNeXt_S_kernel3(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2                 ,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually disable residual connections
            do_res_up_down=True             , # Can be used to disable residual connections in up/downsampling blocks
            block_counts=[2,2,2,2,2,2,2,2,2],  # Can be used to test shallow/deep MedNeXt variants
        )
        
        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_B_kernel3(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels,  
            n_channels = 32, 
            n_classes = self.num_classes, 
            exp_r=2                 , 
            kernel_size=3,                 
            deep_supervision=True,            
            do_res=True,                      
            do_res_up_down=True             , 
            block_counts=[3,3,3,3,3,3,3,3,3],
        )
        
        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_M_kernel3(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually disable residual connections
            do_res_up_down=True             , # Can be used to disable residual connections in up/downsampling blocks
            block_counts=[3,3,3,3,3,3,3,3,3],
        )
        
        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_L_kernel3(nnUNetTrainerV2_Optim_and_LR):  
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2], 
            kernel_size=3,                 
            deep_supervision=True,            
            do_res=True,                      
            do_res_up_down=True             , 
            block_counts=[3,3,9,3,3,9,3,3,3],
        )
        
        if torch.cuda.is_available():
            self.network.cuda()


# 5x5 variant

class nnUNetTrainerV2_MedNeXt_S_kernel5(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2                 ,         # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually disable residual connections
            do_res_up_down=True             , # Can be used to disable residual connections in up/downsampling blocks
            block_counts=[2,2,2,2,2,2,2,2,2],  # Can be used to test shallow/deep MedNeXt variants
        )
        
        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_B_kernel5(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels,  
            n_channels = 32, 
            n_classes = self.num_classes, 
            exp_r=2                 , 
            kernel_size=5,                 
            deep_supervision=True,            
            do_res=True,                      
            do_res_up_down=True             , 
            block_counts=[3,3,3,3,3,3,3,3,3],
        )
        
        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_M_kernel5(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually disable residual connections
            do_res_up_down=True             , # Can be used to disable residual connections in up/downsampling blocks
            block_counts=[3,3,3,3,3,3,3,3,3],
        )
        
        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_L_kernel5(nnUNetTrainerV2_Optim_and_LR):  
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2], 
            kernel_size=5,                 
            deep_supervision=True,            
            do_res=True,                      
            do_res_up_down=True             , 
            block_counts=[3,3,9,3,3,9,3,3,3],
        )
        
        if torch.cuda.is_available():
            self.network.cuda()


# lr sweep configs

class nnUNetTrainerV2_MedNeXt_S_kernel3_lr_5e_4(nnUNetTrainerV2_MedNeXt_S_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_S_kernel3_lr_25e_5(nnUNetTrainerV2_MedNeXt_S_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_S_kernel3_lr_1e_4(nnUNetTrainerV2_MedNeXt_S_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_S_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_S_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_S_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_S_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_S_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_S_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_B_kernel3_lr_5e_4(nnUNetTrainerV2_MedNeXt_B_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_B_kernel3_lr_25e_5(nnUNetTrainerV2_MedNeXt_B_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_B_kernel3_lr_1e_4(nnUNetTrainerV2_MedNeXt_B_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_B_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_B_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_B_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_M_kernel3_lr_5e_4(nnUNetTrainerV2_MedNeXt_M_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_M_kernel3_lr_25e_5(nnUNetTrainerV2_MedNeXt_M_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_M_kernel3_lr_1e_4(nnUNetTrainerV2_MedNeXt_M_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_M_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_M_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_M_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_L_kernel3_lr_5e_4(nnUNetTrainerV2_MedNeXt_L_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_L_kernel3_lr_25e_5(nnUNetTrainerV2_MedNeXt_L_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_L_kernel3_lr_1e_4(nnUNetTrainerV2_MedNeXt_L_kernel3):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_L_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_L_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_L_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


# --------------------------------------------------
#   新增的 Dice + IoU 版本 S_kernel3 trainer
# --------------------------------------------------
class nnUNetTrainerV2_MedNeXt_S_kernel3_dice_iou(nnUNetTrainerV2_MedNeXt_S_kernel3):
    """
    Same architecture and MedNeXt-specific settings as nnUNetTrainerV2_MedNeXt_S_kernel3,
    but with Dice + IoU loss and a smaller learning rate for stability.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # use the lr you tuned for Dice+IoU (1e-4)
        self.initial_lr = 1e-4

    def initialize(self, training=True):
        """
        Call the parent initialize (which builds network, sets ds_loss_weights, etc.),
        then replace the loss with Dice+IoU wrapped in MultipleOutputLoss2 for deep supervision.
        """
        super().initialize(training)

        base_loss = DicePlusIoULoss(
            apply_nonlin=softmax_helper,
            dice_weight=0.5,
            iou_weight=0.5,
            smooth=1e-5,
            do_bg=False
        )
        self.loss = MultipleOutputLoss2(base_loss, self.ds_loss_weights)
