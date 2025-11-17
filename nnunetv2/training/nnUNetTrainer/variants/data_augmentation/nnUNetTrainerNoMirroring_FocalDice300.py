import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.focal import RobustFocalLoss


class nnUNetTrainerNoMirroring_FocalDice300(nnUNetTrainerNoMirroring):
    """
    Trainer variant using Focal + Dice loss (with alpha weighting)
    Epochs: 300
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300  # limit to 300 epochs

    def _build_loss(self):
        alpha_vec = [
            0.05, 1.221561, 0.282983, 1.598269, 0.409259,
            3.040343, 2.660109, 0.146232, 0.423893,
            0.631168, 0.361307, 0.605790, 0.619086
        ]

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {'batch_dice': self.configuration_manager.batch_dice,
                 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            loss = DC_and_CE_loss(
                # Dice kwargs
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'smooth': 1e-5,
                    'do_bg': False,
                    'ddp': self.is_ddp
                },
                # Focal kwargs
                {
                    'gamma': 2.0,
                    'alpha': alpha_vec
                },
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
                ce_class=RobustFocalLoss
            )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # Deep supervision weighting
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
