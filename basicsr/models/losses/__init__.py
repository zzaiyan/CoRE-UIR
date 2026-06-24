# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .loss_builder import build_loss_modules, compute_losses
from .losses import (L1Loss, MSELoss, PSNRLoss, LPIPSLoss, SSIMLoss,
                     MSSSIMLoss)

__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'LPIPSLoss', 'SSIMLoss', 'MSSSIMLoss',
    'build_loss_modules', 'compute_losses',
]
