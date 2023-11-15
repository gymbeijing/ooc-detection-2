import os
import os.path as osp
from re import template

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

torch.backends.cuda.matmul.allow_tf32 = True   # A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs.
torch.backends.cudnn.benchmark = True   # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
torch.backends.cudnn.deterministic = False   # if True, causes cuDNN to only use deterministic convolution algorithms
torch.backends.cudnn.allow_tf32 = True   # A bool that controls where TensorFloat-32 tensor cores may be used in cuDNN convolutions on Ampere or newer GPUs


class GraphAdapter(nn.Module):
    def __init__(self, cfg, classnames, clip_model, base_text_features):
        super().__init__()

    def forward(self):
        return None


class Net(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.graph_adapter = GraphAdapter(cfg, classnames)

