"""
Use CLIP as the base
"""

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

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "Twitter-COMMs": "a piece of news in {}."   # {domain}
}


class TextEncoder(nn.Module):
    """
    Encode the prompts into embeddings(?)
    """
    def __init__(self, clip_model):
        super().__init__()
        # Take out clip's components
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # Prompt embedding + clip's positional embedding
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)   # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)   # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]   n_ctx is the max. number of tokens in an input sequence
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection   # for each row, what does argmax refer to?

        return x