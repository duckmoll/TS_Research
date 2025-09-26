import sys
import torch
from torch import nn
from models.backbone import PatchTST_Backbone
from layers.Autoformer_EncDec import series_decomp_multi, series_decomp, series_decomp_multi_moe

torch.autograd.set_detect_anomaly(True)

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.pred_len = configs.pred_len
        # kernel_size = configs.moving_avg
        self.model = PatchTST_Backbone(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x = self.model(x_enc)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

