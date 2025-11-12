import sys
import torch
from torch import nn
from models.backbone import PatchTST_Backbone
from layers.Autoformer_EncDec import series_decomp_multi, series_decomp, series_decomp_multi_moe, series_decomp_multi_moe_mark, series_decomp_multi_moe_mark_topk

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
        self.moving_avg_type = configs.moving_avg_type
        # kernel_size = configs.moving_avg
        if configs.moving_avg_type == "single":
            self.decomp_module = series_decomp(24)
        elif configs.moving_avg_type == "multiple":
            self.decomp_module = series_decomp_multi([7, 12, 14, 24, 48])
        elif configs.moving_avg_type == "moe":
            self.decomp_module = series_decomp_multi_moe([7, 12, 14, 24, 48])
        elif configs.moving_avg_type == "multiple_all":
            self.decomp_module = series_decomp_multi(range(2, configs.seq_len // 2))
        elif configs.moving_avg_type == "multiple_all_moe":
            self.decomp_module = series_decomp_multi_moe(range(2, configs.seq_len // 2))
        elif configs.moving_avg_type == "multiple_all_moe_mark":
            self.decomp_module = series_decomp_multi_moe_mark(range(2, configs.seq_len // 2))
        elif configs.moving_avg_type == "multiple_all_moe_mark_topk":
            self.decomp_module = series_decomp_multi_moe_mark_topk(range(2, configs.seq_len // 2), configs.moe_topk)
        else:
            self.decomp_module = None
        # self.decomp_module = series_decomp_multi_moe([7, 12, 14, 24, 48])
        self.model_trend = PatchTST_Backbone(configs)
        self.model_res = PatchTST_Backbone(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.decomp_module is None:
            return self.model_trend(x_enc)
        elif self.moving_avg_type in ["multiple_all_moe_mark", "multiple_all_moe_mark_topk"]:
            res_init, trend_init = self.decomp_module(x_enc, x_mark_enc)
        else:
            res_init, trend_init = self.decomp_module(x_enc)
        res = self.model_res(res_init)
        trend = self.model_trend(trend_init)
        x = res + trend
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

