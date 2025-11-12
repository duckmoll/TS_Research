import sys
import torch
from torch import nn
from models.PatchTST_Decomp import Model as PatchTST_Decomp


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
        self.topk = configs.moe_topk
        self.experts = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)
        batch_len_ls = range(8, configs.seq_len // 8, 4)
        for num in range(len(batch_len_ls)):
            configs.patch_len = batch_len_ls[num]
            configs.patch_stride = int(batch_len_ls[num] // 2)
            print(configs.patch_len, configs.patch_stride)
            self.experts.append(PatchTST_Decomp(configs))

        self.gating_layer = torch.nn.Linear(1, len(batch_len_ls))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x: [Batch, Input Length, Channel]
        # Input
        series_summary = torch.mean(torch.cat((x_enc, x_mark_enc), dim=-1), dim=1) # [Batch, Channel]
        gating_input = series_summary.unsqueeze(-1) # [Batch, Channel, 1]
        logits = self.gating_layer(gating_input) # [Batch, Channel, num_experts]
        top_k_values, top_k_indices = logits.topk(self.topk, dim=-1)
        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits = masked_logits.scatter(dim=-1, index=top_k_indices, src=top_k_values)
        weights = self.softmax(masked_logits)   # [Batch, Channel, num_experts]
        weights_broadcast = weights.unsqueeze(1)  # [Batch, 1, Channel, num_experts]

        # Output
        pred_ls = []
        for i in range(len(self.experts)):
            expert = self.experts[i]
            pred_i = expert(x_enc, x_mark_enc, x_dec, x_mark_dec)
            pred_ls.append(pred_i.unsqueeze(-1))
        preds_all = torch.cat(pred_ls, dim=-1)

        # Combine
        num_data_channels = x_enc.shape[-1]  # Gets the number of channels (7) from x_enc
        relevant_weights = weights_broadcast[:, :, :num_data_channels, :]  # Selects the first 7 channels of weights
        combined_pred = torch.sum(preds_all * relevant_weights, dim=-1)
        return combined_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


