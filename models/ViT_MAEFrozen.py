import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import timm

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.padding = stride
        self.patch_len = patch_len
        self.stride = stride
        self.pretrain_img_size = 224

        # patching and embedding
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, self.padding, configs.dropout)
        patch_num = int((configs.seq_len - patch_len) / stride + 2)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Imaging Components
        self.vit = timm.create_model('vit_base_patch16_224.mae', pretrained=True)

        # Remove the classification head
        self.vit.head = nn.Identity()

        # Freeze all parameters in the ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # Forecasting head, input dim must match the ViT's embedding dim
        # self.vit.embed_dim for vit_base_patch16_224 is 768
        self.vit_forecast = nn.Linear(self.vit.embed_dim, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)

        # Imaging
        B, C, L = x_enc.shape
        x = self.padding_patch_layer(x_enc)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[3], x.shape[2]))

        # Reshape to [B*C, 1, H, W] where H=patch_len, W=patch_num
        x = x.unsqueeze(1)
        BC_batch, _, H_small, W_small = x.shape

        # 1. Handle Channel Mismatch: Repeat 1 channel to 3 channels
        x = x.repeat(1, 3, 1, 1)  # Shape: [B*C, 3, H_small, W_small]

        # 2. Handle Size Mismatch: Pad to pre-trained model's input size (e.g., 224x224)
        # Create a zero tensor (canvas) of the target size
        padded_x = torch.zeros(
            BC_batch, 3, self.pretrain_img_size, self.pretrain_img_size,
            device=x.device, dtype=x.dtype
        )

        # Place the small image at the top-left corner
        padded_x[:, :, :H_small, :W_small] = x

        # Pass the padded 224x224 image to the ViT
        # With head=nn.Identity(), self.vit(padded_x) returns the CLS token embedding
        cls_token_embedding = self.vit(padded_x)  # Shape: [B*C, embed_dim (768)]

        y = self.vit_forecast(cls_token_embedding)
        y = torch.reshape(y, (B, C, self.pred_len))
        dec_out = y.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
