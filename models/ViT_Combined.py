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

class ChannelGatedCombine(nn.Module):
    """
    Combines two model outputs [B, C, pred_len] using channel-wise weights
    derived from an input tensor [B, C, input_len].
    """
    def __init__(self, input_len, hidden_dim=64):
        super().__init__()
        
        # This is the gating network. It's a small MLP
        # that is applied to each channel's input features (input_len)
        # to produce a single logit for that channel.
        self.gating_network = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output one logit per channel
        )

    def forward(self, x_input, y_model_a, y_model_b):
        """
        x_input:   [batch, channel, input_len] (The input tensor)
        y_model_a: [batch, channel, pred_len]  (Output from model A)
        y_model_b: [batch, channel, pred_len]  (Output from model B)
        
        Note: input_len and pred_len can be different.
        Your prompt said the input was also [B, C, pred_len],
        so in your case, input_len == pred_len.
        """
        
        # 1. Pass the input tensor through the gating network
        # The Linear layers are applied to the last dimension (input_len)
        # Shape: [B, C, input_len] -> [B, C, 1]
        gate_logits = self.gating_network(x_input)
        
        # 2. Apply Sigmoid to get a weight 'w_a' between 0 and 1
        # This is the weight for Model A.
        # Shape: [B, C, 1]
        w_a = torch.sigmoid(gate_logits)
        
        # 3. The weight for Model B is (1 - w_a)
        # Shape: [B, C, 1]
        w_b = 1.0 - w_a
        
        # 4. Calculate the weighted sum
        # [B, C, pred_len] * [B, C, 1] (broadcasting)
        combined_output = (y_model_a * w_a) + (y_model_b * w_b)
        
        # 5. Return the combined output and the weights
        # final_output shape: [B, C, pred_len]
        return combined_output

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
        self.vit = timm.models.VisionTransformer(
            img_size=(patch_num, patch_len),
            patch_size=(stride, stride),
            in_chans=1,
            num_classes=0,  # No classification head
            embed_dim=configs.d_model,
            depth=3,
            num_heads=4,
            qkv_bias=True,
            drop_rate=configs.dropout,
            attn_drop_rate=configs.dropout
        )
        self.vit_forecast = nn.Linear(configs.d_model, configs.pred_len)

        self.combiner = ChannelGatedCombine(input_len=configs.seq_len)

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        num_out = self.head(enc_out)  # z: [bs x nvars x target_window]

        # Imaging
        B, C, L = x_enc.shape
        x = self.padding_patch_layer(x_enc)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = x.unsqueeze(1)
        x_img_enc = self.vit.forward_features(x)
        cls_token_embedding = x_img_enc[:, 0]

        y = self.vit_forecast(cls_token_embedding)
        image_out = torch.reshape(y, (B, C, self.pred_len))
        
        dec_out = self.combiner(x_enc, num_out, image_out)
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
