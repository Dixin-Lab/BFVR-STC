import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

from basicsr.archs.vqgan3D_arch import *
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(
        self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):

        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)

        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


@ARCH_REGISTRY.register()
class CodeFormer3D(VQAutoEncoder3D):
    def __init__(
        self,
        dim_embd=256,
        n_head=2,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=["32", "64", "128", "256"],
        fix_modules=["quantize", "generator"],
        vqgan_path=None,
    ):
        super(CodeFormer3D, self).__init__(
            512, 32, [1, 2, 4, 8], "nearest", 2, [32], codebook_size
        )

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location="cpu")["params_ema"]
            )

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.h = self.w = int(self.resolution / (2 ** (len(self.ch_mult) - 1)))
        self.latent_size = int(self.h * self.w)
        self.position_emb_s = nn.Parameter(torch.zeros(self.latent_size, self.dim_embd))
        self.position_emb_t = nn.Parameter(torch.zeros(self.latent_size, self.dim_embd))
        self.s_feat_emb = nn.Linear(self.embed_dim, self.dim_embd)
        self.t_feat_emb = nn.Linear(self.embed_dim, self.dim_embd)

        # transformer
        self.s_ft_layers = nn.Sequential(
            *[
                TransformerSALayer(
                    embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0
                )
                for _ in range(self.n_layers)
            ]
        )

        self.t_ft_layers = nn.Sequential(
            *[
                TransformerSALayer(
                    embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0
                )
                for _ in range(self.n_layers)
            ]
        )

        # logits_predict head
        self.s_idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd), nn.Linear(dim_embd, codebook_size, bias=False)
        )

        self.t_idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd), nn.Linear(dim_embd, codebook_size, bias=False)
        )

        self.channels = {
            "16": 512,
            "32": 256,
            "64": 256,
            "128": 128,
            "256": 128,
            "512": 64,
        }

        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        # ################### Encoder #####################
        batch, _, nframe = x.shape[:3]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)

        lq_feat = x
        b, c, t, hh, ww = x.size()
        # ################# spatial Transformer ###################
        x_s = x
        pos_emb_s = self.position_emb_s.unsqueeze(1).repeat(1, b * t, 1)
        # BCTHW -> (BT)C(HW) -> (HW)(BT)C
        x_s = rearrange(x_s, "B C T H W -> (H W) (B T) C")
        query_emb_s = x_s

        # Transformer encoder
        for layer in self.s_ft_layers:
            query_emb_s = layer(query_emb_s, query_pos=pos_emb_s)

        # output logits
        logits_s = self.s_idx_pred_layer(query_emb_s)  # (thw)(b)n
        logits_s = logits_s.permute(1, 0, 2)  # (thw)(b)n -> (b)(thw)n

        # ################# temporal Transformer ###################
        # temporal attention between frames
        # bcthw -> bthwc
        x_t1 = x.permute(0, 2, 3, 4, 1).contiguous()
        z_t_flattened = self.quantize.SA(x_t1)
        # z -> bthwc
        z_t_flattened = z_t_flattened.permute(0, 2, 3, 4, 1).contiguous()

        # adjacent motion residual
        # bthwc -> tbhwc
        x_t2 = x.permute(1, 0, 2, 3, 4).contiguous()
        z_t_flattened_ = x_t2.reshape(t, -1)
        t, D = z_t_flattened_.size()
        first_row = z_t_flattened_[0:1]
        z_t_flattened_ = torch.cat((first_row, z_t_flattened_), dim=0)
        z_t_flattened_ = z_t_flattened_[1:] - z_t_flattened_[:-1]
        z_t_flattened_ = z_t_flattened_.reshape(t, b, hh, ww, c)
        # tbhwc -> bthwc
        z_t_flattened_ = z_t_flattened_.permute(1, 0, 2, 3, 4).contiguous()

        x_t = z_t_flattened + z_t_flattened_

        x_t = rearrange(x_t, "B T H W C -> (H W) (B T) C", B=b, H=hh, W=ww, C=c)
        pos_emb_t = self.position_emb_t.unsqueeze(1).repeat(1, b * t, 1)
        query_emb_t = x_t

        # Transformer encoder
        for layer in self.t_ft_layers:
            query_emb_t = layer(query_emb_t, query_pos=pos_emb_t)

        # output logits
        logits_t = self.t_idx_pred_layer(query_emb_t)  # (thw)(b)n
        logits_t = logits_t.permute(1, 0, 2)  # (thw)(b)n -> (b)(thw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits_s, logits_t, x

        # ################# Quantization ###################
        soft_one_hot_s = F.softmax(logits_s, dim=2)
        _, top_idx_s = torch.topk(soft_one_hot_s, 1, dim=2)
        soft_one_hot_t = F.softmax(logits_t, dim=2)
        _, top_idx_t = torch.topk(soft_one_hot_t, 1, dim=2)
        shape = [batch, int(nframe / 2.0), self.h, self.w, self.embed_dim]
        quant_feat = self.quantize.get_codebook_feat(top_idx_s, top_idx_t, shape=shape)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## Generator ####################
        x = quant_feat
        for i, block in enumerate(self.generator.blocks):
            x = block(x)

        x = self.generator.act_func(x)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        return out, logits_s, logits_t, lq_feat
