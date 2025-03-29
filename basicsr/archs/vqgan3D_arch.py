import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY
import random
from torch.hub import load
import torchvision.models as models
from einops import rearrange


def normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


def normalize2(in_channels):
    return torch.nn.GroupNorm(
        num_groups=16, num_channels=in_channels, eps=1e-6, affine=True
    )


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


class SA2(nn.Module):
    """
    Temporal Attention between frames
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize2(in_channels)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # x: bthwc -> bcthw
        x = x.permute(0, 4, 1, 2, 3)
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        # bcthw -> bhwtc
        q = q.permute(0, 3, 4, 2, 1)
        q = q.reshape(b * h * w, t, c)
        # bcthw -> bhwct
        k = k.permute(0, 3, 4, 1, 2)
        k = k.reshape(b * h * w, c, t)
        # w_ : (b*t, h*w, h*w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        # bcthw -> bhwct
        v = v.permute(0, 3, 4, 1, 2)
        v = v.reshape(b * h * w, c, t)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, h, w, c, t)
        h_ = h_.permute(0, 3, 4, 1, 2)
        h_ = self.proj_out(h_)

        return h_


class VectorQuantizerST(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizerST, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding_s = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding_s.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )
        self.embedding_t = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding_t.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )
        self.SA = SA2(emb_dim)

    def forward(self, z, training=True):
        # input z: (b,c,t,h,w)
        b, c, t, h, w = z.size()
        # z -> (b,t,h,w,c)
        z = z.permute(0, 2, 3, 4, 1).contiguous()

        z_s_flattened = z.view(-1, self.emb_dim)
        d = (
            (z_s_flattened**2).sum(dim=1, keepdim=True)
            + (self.embedding_s.weight**2).sum(1)
            - 2 * torch.matmul(z_s_flattened, self.embedding_s.weight.t())
        )
        mean_distance = torch.mean(d)
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.codebook_size
        ).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # get quantized latent vectors
        z_s_q = torch.matmul(min_encodings, self.embedding_s.weight).view(z.shape)
        min_encoding_indices = min_encoding_indices.view((t, b, h, w))
        min_encoding_indices = rearrange(min_encoding_indices, "T B H W -> (B T) (H W)")

        # temporal attention between frames
        z_t_flattened = self.SA(z)
        # z -> (b,t,h,w,c)
        z_t_flattened = z_t_flattened.permute(0, 2, 3, 4, 1).contiguous()

        # adjacent motion residual
        # bthwc -> tbhwc
        z_ = z.permute(1, 0, 2, 3, 4).contiguous()
        z_t_flattened_ = z_.reshape(t, -1)
        t, D = z_t_flattened_.size()
        first_row = z_t_flattened_[0:1]
        z_t_flattened_ = torch.cat((first_row, z_t_flattened_), dim=0)
        z_t_flattened_ = z_t_flattened_[1:] - z_t_flattened_[:-1]
        z_t_flattened_ = z_t_flattened_.reshape(t, b, h, w, c)
        # tbhwc -> bthwc
        z_t_flattened_ = z_t_flattened_.permute(1, 0, 2, 3, 4).contiguous()

        z_t_flattened = z_t_flattened + z_t_flattened_

        z_t_flattened = z_t_flattened.view(-1, self.emb_dim)
        d2 = (
            (z_t_flattened**2).sum(dim=1, keepdim=True)
            + (self.embedding_t.weight**2).sum(1)
            - 2 * torch.matmul(z_t_flattened, self.embedding_t.weight.t())
        )
        mean_distance2 = torch.mean(d2)
        # find closest encodings
        min_encoding_indices2 = torch.argmin(d2, dim=1).unsqueeze(1)
        min_encodings2 = torch.zeros(
            min_encoding_indices2.shape[0], self.codebook_size
        ).to(z)
        min_encodings2.scatter_(1, min_encoding_indices2, 1)
        # get quantized latent vectors
        z_t_q = torch.matmul(min_encodings2, self.embedding_t.weight).view(z.shape)
        min_encoding_indices2 = min_encoding_indices2.view((t, b, h, w))
        min_encoding_indices2 = rearrange(
            min_encoding_indices2, "T B H W -> (B T) (H W)"
        )

        z_q = z_t_q + z_s_q

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity_s = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        e_mean = torch.mean(min_encodings2, dim=0)
        perplexity_t = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        # bthwc -> bcthw
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        # marginal regularization
        kl = nn.KLDivLoss()

        d_n = 1 / d
        d_n = d_n / d_n.sum(dim=1, keepdim=True)
        avg_probs_s = d_n.sum(dim=0)
        avg_probs_s /= b * t * h * w
        p_s = (
            torch.ones(self.codebook_size, 1).to(avg_probs_s.device)
            / self.codebook_size
        )
        d_kl_s = kl(
            F.log_softmax(torch.flatten(avg_probs_s), dim=0),
            F.softmax(torch.flatten(p_s), dim=0),
        )
        loss2 = d_kl_s

        d2_n = 1 / d2
        d2_n = d2_n / d2_n.sum(dim=1, keepdim=True)
        avg_probs_t = d2_n.sum(dim=0)
        avg_probs_t /= b * t * h * w
        p_t = (
            torch.ones(self.codebook_size, 1).to(avg_probs_t.device)
            / self.codebook_size
        )
        d_kl_t = kl(
            F.log_softmax(torch.flatten(avg_probs_t), dim=0),
            F.softmax(torch.flatten(p_t), dim=0),
        )
        loss2 += d_kl_t

        return (
            z_q,
            loss,
            0.1 * loss2,
            {
                "min_encoding_indices_s": min_encoding_indices,  # (bt)(hw)
                "min_encoding_indices_t": min_encoding_indices2,  # (bt)(hw)
                "perplexity_s": perplexity_s,
                "perplexity_t": perplexity_t,
            },
        )

    def get_codebook_feat(self, indices, indices2, shape):
        # input indices: batch*token_num -> (batch*token_num)*1
        # shape: batch, height, width, channel
        indices = indices.view(-1, 1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_s_q = torch.matmul(min_encodings.float(), self.embedding_s.weight)

        indices2 = indices2.view(-1, 1)
        min_encodings2 = torch.zeros(indices2.shape[0], self.codebook_size).to(indices2)
        min_encodings2.scatter_(1, indices2, 1)
        # get quantized latent vectors
        z_t_q = torch.matmul(min_encodings2.float(), self.embedding_t.weight)

        z_q = z_t_q + z_s_q
        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 4, 1, 2, 3).contiguous()

        return z_q


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Downsample2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Upsample2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = q.reshape(b, c, t * h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, t * h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, t * h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, t, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        nf,
        emb_dim,
        ch_mult,
        num_res_blocks,
        resolution,
        attn_resolutions,
    ):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # initial convultion
        blocks.append(nn.Conv3d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                if i == self.num_resolutions - 2 or i == self.num_resolutions - 3:
                    blocks.append(Downsample2D(block_in_ch))
                else:
                    blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(
            nn.Conv3d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1)
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)

        blocks = []
        # initial conv
        blocks.append(
            nn.Conv3d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1)
        )

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                if i == self.num_resolutions - 2 or i == self.num_resolutions - 3:
                    blocks.append(Upsample2D(block_in_ch))
                else:
                    blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(
            nn.Conv3d(
                block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1
            )
        )

        self.blocks = nn.ModuleList(blocks)
        self.act_func = nn.Sigmoid()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.act_func(x)
        return x


@ARCH_REGISTRY.register()
class VQAutoEncoder3D(nn.Module):
    def __init__(
        self,
        img_size,
        nf,
        ch_mult,
        quantizer="nearest",
        res_blocks=2,
        attn_resolutions=[32],
        codebook_size=1024,
        emb_dim=256,
        beta=0.25,
        gumbel_straight_through=False,
        gumbel_kl_weight=1e-8,
        model_path=None,
    ):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3
        self.nf = nf
        self.n_blocks = res_blocks
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            256,
            self.attn_resolutions,
        )

        self.beta = beta  # 0.25
        self.quantize = VectorQuantizerST(self.codebook_size, self.embed_dim, self.beta)

        self.generator = Generator(
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            256,
            self.attn_resolutions,
        )

        if model_path is not None:
            chkpt = torch.load(model_path, map_location="cpu")
            if "params_ema" in chkpt:
                self.load_state_dict(
                    torch.load(model_path, map_location="cpu")["params_ema"]
                )
                logger.info(f"vqgan is loaded from: {model_path} [params_ema]")
            elif "params" in chkpt:
                self.load_state_dict(
                    torch.load(model_path, map_location="cpu")["params"]
                )
                logger.info(f"vqgan is loaded from: {model_path} [params]")
            else:
                raise ValueError(f"Wrong params!")

    def forward(self, x):
        x = self.encoder(x)
        quant, codebook_loss, loss2, quant_stats = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss, loss2, quant_stats


@ARCH_REGISTRY.register()
class VQGANDiscriminator3D(nn.Module):
    def __init__(self, backbone="dinov2_s", nc=3, ndf=32, model_path=None):
        super().__init__()
        self.backbones = {
            "dinov2_s": {
                "name": "dinov2_vits14",
                "embedding_size": 384,
                "patch_size": 14,
            },
            "dinov2_b": {
                "name": "dinov2_vitb14",
                "embedding_size": 768,
                "patch_size": 14,
            },
            "dinov2_l": {
                "name": "dinov2_vitl14",
                "embedding_size": 1024,
                "patch_size": 14,
            },
            "dinov2_g": {
                "name": "dinov2_vitg14",
                "embedding_size": 1536,
                "patch_size": 14,
            },
        }
        self.backbone = load(
            "facebookresearch/dinov2",
            self.backbones[backbone]["name"]
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.disc1 = nn.Linear(self.backbones[backbone]["embedding_size"], 1)
        self.disc2 = nn.Linear(self.backbones[backbone]["embedding_size"], 2)
        self.disc4 = nn.Linear(self.backbones[backbone]["embedding_size"], 4)

        if model_path is not None:
            chkpt = torch.load(model_path, map_location="cpu")
            if "params_d" in chkpt:
                self.load_state_dict(
                    torch.load(model_path, map_location="cpu")["params_d"]
                )
            elif "params" in chkpt:
                self.load_state_dict(
                    torch.load(model_path, map_location="cpu")["params"]
                )
            else:
                raise ValueError(f"Wrong params!")

    def forward(self, x):
        # input x shape: b c t h w
        x = x.permute(0, 2, 1, 3, 4)  # x: b t c h w
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = F.interpolate(x, size=(224, 224))
        with torch.no_grad():
            x = self.backbone(x)
        x = x.to(torch.float32)
        x1 = torch.mean(self.disc1(x), dim=1)
        x2 = torch.mean(self.disc2(x), dim=1)
        x4 = torch.mean(self.disc4(x), dim=1)

        return torch.cat((x1, x2, x4), dim=0)
