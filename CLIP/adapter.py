import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from parse_options import parse_option

args = parse_option()
use_cuda = torch.cuda.is_available()
device = torch.device(args.gpu if use_cuda else "cpu")


# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y


class ChannelAttentionMLP(nn.Module):
    def __init__(self, c_in, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c_in),
            nn.Softplus()  # Ensures output >= 0
        )
    def forward(self, attn_input):
        # attn_input: [c_in]
        return self.mlp(attn_input)

        
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features, variance_amplify=1.0):
        super().__init__()
        self.variance_amplify = variance_amplify
        self.text_features = None
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for _ in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for _ in range(len(features))])
        
        # Separate MLPs for segmentation and detection
        self.seg_channel_attention_mlp = ChannelAttentionMLP(c_in=768)
        self.det_channel_attention_mlp = ChannelAttentionMLP(c_in=768)


    def set_text_features(self, text_features):
        self.text_features = text_features


    def forward(self, x, text_features=None):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []

        if text_features is None:
            text_features = self.text_features
        if text_features is None:
            raise ValueError("text_features must be provided or set before forward call")

        for i in range(24):
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                attn_out.append(attn)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        B, C, L = attn_out[0].shape
        H = int(math.sqrt(L - 1))
        out_attn = torch.zeros([H, H]).to(x.device)

        for i in range(len(attn_out)):
            out_attn = out_attn + attn_out[i][0, 0, 1:].view(H, H).to(x.device)
        x = x.permute(1, 0, 2)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        seg_patch_tokens = [p / (p.norm(dim=-1, keepdim=True) + 1e-7) for p in seg_patch_tokens]
        det_patch_tokens = [p / (p.norm(dim=-1, keepdim=True) + 1e-7) for p in det_patch_tokens]

        text_norm = text_features / (text_features.norm(dim=0, keepdim=True) + 1e-7)  # [c_in, 2]

        def apply_mlp_attention(patch_tokens, text_norm, attn_mlp_module, variance_amplify):
            # patch_tokens: [B, num_patches, c_in], already normalized
            B, N, C = patch_tokens.shape
            patch_exp = patch_tokens.unsqueeze(2)  # [B, N, c_in, 1]
            patch_exp = patch_exp.permute(0, 1, 3, 2)  # [B, N, 1, c_in]
            text_exp = text_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, c_in, 2]
            contrib = patch_exp * text_exp  # [B, N, c_in, 2]
            mean_contrib = contrib.mean(dim=(0, 1))  # [c_in, 2]
            diff = (mean_contrib[:, 0] - mean_contrib[:, 1]).abs()  # [c_in]

            attn_base = attn_mlp_module(diff)  # Output: [c_in], ≥ 0
            attention_weights = 1.0 + variance_amplify * attn_base  # [c_in], ≥ 1

            # Compute variance over channel dimension before applying weights for returning variance (for loss)
            # Variance computed across batch and patches axes (0,1)
            weighted_diff = diff * attention_weights
            variance_val = weighted_diff.var()

            return patch_tokens * attention_weights.view(1, 1, -1), variance_val, diff

        # Apply mask and also collect variances to return
        seg_patch_tokens_out = []
        det_patch_tokens_out = []
        seg_variances = []
        det_variances = []
        seg_diffs = []
        det_diffs = []

        for p in seg_patch_tokens:
            p_masked, var_val, diff = apply_mlp_attention(p, text_norm, self.seg_channel_attention_mlp, self.variance_amplify)
            seg_patch_tokens_out.append(p_masked)
            seg_variances.append(var_val)
            seg_diffs.append(diff)

        for p in det_patch_tokens:
            p_masked, var_val, diff = apply_mlp_attention(p, text_norm, self.det_channel_attention_mlp, self.variance_amplify)
            det_patch_tokens_out.append(p_masked)
            det_variances.append(var_val)
            det_diffs.append(diff)

        # Average variance over layers
        mean_seg_variance = torch.stack(seg_variances).mean() if seg_variances else torch.tensor(0.0, device=x.device)
        mean_det_variance = torch.stack(det_variances).mean() if det_variances else torch.tensor(0.0, device=x.device)

        # Aggregate discriminative power vectors per modality
        avg_seg_diff = torch.stack(seg_diffs).mean(dim=0) if seg_diffs else torch.zeros_like(diff)
        avg_det_diff = torch.stack(det_diffs).mean(dim=0) if det_diffs else torch.zeros_like(diff)

        return pooled, seg_patch_tokens_out, det_patch_tokens_out, mean_seg_variance, mean_det_variance, avg_seg_diff, avg_det_diff
