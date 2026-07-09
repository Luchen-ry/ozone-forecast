#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clean-room PE-conditioned diffusion WaveNet-style model.

The implementation intentionally does not import the vendored Graph WaveNet or
MTGNN code. It uses common published ideas: gated dilated temporal convolution,
graph propagation, residual/skip connections, PE conditioning, and diffusion
refinement.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(self, num_steps: int = 50):
        super().__init__()
        self.num_steps = int(num_steps)
        s = 0.008
        steps = torch.arange(self.num_steps + 1, dtype=torch.float64)
        f = torch.cos((steps / self.num_steps + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]
        betas = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], min=1e-5, max=0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())
        self.register_buffer("sqrt_ab", torch.sqrt(alpha_bars).float())
        self.register_buffer("sqrt_1m_ab", torch.sqrt(1.0 - alpha_bars).float())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_ab[t]
        sqrt_1m = self.sqrt_1m_ab[t]
        while sqrt_ab.dim() < x0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1m = sqrt_1m.unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_1m * noise, noise


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = max(self.dim // 2, 1)
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


def row_normalize_supports(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.float()
    eye = torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype)
    if adj.dim() == 2:
        if torch.count_nonzero(adj).item() == 0:
            return adj
        adj = adj + eye
        denom = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj / denom
    zero_mask = adj.abs().sum(dim=(-1, -2)) == 0
    adj = adj + eye.unsqueeze(0)
    denom = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    out = adj / denom
    if torch.any(zero_mask):
        out = out.clone()
        out[zero_mask] = 0.0
    return out


class GraphMixProp(nn.Module):
    def __init__(self, channels: int, num_supports: int, order: int = 2, dropout: float = 0.1):
        super().__init__()
        self.order = int(order)
        self.num_supports = int(num_supports)
        in_channels = channels * (1 + self.num_supports * self.order)
        self.proj = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def propagate(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, support)

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        if len(supports) != self.num_supports:
            raise ValueError(f"expected {self.num_supports} supports, got {len(supports)}")
        outs = [x]
        for support in supports:
            h = x
            for _ in range(self.order):
                h = self.propagate(h, support)
                outs.append(h)
        return self.dropout(self.proj(torch.cat(outs, dim=1)))


class PEDilatedGraphBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_supports: int,
        dilation: int,
        kernel_size: int = 2,
        dropout: float = 0.1,
        graph_order: int = 2,
        use_pe_film: bool = True,
        pe_dim: int = 6,
        pe_film_scale: float = 1.0,
        pe_film_zero_init: bool = False,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.use_pe_film = bool(use_pe_film)
        self.pe_film_scale = float(pe_film_scale)
        padding = (self.kernel_size - 1) * self.dilation
        self.filter_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, self.kernel_size),
            dilation=(1, self.dilation),
            padding=(0, padding),
        )
        self.gate_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, self.kernel_size),
            dilation=(1, self.dilation),
            padding=(0, padding),
        )
        self.graph = GraphMixProp(channels, num_supports=num_supports, order=graph_order, dropout=dropout)
        self.residual_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.skip_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        if self.use_pe_film:
            self.pe_film = nn.Linear(pe_dim, channels * 2)
            if pe_film_zero_init:
                nn.init.zeros_(self.pe_film.weight)
                nn.init.zeros_(self.pe_film.bias)
        else:
            self.pe_film = None

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        trim = (self.kernel_size - 1) * self.dilation
        return x[:, :, :, :-trim] if trim > 0 else x

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor], pe_features: torch.Tensor):
        z = torch.tanh(self._trim(self.filter_conv(x))) * torch.sigmoid(self._trim(self.gate_conv(x)))
        z = self.graph(z, supports)
        if self.pe_film is not None:
            film = self.pe_film(pe_features.float()) * self.pe_film_scale
            gamma, beta = film.chunk(2, dim=-1)
            gamma = torch.tanh(gamma).transpose(0, 1)[None, :, :, None]
            beta = beta.transpose(0, 1)[None, :, :, None]
            z = z * (1.0 + gamma) + beta
        skip = self.skip_proj(z)
        out = self.norm(self.residual_proj(z) + x)
        return F.relu(out), skip


class HorizonDenoiser(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        pre_len: int,
        num_nodes: int,
        num_supports: int,
        pe_dim: int,
        horizon_dim: int = 16,
        time_dim: int = 32,
        dropout: float = 0.1,
        graph_order: int = 1,
        pe_film_scale: float = 1.0,
        pe_film_zero_init: bool = False,
    ):
        super().__init__()
        self.pre_len = int(pre_len)
        self.num_nodes = int(num_nodes)
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.time_proj = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.horizon_emb = nn.Embedding(self.pre_len, horizon_dim)
        self.pe_proj = nn.Linear(pe_dim, horizon_dim)
        in_channels = 2 + hidden_size + time_dim + horizon_dim + horizon_dim
        self.in_proj = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.block1 = PEDilatedGraphBlock(
            hidden_size,
            num_supports=num_supports,
            dilation=1,
            kernel_size=2,
            dropout=dropout,
            graph_order=graph_order,
            use_pe_film=True,
            pe_dim=pe_dim,
            pe_film_scale=pe_film_scale,
            pe_film_zero_init=pe_film_zero_init,
        )
        self.block2 = PEDilatedGraphBlock(
            hidden_size,
            num_supports=num_supports,
            dilation=2,
            kernel_size=2,
            dropout=dropout,
            graph_order=graph_order,
            use_pe_film=True,
            pe_dim=pe_dim,
            pe_film_scale=pe_film_scale,
            pe_film_zero_init=pe_film_zero_init,
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, 1, kernel_size=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        y_coarse: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        supports: List[torch.Tensor],
        pe_features: torch.Tensor,
    ) -> torch.Tensor:
        # x_t, y_coarse: (B, L, N); context: (B, H, N)
        bsz, horizon, nodes = x_t.shape
        base = torch.stack([x_t, y_coarse], dim=1).permute(0, 1, 3, 2)
        ctx = context.unsqueeze(-1).expand(-1, -1, -1, horizon)
        time = self.time_proj(self.time_emb(t)).view(bsz, -1, 1, 1).expand(-1, -1, nodes, horizon)
        h_idx = torch.arange(horizon, device=x_t.device)
        h_emb = self.horizon_emb(h_idx).transpose(0, 1)[None, :, None, :].expand(bsz, -1, nodes, -1)
        pe = self.pe_proj(pe_features.float()).transpose(0, 1)[None, :, :, None].expand(bsz, -1, nodes, horizon)
        z = self.in_proj(torch.cat([base, ctx, time, h_emb, pe], dim=1))
        z, _ = self.block1(z, supports, pe_features)
        z, _ = self.block2(z, supports, pe_features)
        out = self.out_proj(z).squeeze(1).permute(0, 2, 1)
        return torch.sigmoid(out)


class PEDiffWaveNet(nn.Module):
    def __init__(
        self,
        num_nodes: int = 95,
        input_dim: int = 15,
        hidden_size: int = 64,
        pre_len: int = 6,
        diff_steps: int = 50,
        pe_features: Optional[np.ndarray] = None,
        pe_dim: int = 6,
        dilations: Iterable[int] = (1, 2, 4, 8, 1, 2, 4, 8),
        kernel_size: int = 2,
        dropout: float = 0.1,
        graph_order: int = 2,
        use_pe_film: bool = True,
        use_diffusion: bool = True,
        use_adaptive_adj: bool = True,
        adaptive_rank: int = 10,
        pe_graph_alpha: float = 1.0,
        pe_film_scale: float = 1.0,
        pe_film_zero_init: bool = False,
        normalize_pe_features: bool = True,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.pre_len = int(pre_len)
        self.use_diffusion = bool(use_diffusion)
        self.use_adaptive_adj = bool(use_adaptive_adj)
        self.pe_graph_alpha = float(pe_graph_alpha)
        self.diffusion = GaussianDiffusion(diff_steps)

        if pe_features is None:
            pe_arr = np.zeros((self.num_nodes, pe_dim), dtype=np.float32)
        else:
            pe_arr = np.asarray(pe_features, dtype=np.float32)
            if pe_arr.shape[0] != self.num_nodes:
                raise ValueError(f"pe_features nodes mismatch: {pe_arr.shape[0]} vs {self.num_nodes}")
            pe_dim = int(pe_arr.shape[1])
        if normalize_pe_features and pe_arr.size > 0:
            mean = pe_arr.mean(axis=0, keepdims=True)
            std = pe_arr.std(axis=0, keepdims=True)
            pe_arr = (pe_arr - mean) / np.maximum(std, 1e-6)
        self.pe_dim = int(pe_dim)
        self.register_buffer("pe_features", torch.tensor(pe_arr, dtype=torch.float32))

        base_supports = 3
        self.num_supports = base_supports + (1 if self.use_adaptive_adj else 0)
        self.input_proj = nn.Conv2d(self.input_dim, self.hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                PEDilatedGraphBlock(
                    self.hidden_size,
                    num_supports=self.num_supports,
                    dilation=d,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    graph_order=graph_order,
                    use_pe_film=use_pe_film,
                    pe_dim=self.pe_dim,
                    pe_film_scale=pe_film_scale,
                    pe_film_zero_init=pe_film_zero_init,
                )
                for d in dilations
            ]
        )
        self.end_proj = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_size, self.pre_len, kernel_size=1),
        )
        self.context_proj = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1)
        self.denoiser = HorizonDenoiser(
            hidden_size=self.hidden_size,
            pre_len=self.pre_len,
            num_nodes=self.num_nodes,
            num_supports=self.num_supports,
            pe_dim=self.pe_dim,
            dropout=dropout,
            graph_order=1,
            pe_film_scale=pe_film_scale,
            pe_film_zero_init=pe_film_zero_init,
        )
        if self.use_adaptive_adj:
            rank = int(max(1, adaptive_rank))
            self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, rank) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(rank, self.num_nodes) * 0.1)
        else:
            self.nodevec1 = None
            self.nodevec2 = None

    def _supports(self, adj: torch.Tensor) -> List[torch.Tensor]:
        supports = [
            row_normalize_supports(adj[0]),
            row_normalize_supports(adj[1]),
            row_normalize_supports(adj[2]) * self.pe_graph_alpha,
        ]
        if self.use_adaptive_adj:
            adaptive = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=-1)
            supports.append(adaptive)
        return supports

    def encode(self, x: torch.Tensor, adj: torch.Tensor):
        # x: (B, T, N, F)
        supports = self._supports(adj)
        z = x.permute(0, 3, 2, 1).contiguous()
        z = self.input_proj(z)
        skip_sum = 0.0
        for block in self.blocks:
            z, skip = block(z, supports, self.pe_features.to(z.device))
            skip_sum = skip_sum + skip
        skip_sum = F.relu(skip_sum)
        context_seq = self.context_proj(skip_sum)
        context = context_seq[:, :, :, -1]
        coarse = self.end_proj(skip_sum)[:, :, :, -1]
        coarse = torch.sigmoid(coarse)
        return coarse, context, supports

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        y_noisy: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
        return_coarse: bool = False,
        y_target: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_diffusion: bool = False,
    ):
        y_coarse, context, supports = self.encode(x, adj)
        if not self.use_diffusion:
            pred = y_coarse
            diff_pred = pred
            diff_target = y_target if y_target is not None else pred
        else:
            if y_noisy is None:
                if y_target is None or t is None:
                    raise ValueError("diffusion forward requires y_noisy or (y_target, t)")
                y_noisy = self.diffusion.q_sample(y_target, t, noise)[0]
            if t is None:
                t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            pred = self.denoiser(
                y_noisy,
                y_coarse,
                context,
                t,
                supports,
                self.pe_features.to(x.device),
            )
            diff_pred = pred
            diff_target = y_target if y_target is not None else pred
        if return_coarse and return_diffusion:
            return pred, y_coarse, diff_pred, diff_target
        if return_coarse:
            return pred, y_coarse
        return pred

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        num_steps: int = 50,
        num_samples: int = 1,
        t_start_ratio: float = 0.25,
        coarse_only: bool = False,
        self_condition_mode: str = "prev_pred",
        self_condition_mix: float = 0.5,
    ) -> torch.Tensor:
        y_coarse, context, supports = self.encode(x, adj)
        if coarse_only or not self.use_diffusion:
            return y_coarse.clamp(0.0, 1.0)

        device = x.device
        bsz = x.shape[0]
        schedule = torch.linspace(
            self.diffusion.num_steps - 1,
            0,
            int(num_steps),
            dtype=torch.long,
            device=device,
        )
        start_idx = max(0, min(len(schedule) - 1, int(len(schedule) * float(t_start_ratio))))
        schedule = schedule[start_idx:]
        samples = []
        pe_features = self.pe_features.to(device)
        for _ in range(int(num_samples)):
            t_init = int(schedule[0].item())
            ab_init = self.diffusion.alpha_bars[t_init].to(device=device)
            x_t = torch.sqrt(ab_init) * y_coarse + torch.sqrt(1.0 - ab_init) * torch.randn_like(y_coarse)
            for i, t_now_tensor in enumerate(schedule):
                t_now = int(t_now_tensor.item())
                t_batch = torch.full((bsz,), t_now, dtype=torch.long, device=device)
                x0_pred = self.denoiser(x_t, y_coarse, context, t_batch, supports, pe_features).clamp(0.0, 1.0)
                if i + 1 < len(schedule):
                    ab_now = self.diffusion.alpha_bars[t_now].to(device=device)
                    ab_next = self.diffusion.alpha_bars[int(schedule[i + 1].item())].to(device=device)
                    eps = (x_t - torch.sqrt(ab_now) * x0_pred) / torch.sqrt(1.0 - ab_now).clamp(min=1e-8)
                    x_t = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1.0 - ab_next) * eps
                else:
                    x_t = x0_pred
            samples.append(x_t.clamp(0.0, 1.0))
        return torch.stack(samples, dim=0).mean(dim=0)
