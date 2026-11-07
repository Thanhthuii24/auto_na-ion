"""Run one locked autoresearch experiment with compute-only time accounting."""

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare
from workflow_preflight import validate


ROOT = Path(__file__).resolve().parent
RUNNER_VERSION = "r4-relative-huber-v2"


def checkpoint_candidate_is_better(
    candidate_metrics, selected_metrics, minimum_mape, tolerance
):
    """Apply the locked epsilon-Pareto checkpoint rule deterministically."""
    if selected_metrics is None:
        return True
    selected_is_eligible = selected_metrics["mape"] <= minimum_mape + tolerance
    if not selected_is_eligible:
        return True
    candidate_is_eligible = (
        candidate_metrics["mape"] <= minimum_mape + tolerance
    )
    if not candidate_is_eligible:
        return False
    if candidate_metrics["acc15"] != selected_metrics["acc15"]:
        return candidate_metrics["acc15"] > selected_metrics["acc15"]
    return candidate_metrics["mape"] < selected_metrics["mape"]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, inputs):
        hidden = self.dropout(F.relu(self.in_linear(inputs)))
        return self.ln(self.dropout(self.out_linear(hidden)) + inputs)


class CPMLP(nn.Module):
    def __init__(self, d_model=32, d_ff=256, e_layers=12, d_layers=7, dropout=0.0):
        super().__init__()
        self.intra_embed = nn.Linear(300 * 3, d_model)
        self.intra_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(e_layers)]
        )
        self.inter_flatten = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(100 * d_model, d_model)
        )
        self.inter_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(d_layers)]
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, curves, _mask=None):
        values = self.intra_embed(torch.flatten(curves, start_dim=2))
        for block in self.intra_mlp:
            values = block(values)
        values = self.inter_flatten(values)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        """Expose early, middle, and late representations for the baseline."""
        return {
            "intra_embed": self.intra_embed,
            "intra_mlp_first": self.intra_mlp[0],
            "intra_mlp_middle": self.intra_mlp[len(self.intra_mlp) // 2],
            "intra_mlp_last": self.intra_mlp[-1],
            "inter_flatten": self.inter_flatten,
            "inter_mlp_first": self.inter_mlp[0],
            "inter_mlp_middle": self.inter_mlp[len(self.inter_mlp) // 2],
            "inter_mlp_last": self.inter_mlp[-1],
            "head": self.head,
        }


class MaskedSummaryCPMLP(nn.Module):
    """CPMLP cycle encoder with prefix-aware level and change pooling."""

    def __init__(
        self,
        d_model=32,
        d_ff=256,
        e_layers=12,
        d_layers=7,
        dropout=0.0,
        summary_mode="mean_last_delta",
    ):
        super().__init__()
        if summary_mode not in {"mean", "mean_last_delta"}:
            raise ValueError(f"Unknown summary mode: {summary_mode}")
        self.summary_mode = summary_mode
        self.intra_embed = nn.Linear(300 * 3, d_model)
        self.intra_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(e_layers)]
        )
        summary_width = d_model if summary_mode == "mean" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.inter_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(d_layers)]
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("MaskedSummaryCPMLP requires a cycle mask")
        values = self.intra_embed(torch.flatten(curves, start_dim=2))
        for block in self.intra_mlp:
            values = block(values)
        weights = mask.to(values.dtype).unsqueeze(-1)
        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        first = values[:, 0]
        last_index = mask.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        values = self.inter_summary(summary)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "intra_embed": self.intra_embed,
            "intra_mlp_first": self.intra_mlp[0],
            "intra_mlp_middle": self.intra_mlp[len(self.intra_mlp) // 2],
            "intra_mlp_last": self.intra_mlp[-1],
            "inter_summary": self.inter_summary,
            "inter_mlp_first": self.inter_mlp[0],
            "inter_mlp_middle": self.inter_mlp[len(self.inter_mlp) // 2],
            "inter_mlp_last": self.inter_mlp[-1],
            "head": self.head,
        }


class GlobalGatedInnovationCPMLP(nn.Module):
    """E004 encoder with global-anchored adaptive innovation pooling."""

    def __init__(self, d_model=32, d_ff=256, e_layers=12, d_layers=7, dropout=0.0):
        super().__init__()
        self.intra_embed = nn.Linear(300 * 3, d_model)
        self.intra_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(e_layers)]
        )
        self.innovation_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.inter_summary = nn.Linear(d_model, d_model)
        self.inter_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(d_layers)]
        )
        self.head = nn.Linear(d_model, 1)
        self._gate_diagnostics = {}

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("GlobalGatedInnovationCPMLP requires a cycle mask")
        values = self.intra_embed(torch.flatten(curves, start_dim=2))
        for block in self.intra_mlp:
            values = block(values)

        weights = mask.to(values.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        anchor = (values * weights).sum(dim=1) / counts
        innovation = values - anchor.unsqueeze(1)
        steps = values.shape[1]
        positions = torch.arange(
            1, steps + 1, device=values.device, dtype=values.dtype
        ).view(1, steps, 1)
        relative_position = positions / counts.unsqueeze(1)
        gate_inputs = torch.cat(
            [anchor.unsqueeze(1).expand_as(values), innovation, relative_position],
            dim=-1,
        )
        gates = self.innovation_gate(gate_inputs) * weights
        gate_sum = gates.sum(dim=1).clamp_min(1e-6)
        update = (gates * innovation).sum(dim=1) / gate_sum
        latent_state = anchor + update

        with torch.no_grad():
            valid = mask.bool().unsqueeze(-1)
            valid_gates = gates.masked_select(valid)
            probabilities = gates.squeeze(-1) / gate_sum
            entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
            normalizer = mask.sum(dim=1).clamp_min(2.0).log()
            normalized_entropy = torch.where(
                mask.sum(dim=1) > 1,
                entropy / normalizer,
                torch.zeros_like(entropy),
            )
            self._gate_diagnostics = {
                "mean": float(valid_gates.mean().item()),
                "std": float(valid_gates.std(unbiased=False).item()),
                "min": float(valid_gates.min().item()),
                "max": float(valid_gates.max().item()),
                "normalized_entropy": float(normalized_entropy.mean().item()),
                "padding_leakage_max": float(
                    gates.masked_fill(valid, 0.0).abs().max().item()
                ),
                "update_to_anchor_norm": float(
                    (update.norm(dim=-1) / anchor.norm(dim=-1).clamp_min(1e-12))
                    .mean()
                    .item()
                ),
            }

        values = self.inter_summary(latent_state)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def gate_diagnostics(self):
        return dict(self._gate_diagnostics)

    def diagnostic_modules(self):
        return {
            "intra_embed": self.intra_embed,
            "intra_mlp": self.intra_mlp[-1],
            "innovation_gate": self.innovation_gate,
            "inter_summary": self.inter_summary,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }


class AttentionOnlyBlock(nn.Module):
    """Pre-norm self-attention residual block without a feed-forward MLP."""

    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )

    def forward(self, values, key_padding_mask, capture_attention=False):
        normalized = self.norm(values)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        return values + attended, weights


class ConvCycleAttention(nn.Module):
    """Non-MLP curve CNN plus mask-safe attention over observed cycles."""

    def __init__(self, d_model=64, n_heads=4, layers=6):
        super().__init__()
        self.cycle_encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, 101, d_model))
        self.attention_blocks = nn.ModuleList(
            [AttentionOnlyBlock(d_model, n_heads) for _ in range(layers)]
        )
        self.head = nn.Linear(d_model, 1)
        self.capture_attention = False
        self._attention_diagnostics = None
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ConvCycleAttention requires a cycle mask")
        batch, cycles = curves.shape[:2]
        encoded = self.cycle_encoder(curves.reshape(batch * cycles, 3, 300))
        encoded = encoded.squeeze(-1).reshape(batch, cycles, -1)
        observed = mask.bool()
        encoded = encoded * observed.unsqueeze(-1).to(encoded.dtype)
        cls = self.cls_token.expand(batch, -1, -1)
        values = torch.cat([cls, encoded], dim=1) + self.position[:, : cycles + 1]
        valid_tokens = torch.cat(
            [torch.ones(batch, 1, dtype=torch.bool, device=mask.device), observed],
            dim=1,
        )
        key_padding_mask = ~valid_tokens
        captured = None
        for index, block in enumerate(self.attention_blocks):
            capture = self.capture_attention and index == len(self.attention_blocks) - 1
            values, weights = block(values, key_padding_mask, capture)
            if capture:
                captured = weights
        if captured is not None:
            padded_mass = (
                captured * key_padding_mask[:, None, None, :].to(captured.dtype)
            ).sum(dim=-1)
            cls_weights = captured[:, :, 0, 1:] * observed[:, None, :].to(captured.dtype)
            cls_weights = cls_weights / cls_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(cls_weights.clamp_min(1e-12).log() * cls_weights).sum(dim=-1)
            normalizer = observed.sum(dim=1).clamp_min(2).to(entropy.dtype).log()[:, None]
            self._attention_diagnostics = {
                "max_padding_attention": float(padded_mass.max().detach().cpu()),
                "mean_normalized_cls_entropy": float((entropy / normalizer).mean().detach().cpu()),
            }
        return self.head(values[:, 0]).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_encoder": self.cycle_encoder,
            "cycle_attention": self.attention_blocks[-1],
            "head": self.head,
        }

    def attention_diagnostics(self):
        return self._attention_diagnostics


class ConvCycleGRU(nn.Module):
    """Non-MLP curve CNN plus gated recurrence over observed cycles."""

    def __init__(self, d_model=64, hidden_size=128, layers=2):
        super().__init__()
        self.cycle_encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.temporal_gru = nn.GRU(
            d_model, hidden_size, num_layers=layers, batch_first=True, dropout=0.0
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ConvCycleGRU requires a cycle mask")
        batch, cycles = curves.shape[:2]
        encoded = self.cycle_encoder(curves.reshape(batch * cycles, 3, 300))
        encoded = encoded.squeeze(-1).reshape(batch, cycles, -1)
        observed = mask.bool()
        encoded = encoded * observed.unsqueeze(-1).to(encoded.dtype)
        sequence, _ = self.temporal_gru(encoded)
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = sequence.gather(
            1, last_index[:, None, None].expand(-1, 1, sequence.shape[-1])
        ).squeeze(1)
        return self.head(last).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_encoder": self.cycle_encoder,
            "temporal_gru": self.temporal_gru,
            "head": self.head,
        }


class SimpleConvMean(nn.Module):
    """Strided curve CNN with masked mean pooling and a linear head."""

    def __init__(self, d_model=64):
        super().__init__()
        self.cycle_encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mean_pool = nn.Identity()
        self.head = nn.Linear(d_model, 1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("SimpleConvMean requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_encoder(curves.reshape(batch * cycles, 3, 300))
        values = values.squeeze(-1).reshape(batch, cycles, -1)
        weights = mask.to(values.dtype).unsqueeze(-1)
        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        mean = self.mean_pool(mean)
        return self.head(mean).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_encoder": self.cycle_encoder,
            "mean_pool": self.mean_pool,
            "head": self.head,
        }


class ResidualCurveBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, values):
        residual = values
        values = F.relu(self.norm1(self.conv1(values)))
        values = self.norm2(self.conv2(values))
        return F.relu(values + residual)


class NormalizedCurveBlock(nn.Module):
    """Matched ResidualCurveBlock transform without the residual addition."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, values):
        values = F.relu(self.norm1(self.conv1(values)))
        return F.relu(self.norm2(self.conv2(values)))


class MultiScaleResidualCurveBlock(nn.Module):
    """Efficient local-and-wide depthwise branches with residual fusion."""

    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=channels,
                )
                for kernel_size in (3, 7, 15)
            ]
        )
        self.fuse = nn.Conv1d(3 * channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, values):
        transformed = torch.cat(
            [F.relu(branch(values)) for branch in self.branches], dim=1
        )
        transformed = self.norm(self.fuse(transformed))
        return F.relu(values + transformed)


class MeanMaxPool1d(nn.Module):
    def forward(self, values):
        average = F.adaptive_avg_pool1d(values, 1).squeeze(-1)
        maximum = F.adaptive_max_pool1d(values, 1).squeeze(-1)
        return torch.cat([average, maximum], dim=1)


class MaskedTemporalConvBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.left_padding = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, values, mask):
        transformed = F.pad(values, (self.left_padding, 0))
        transformed = self.pointwise(self.depthwise(transformed))
        values = F.relu(values + self.norm(transformed))
        return values * mask


class MaskedTemporalPyramidPool(nn.Module):
    """Average observed states in dynamic normalized-prefix segments."""

    def __init__(self, levels=(1, 2, 4)):
        super().__init__()
        self.levels = tuple(levels)

    @property
    def output_bins(self):
        return sum(self.levels)

    def forward(self, values, observed):
        batch, cycles, _ = values.shape
        positions = torch.arange(cycles, device=values.device)[None, :]
        lengths = observed.sum(dim=1).long().clamp_min(1)
        pooled = []
        for bins in self.levels:
            assignments = torch.div(
                positions * bins,
                lengths[:, None],
                rounding_mode="floor",
            ).clamp_max(bins - 1)
            for bin_index in range(bins):
                selected = observed & assignments.eq(bin_index)
                weights = selected.unsqueeze(-1).to(values.dtype)
                pooled.append(
                    (values * weights).sum(dim=1)
                    / weights.sum(dim=1).clamp_min(1.0)
                )
        return torch.cat(pooled, dim=-1)


class ResidualConvMean(nn.Module):
    """E008 stem plus residual-normalized curve depth and fixed masked mean."""

    def __init__(self, d_model=64, residual_layers=3, residual_dilation=1):
        super().__init__()
        self.cycle_stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[
                ResidualCurveBlock(d_model, dilation=residual_dilation)
                for _ in range(residual_layers)
            ]
        )
        self.curve_pool = nn.AdaptiveAvgPool1d(1)
        self.mean_pool = nn.Identity()
        self.head = nn.Linear(d_model, 1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ResidualConvMean requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)
        weights = mask.to(values.dtype).unsqueeze(-1)
        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        mean = self.mean_pool(mean)
        return self.head(mean).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "mean_pool": self.mean_pool,
            "head": self.head,
        }


class ResidualConvAttentionLast(nn.Module):
    """E010 curve encoder plus one full attention block and last observed state."""

    def __init__(self, d_model=64, n_heads=4, d_ff=128):
        super().__init__()
        self.cycle_stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(ResidualCurveBlock(d_model, dilation=1))
        self.curve_pool = nn.AdaptiveAvgPool1d(1)
        self.position = nn.Parameter(torch.zeros(1, 100, d_model))
        self.attention_norm1 = nn.LayerNorm(d_model)
        self.cycle_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )
        self.attention_norm2 = nn.LayerNorm(d_model)
        self.attention_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.head = nn.Linear(d_model, 1)
        self.capture_attention = False
        self._attention_diagnostics = {}
        nn.init.normal_(self.position, std=0.02)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ResidualConvAttentionLast requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)
        values = (values + self.position[:, :cycles]) * weights

        normalized = self.attention_norm1(values)
        attended, attention = self.cycle_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~observed,
            need_weights=self.capture_attention,
            average_attn_weights=False,
        )
        values = (values + attended) * weights
        values = (values + self.attention_ffn(self.attention_norm2(values))) * weights

        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)

        if self.capture_attention:
            last_weights = attention.gather(
                2,
                last_index[:, None, None, None].expand(
                    -1, attention.shape[1], 1, attention.shape[-1]
                ),
            ).squeeze(2)
            padding_mass = (
                last_weights * (~observed)[:, None, :].to(last_weights.dtype)
            ).sum(dim=-1)
            observed_weights = last_weights * observed[:, None, :].to(last_weights.dtype)
            observed_weights = observed_weights / observed_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            entropy = -(
                observed_weights.clamp_min(1e-12).log() * observed_weights
            ).sum(dim=-1)
            normalizer = observed.sum(dim=1).clamp_min(2).to(entropy.dtype).log()[:, None]
            self._attention_diagnostics = {
                "max_padding_attention": float(padding_mass.max().detach().cpu()),
                "mean_normalized_last_entropy": float(
                    (entropy / normalizer).mean().detach().cpu()
                ),
            }

        return self.head(last).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "cycle_attention": self.cycle_attention,
            "attention_ffn": self.attention_ffn,
            "head": self.head,
        }

    def attention_diagnostics(self):
        return dict(self._attention_diagnostics)


class NormalizedConvMean(ResidualConvMean):
    """E010 matched ablation with the residual addition removed."""

    def __init__(self, d_model=64, layers=1):
        super().__init__(d_model=d_model, residual_layers=0)
        self.normalized_blocks = nn.Sequential(
            *[NormalizedCurveBlock(d_model, dilation=1) for _ in range(layers)]
        )
        del self.residual_blocks

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("NormalizedConvMean requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.normalized_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)
        weights = mask.to(values.dtype).unsqueeze(-1)
        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        mean = self.mean_pool(mean)
        return self.head(mean).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "normalized_blocks": self.normalized_blocks,
            "mean_pool": self.mean_pool,
            "head": self.head,
        }


class MultiScaleResidualConvMean(ResidualConvMean):
    """E010 stem and aggregation with an efficient multi-scale residual block."""

    def __init__(self, d_model=64):
        super().__init__(d_model=d_model, residual_layers=0)
        self.multiscale_block = MultiScaleResidualCurveBlock(d_model)
        del self.residual_blocks

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("MultiScaleResidualConvMean requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.multiscale_block(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)
        weights = mask.to(values.dtype).unsqueeze(-1)
        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        mean = self.mean_pool(mean)
        return self.head(mean).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "multiscale_block": self.multiscale_block,
            "mean_pool": self.mean_pool,
            "head": self.head,
        }


class ResidualConvMeanMax(ResidualConvMean):
    """E010 encoder with concatenated mean/max per-curve statistics."""

    def __init__(self, d_model=64):
        super().__init__(d_model=d_model, residual_layers=1, residual_dilation=1)
        self.curve_pool = MeanMaxPool1d()
        self.head = nn.Linear(2 * d_model, 1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ResidualConvMeanMax requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).reshape(batch, cycles, -1)
        weights = mask.to(values.dtype).unsqueeze(-1)
        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        mean = self.mean_pool(mean)
        return self.head(mean).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "curve_pool": self.curve_pool,
            "mean_pool": self.mean_pool,
            "head": self.head,
        }


class ResidualConvTemporalTCNMean(ResidualConvMean):
    """E010 curve encoder plus mask-safe causal temporal convolution."""

    def __init__(self, d_model=64, temporal_dilations=(1, 2)):
        super().__init__(d_model=d_model, residual_layers=1, residual_dilation=1)
        self.temporal_tcn = nn.ModuleList(
            [
                MaskedTemporalConvBlock(d_model, dilation)
                for dilation in temporal_dilations
            ]
        )

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ResidualConvTemporalTCNMean requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)
        weights = mask.to(values.dtype).unsqueeze(-1)
        values = values * weights
        temporal_mask = weights.transpose(1, 2)
        values = values.transpose(1, 2)
        for block in self.temporal_tcn:
            values = block(values, temporal_mask)
        values = values.transpose(1, 2)
        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        mean = self.mean_pool(mean)
        return self.head(mean).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "temporal_tcn": self.temporal_tcn[-1],
            "mean_pool": self.mean_pool,
            "head": self.head,
        }


class MultiScaleCurveEncoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv1d(3, 16, kernel_size=k, padding=k // 2) for k in (3, 7, 15)]
        )
        self.fuse = nn.Conv1d(48, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualCurveBlock(channels, dilation) for dilation in (1, 2, 4)]
        )

    def forward(self, curves):
        values = torch.cat([F.relu(branch(curves)) for branch in self.branches], dim=1)
        values = F.relu(self.fuse(values))
        for block in self.blocks:
            values = block(values)
        average = F.adaptive_avg_pool1d(values, 1).squeeze(-1)
        maximum = F.adaptive_max_pool1d(values, 1).squeeze(-1)
        return torch.cat([average, maximum], dim=1)


class CausalTCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.left_padding = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, values):
        transformed = values.transpose(1, 2)
        transformed = F.pad(transformed, (self.left_padding, 0))
        transformed = self.pointwise(self.depthwise(transformed)).transpose(1, 2)
        return F.relu(self.norm(values + transformed))


class MultiScaleCurveTCN(nn.Module):
    """Non-MLP multi-scale curve CNN and causal dilated cycle CNN."""

    def __init__(
        self,
        curve_channels=64,
        temporal_channels=128,
        temporal_dilations=(1, 2, 4, 8, 16, 32, 64),
    ):
        super().__init__()
        if temporal_channels != 2 * curve_channels:
            raise ValueError("temporal_channels must equal pooled curve width")
        self.curve_encoder = MultiScaleCurveEncoder(curve_channels)
        self.temporal_tcn = nn.ModuleList(
            [
                CausalTCNBlock(temporal_channels, dilation)
                for dilation in temporal_dilations
            ]
        )
        self.head = nn.Linear(temporal_channels, 1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("MultiScaleCurveTCN requires a cycle mask")
        batch, cycles = curves.shape[:2]
        observed = mask.bool()
        flat_curves = curves.reshape(batch * cycles, 3, 300)
        values = self.curve_encoder(flat_curves).reshape(batch, cycles, -1)
        weights = observed.unsqueeze(-1).to(values.dtype)
        values = values * weights
        for block in self.temporal_tcn:
            values = block(values) * weights
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        return self.head(last).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "curve_encoder": self.curve_encoder,
            "temporal_tcn": self.temporal_tcn[-1],
            "head": self.head,
        }


class ResidualConvMaskedSummary(nn.Module):
    """Hybrid 1D ResNet curve encoder + prefix-aware masked summary pooling + MLP."""

    def __init__(
        self,
        d_model=64,
        residual_layers=1,
        d_ff=256,
        mlp_layers=4,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__()
        self.summary_mode = summary_mode
        self.cycle_stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualCurveBlock(d_model, dilation=1) for _ in range(residual_layers)]
        )
        self.curve_pool = nn.AdaptiveAvgPool1d(1)
        summary_width = d_model if summary_mode == "mean" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.inter_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(mlp_layers)]
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ResidualConvMaskedSummary requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

        weights = mask.to(values.dtype).unsqueeze(-1)
        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        first = values[:, 0]
        last_index = mask.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        values = self.inter_summary(summary)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "inter_summary": self.inter_summary,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }


class ResidualConvTemporalTCNMaskedSummary(ResidualConvMaskedSummary):
    """R4-E005 curve CNN plus causal masked cycle TCN before fixed summary."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        temporal_dilations=(1, 2, 4),
        d_ff=256,
        mlp_layers=7,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        self.temporal_tcn = nn.ModuleList(
            [MaskedTemporalConvBlock(d_model, dilation) for dilation in temporal_dilations]
        )

    def encode_curve_features(self, curves):
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        return self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

    def encode_temporal_summary(self, values, mask):
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)
        temporal_mask = weights.transpose(1, 2)
        values = (values * weights).transpose(1, 2)
        for block in self.temporal_tcn:
            values = block(values, temporal_mask)
        values = values.transpose(1, 2) * weights

        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = values.sum(dim=1) / counts
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        return self.inter_summary(summary)

    def encode(self, curves, mask):
        values = self.encode_curve_features(curves)
        return self.encode_temporal_summary(values, mask)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError(
                "ResidualConvTemporalTCNMaskedSummary requires a cycle mask"
            )
        values = self.encode(curves, mask)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "temporal_tcn_first": self.temporal_tcn[0],
            "temporal_tcn_last": self.temporal_tcn[-1],
            "inter_summary": self.inter_summary,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }


class ReferencePairTemporalTCN(ResidualConvTemporalTCNMaskedSummary):
    """BatLiNet-style reference-pair learning over R4-E006 causal TCN cycle representations."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        temporal_dilations=(1, 2, 4),
        d_ff=256,
        summary_mode="mean_last_delta",
        dropout=0.0,
        k_references=16,
        shuffle_reference_labels=False,
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            temporal_dilations=temporal_dilations,
            d_ff=d_ff,
            mlp_layers=0,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        del self.inter_mlp
        del self.head

        self.k_references = k_references
        self.shuffle_reference_labels = shuffle_reference_labels
        self.register_buffer("ref_curves", torch.zeros(k_references, 100, 3, 300))
        self.register_buffer("ref_labels", torch.zeros(k_references))
        self.scaler_mean = 0.0
        self.scaler_scale = 1.0
        self.ref_cell_ids = []
        self.is_reference_bank_set = False

        self.pair_head = nn.Sequential(
            nn.Linear(3 * d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def set_reference_bank(self, dataset_or_cells, scaler=None):
        if hasattr(dataset_or_cells, "cells"):
            cells = dataset_or_cells.cells
            scaler = dataset_or_cells.scaler
        else:
            cells = dataset_or_cells

        if scaler is not None:
            self.scaler_mean = float(scaler.mean)
            self.scaler_scale = float(scaler.scale)

        sorted_cells = sorted(cells, key=lambda c: (float(c["eol"]), str(c.get("filename", ""))))
        n = len(sorted_cells)
        rng = np.random.RandomState(2021)
        k = self.k_references
        picks = []
        for i in range(k):
            start = int(i * n / k)
            end = int((i + 1) * n / k)
            picks.append(rng.randint(start, end))

        selected = [sorted_cells[p] for p in picks]
        ref_tensors = torch.stack([c["tensor"] for c in selected], dim=0)
        ref_eols = torch.tensor([float(c["eol"]) for c in selected], dtype=torch.float32)

        if self.shuffle_reference_labels:
            perm_rng = np.random.RandomState(2021)
            perm = torch.from_numpy(perm_rng.permutation(len(ref_eols)))
            ref_eols = ref_eols[perm]

        self.ref_curves.copy_(ref_tensors.to(self.ref_curves.device, dtype=self.ref_curves.dtype))
        self.ref_labels.copy_(ref_eols.to(self.ref_labels.device, dtype=self.ref_labels.dtype))
        self.ref_cell_ids = [c.get("filename", "") for c in selected]
        self.is_reference_bank_set = True

    def forward(self, curves, mask, cell_ids=None):
        if mask is None:
            raise ValueError("ReferencePairTemporalTCN requires a cycle mask")

        curves = curves * mask.unsqueeze(-1).unsqueeze(-1)
        batch = curves.shape[0]

        query_cycles = self.encode_curve_features(curves)
        h_x = self.encode_temporal_summary(query_cycles, mask)

        if not self.is_reference_bank_set and self.ref_labels.sum() == 0:
            return torch.zeros(batch, device=curves.device, dtype=curves.dtype)

        ref_curves = self.ref_curves.to(curves.device, dtype=curves.dtype)
        ref_cycles = self.encode_curve_features(ref_curves)

        k = self.k_references
        ref_cycles_exp = ref_cycles.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * k, 100, -1)
        mask_exp = mask.unsqueeze(1).expand(-1, k, -1).reshape(batch * k, 100)
        h_ref = self.encode_temporal_summary(ref_cycles_exp, mask_exp).reshape(batch, k, -1)

        h_x_exp = h_x.unsqueeze(1).expand(-1, k, -1)
        diff = h_x_exp - h_ref
        pair_in = torch.cat([h_x_exp, h_ref, diff], dim=-1)
        g = self.pair_head(pair_in).squeeze(-1).clamp(min=-10.0, max=10.0)

        ref_labels = self.ref_labels.to(curves.device, dtype=curves.dtype)
        y_k = ref_labels.unsqueeze(0) * torch.exp(g)

        valid = torch.ones(batch, k, dtype=torch.bool, device=curves.device)
        if cell_ids is not None and len(self.ref_cell_ids) == k:
            for b in range(batch):
                cid = cell_ids[b]
                for r in range(k):
                    if cid == self.ref_cell_ids[r]:
                        valid[b, r] = False
        else:
            diff_c0 = (curves[:, None, 0] - ref_curves[None, :, 0]).abs().amax(dim=(-2, -1))
            self_match = (diff_c0 < 1e-4) & (curves[:, 0].abs().sum(dim=(-2, -1)) > 0)[:, None]  # [B,K]
            valid = valid & (~self_match)

        count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        y_raw = (y_k * valid.to(y_k.dtype)).sum(dim=-1) / count.squeeze(-1)

        scale = self.scaler_scale if self.scaler_scale != 0 else 1.0
        y_std = (y_raw - self.scaler_mean) / scale
        return y_std

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "temporal_tcn_first": self.temporal_tcn[0],
            "temporal_tcn_last": self.temporal_tcn[-1],
            "inter_summary": self.inter_summary,
            "pair_head_first": self.pair_head[0],
            "pair_head_last": self.pair_head[-1],
        }


class ResidualConvTemporalTCNPhysicalDifferential(
    ResidualConvTemporalTCNMaskedSummary
):
    """R4-E006 with one phase-local stabilized dQ/dV input channel."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        temporal_dilations=(1, 2, 4),
        d_ff=256,
        mlp_layers=7,
        summary_mode="mean_last_delta",
        dropout=0.0,
        phase_split=150,
        derivative_epsilon=1e-4,
        derivative_transform="none",
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            temporal_dilations=temporal_dilations,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        if phase_split != 150:
            raise ValueError("The locked 300-point curve requires phase_split=150")
        if derivative_epsilon <= 0:
            raise ValueError("derivative_epsilon must be positive")
        self.phase_split = phase_split
        self.derivative_epsilon = derivative_epsilon
        if derivative_transform not in {"none", "asinh"}:
            raise ValueError("derivative_transform must be 'none' or 'asinh'")
        self.derivative_transform = derivative_transform
        self.cycle_stem[0] = nn.Conv1d(
            4, 32, kernel_size=7, stride=2, padding=3
        )

    def _phase_derivative(self, voltage, capacity, start, end):
        d_voltage = voltage[..., start + 1 : end] - voltage[..., start : end - 1]
        d_capacity = capacity[..., start + 1 : end] - capacity[..., start : end - 1]
        sign = torch.where(
            d_voltage >= 0,
            torch.ones_like(d_voltage),
            -torch.ones_like(d_voltage),
        )
        denominator = sign * d_voltage.abs().clamp_min(self.derivative_epsilon)
        derivative = d_capacity / denominator
        return torch.cat([torch.zeros_like(derivative[..., :1]), derivative], dim=-1)

    def augment_curves(self, curves):
        voltage = curves[..., 0, :]
        capacity = curves[..., 2, :]
        charge = self._phase_derivative(
            voltage, capacity, 0, self.phase_split
        )
        discharge = self._phase_derivative(
            voltage, capacity, self.phase_split, curves.shape[-1]
        )
        derivative = torch.cat([charge, discharge], dim=-1).unsqueeze(-2)
        if self.derivative_transform == "asinh":
            derivative = torch.asinh(derivative)
        return torch.cat([curves, derivative], dim=-2)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError(
                "ResidualConvTemporalTCNPhysicalDifferential requires a cycle mask"
            )
        batch, cycles = curves.shape[:2]
        augmented = self.augment_curves(curves)
        values = self.cycle_stem(augmented.reshape(batch * cycles, 4, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)
        temporal_mask = weights.transpose(1, 2)
        values = (values * weights).transpose(1, 2)
        for block in self.temporal_tcn:
            values = block(values, temporal_mask)
        values = values.transpose(1, 2) * weights

        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = values.sum(dim=1) / counts
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        values = self.inter_summary(summary)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)


class DualPhaseCoordConvTemporalTCN(nn.Module):
    """Phase-aware curve encoder followed by the retained R4-E006 TCN path."""

    def __init__(
        self,
        d_model=32,
        temporal_dilations=(1, 2, 4),
        d_ff=256,
        mlp_layers=7,
        summary_mode="mean_last_delta",
        dropout=0.0,
        phase_split=150,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("DualPhaseCoordConvTemporalTCN requires even d_model")
        if phase_split != 150:
            raise ValueError("The locked 300-point curve requires phase_split=150")
        self.d_model = d_model
        self.phase_split = phase_split
        self.summary_mode = summary_mode
        phase_width = d_model // 2

        def phase_encoder():
            return nn.Sequential(
                nn.Conv1d(4, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(16, 24, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(24, phase_width, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

        self.charge_encoder = phase_encoder()
        self.discharge_encoder = phase_encoder()
        self.temporal_tcn = nn.ModuleList(
            [MaskedTemporalConvBlock(d_model, dilation) for dilation in temporal_dilations]
        )
        summary_width = d_model if summary_mode == "mean" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.inter_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(mlp_layers)]
        )
        self.head = nn.Linear(d_model, 1)

    def _encode_phase(self, values, encoder):
        position = torch.linspace(
            0.0,
            1.0,
            values.shape[-1],
            device=values.device,
            dtype=values.dtype,
        )
        position = position.view(1, 1, -1).expand(values.shape[0], 1, -1)
        return encoder(torch.cat([values, position], dim=1)).squeeze(-1)

    def _fuse_phase_states(self, charge_state, discharge_state):
        return torch.cat([charge_state, discharge_state], dim=-1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("DualPhaseCoordConvTemporalTCN requires a cycle mask")
        if curves.shape[-1] != 2 * self.phase_split:
            raise ValueError("Dual-phase curve input must contain exactly 300 points")
        batch, cycles = curves.shape[:2]
        flat_curves = curves.reshape(batch * cycles, 3, curves.shape[-1])
        charge = flat_curves[:, :, : self.phase_split]
        discharge = flat_curves[:, :, self.phase_split :]
        charge_state = self._encode_phase(charge, self.charge_encoder)
        discharge_state = self._encode_phase(discharge, self.discharge_encoder)
        values = self._fuse_phase_states(charge_state, discharge_state)
        values = values.reshape(batch, cycles, self.d_model)

        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)
        temporal_mask = weights.transpose(1, 2)
        values = (values * weights).transpose(1, 2)
        for block in self.temporal_tcn:
            values = block(values, temporal_mask)
        values = values.transpose(1, 2) * weights

        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = values.sum(dim=1) / counts
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        values = self.inter_summary(summary)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "charge_encoder": self.charge_encoder,
            "discharge_encoder": self.discharge_encoder,
            "temporal_tcn_first": self.temporal_tcn[0],
            "temporal_tcn_last": self.temporal_tcn[-1],
            "inter_summary": self.inter_summary,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }


class DualPhaseResidualCoordConvTemporalTCN(DualPhaseCoordConvTemporalTCN):
    """Capacity-matched repair of the dual-phase CoordConv curve encoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        phase_width = self.d_model // 2

        def phase_encoder():
            return nn.Sequential(
                nn.Conv1d(4, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, phase_width, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                ResidualCurveBlock(phase_width, dilation=1),
                nn.AdaptiveAvgPool1d(1),
            )

        self.charge_encoder = phase_encoder()
        self.discharge_encoder = phase_encoder()


class DualPhaseCoordConvPhaseContrastTCN(DualPhaseResidualCoordConvTemporalTCN):
    """Dual-phase TCN with an explicit within-cycle phase contrast."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        phase_width = self.d_model // 2
        self.phase_contrast_projection = nn.Linear(
            self.d_model + phase_width, self.d_model
        )
        with torch.no_grad():
            self.phase_contrast_projection.weight.zero_()
            self.phase_contrast_projection.weight[:, : self.d_model].copy_(
                torch.eye(self.d_model)
            )
            self.phase_contrast_projection.bias.zero_()

    def _fuse_phase_states(self, charge_state, discharge_state):
        phase_contrast = discharge_state - charge_state
        return self.phase_contrast_projection(
            torch.cat([charge_state, discharge_state, phase_contrast], dim=-1)
        )

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["phase_contrast_projection"] = self.phase_contrast_projection
        return modules


class DualPhaseCoordConvResidualContrastTCN(DualPhaseResidualCoordConvTemporalTCN):
    """Exact dual-phase base path plus a learned additive phase contrast."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        phase_width = self.d_model // 2
        self.residual_contrast_projection = nn.Linear(
            phase_width, self.d_model, bias=False
        )
        with torch.no_grad():
            self.residual_contrast_projection.weight.zero_()

    def _fuse_phase_states(self, charge_state, discharge_state):
        base = torch.cat([charge_state, discharge_state], dim=-1)
        contrast = discharge_state - charge_state
        return base + self.residual_contrast_projection(contrast)

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["residual_contrast_projection"] = self.residual_contrast_projection
        return modules


class DualPhaseCoordConvAnchorTCN(DualPhaseResidualCoordConvTemporalTCN):
    """R4-E021 dual-phase features with a Cycle-1-relative TCN input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_projection = nn.Conv1d(2 * self.d_model, self.d_model, 1)
        with torch.no_grad():
            self.anchor_projection.weight.zero_()
            identity = torch.eye(self.d_model).unsqueeze(-1)
            self.anchor_projection.weight[:, : self.d_model].copy_(identity)
            self.anchor_projection.bias.zero_()

    def _anchor_relative(self, values, weights):
        anchor = values[:, :1]
        return (values - anchor) * weights

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("DualPhaseCoordConvAnchorTCN requires a cycle mask")
        if curves.shape[-1] != 2 * self.phase_split:
            raise ValueError("Dual-phase curve input must contain exactly 300 points")
        observed = mask.bool()
        if not bool(observed[:, 0].all()):
            raise ValueError("Cycle-1 anchor must be observed for every sample")
        batch, cycles = curves.shape[:2]
        flat_curves = curves.reshape(batch * cycles, curves.shape[2], curves.shape[-1])
        charge = flat_curves[:, :, : self.phase_split]
        discharge = flat_curves[:, :, self.phase_split :]
        charge_state = self._encode_phase(charge, self.charge_encoder)
        discharge_state = self._encode_phase(discharge, self.discharge_encoder)
        values = torch.cat([charge_state, discharge_state], dim=-1)
        values = values.reshape(batch, cycles, self.d_model)

        weights = observed.unsqueeze(-1).to(values.dtype)
        values = values * weights
        relative = self._anchor_relative(values, weights)
        anchored = torch.cat([values, relative], dim=-1).transpose(1, 2)
        values = self.anchor_projection(anchored).transpose(1, 2) * weights

        temporal_mask = weights.transpose(1, 2)
        values = values.transpose(1, 2)
        for block in self.temporal_tcn:
            values = block(values, temporal_mask)
        values = values.transpose(1, 2) * weights

        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = values.sum(dim=1) / counts
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        summary = self._condition_summary(summary)
        values = self.inter_summary(summary)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def _condition_summary(self, summary):
        return summary

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["anchor_projection"] = self.anchor_projection
        return modules


class MetaConditionedAnchorTCN(DualPhaseCoordConvAnchorTCN):
    """A7: E025 + test-condition metadata (log SOC width, SOC start, log1p charge/discharge C-rate).
    Input carries a 4th channel; metadata sits at [cycle 0, channel 3, points 0:4]. A zero-initialized
    linear map adds the metadata to the cycle summary, so at initialization the model equals E025."""

    META_DIM = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_projection = nn.Linear(self.META_DIM, self.inter_summary.in_features)
        with torch.no_grad():
            self.meta_projection.weight.zero_(); self.meta_projection.bias.zero_()
        self._meta = None

    def forward(self, curves, mask):
        if curves.shape[2] == 4:
            self._meta = curves[:, 0, 3, : self.META_DIM]
            curves = curves[:, :, :3]
        else:
            self._meta = None
        return super().forward(curves, mask)

    def _condition_summary(self, summary):
        if self._meta is None:
            return summary
        return summary + self.meta_projection(self._meta.to(summary.dtype))

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["meta_projection"] = self.meta_projection
        return modules


class DeltaQFingerprintAnchorTCN(DualPhaseCoordConvAnchorTCN):
    """R4-E072: E025 learned stream + explicit raw-curve fingerprint stream.
    Features come from the raw discharge curves (no learned encoder):
    Severson-style log-variance/min/mean of dQ = Q_last - Q_1 and Q_last - Q_mid,
    log-var of dV, a masked least-squares fade slope of end-of-discharge capacity,
    capacity ratio and prefix length. BatchNorm -> MLP -> zero-initialised projection
    added to the cycle summary, so at initialization the model equals E025."""

    FEAT_DIM = 12
    CHARGE_DIM = 6

    LATE_DIM = 7

    def __init__(self, *args, charge_features=False, unbounded_head=False, late_fade_features=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.charge_features = bool(charge_features)
        self.late_fade_features = bool(late_fade_features)
        self.FEAT_DIM = (DeltaQFingerprintAnchorTCN.FEAT_DIM + (self.CHARGE_DIM if self.charge_features else 0)
                         + (self.LATE_DIM if self.late_fade_features else 0))
        self._dual_q = None
        self.unbounded_head = bool(unbounded_head)
        if self.unbounded_head:
            # R4-E080: the last MLPBlock ends in LayerNorm, so head(relu(LN(.))) is norm-bounded and
            # predictions saturate (E076: max ~1880 cycles even on train cells with EOL 3000-5000).
            # A learnable log-gain on the head output and a direct linear fingerprint path (both
            # neutral at init) remove the ceiling.
            self.head_log_gain = nn.Parameter(torch.zeros(()))
            self.fp_direct = nn.Linear(self.FEAT_DIM, 1)
            with torch.no_grad():
                self.fp_direct.weight.zero_(); self.fp_direct.bias.zero_()
        self._z = None
        width = self.inter_summary.in_features
        # fixed standardisation from the training set (set_reference_bank); no batch statistics,
        # so truncated / what-if branches used by Phase-2 losses see the same normalisation.
        self.register_buffer("fp_mean", torch.zeros(self.FEAT_DIM))
        self.register_buffer("fp_std", torch.ones(self.FEAT_DIM))
        self.fp_mlp = nn.Sequential(nn.Linear(self.FEAT_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.fp_projection = nn.Linear(64, width)
        with torch.no_grad():
            self.fp_projection.weight.zero_(); self.fp_projection.bias.zero_()
        self._fp = None

    @staticmethod
    def _logstat(x, eps):
        return torch.log(x.abs() + eps)

    def _fingerprint(self, curves, mask):
        curves = curves[:, :, :3].float()
        observed = mask.bool()
        count = observed.sum(dim=1).long().clamp_min(1)
        idx = torch.arange(curves.shape[0], device=curves.device)
        last = count - 1
        mid = (count // 2).clamp_min(0)
        disch = curves[:, :, :, 150:]
        q, v = disch[:, :, 2], disch[:, :, 0]
        q1, ql, qm = q[:, 0], q[idx, last], q[idx, mid]
        v1, vl = v[:, 0], v[idx, last]
        dq, dqm, dv = ql - q1, ql - qm, vl - v1
        cap = q[:, :, -1]                      # end-of-discharge capacity per cycle
        w = observed.float()
        t = torch.arange(q.shape[1], device=q.device, dtype=q.dtype).unsqueeze(0)
        n = w.sum(1).clamp_min(1.0)
        tm = (w * t).sum(1) / n
        cm = (w * cap).sum(1) / n
        var_t = (w * (t - tm.unsqueeze(1)) ** 2).sum(1).clamp_min(1e-6)
        slope = (w * (t - tm.unsqueeze(1)) * (cap - cm.unsqueeze(1))).sum(1) / var_t
        ratio = (cap[idx, last] / cap[:, 0].abs().clamp_min(1e-3)).clamp(0.0, 2.0)
        feats = torch.stack([
            torch.log(dq.var(dim=1) + 1e-10),
            self._logstat(dq.min(dim=1).values, 1e-5),
            self._logstat(dq.mean(dim=1), 1e-5),
            torch.log(dqm.var(dim=1) + 1e-10),
            self._logstat(dqm.mean(dim=1), 1e-5),
            torch.log(dv.var(dim=1) + 1e-10),
            self._logstat(slope, 1e-7),
            torch.sign(slope),
            ratio,
            cap[:, 0],
            torch.log(count.float()),
            (count.float() >= 2).float(),
        ], dim=1)
        if self.charge_features:
            # charge side (points 0:150): voltage rise (resistance growth), charge-capacity change,
            # and constant-voltage share (points with current below 90% of the cycle maximum).
            charge = curves[:, :, :, :150]
            cv, ci, cq = charge[:, :, 0], charge[:, :, 1], charge[:, :, 2]
            dvc = cv[idx, last] - cv[:, 0]
            dqc = cq[idx, last] - cq[:, 0]
            imax = ci.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            cv_share = (ci.abs() < 0.9 * imax).float().mean(dim=-1)  # [B, cycles]
            cvs_last, cvs_first = cv_share[idx, last], cv_share[:, 0]
            cvs_slope = (w * (t - tm.unsqueeze(1)) * (cv_share - ((w * cv_share).sum(1) / n).unsqueeze(1))).sum(1) / var_t
            extra = torch.stack([
                torch.log(dvc.var(dim=1) + 1e-10),
                dvc.mean(dim=1),
                torch.log(dqc.var(dim=1) + 1e-10),
                self._logstat(dqc.mean(dim=1), 1e-5),
                cvs_last - cvs_first,
                cvs_slope * 100.0,
            ], dim=1)
            feats = torch.cat([feats, extra], dim=1)
        if self.late_fade_features:
            # R4-E086 (Codex brainstorm #2): early (cycles 1-40) vs late (41-last) fade slope of
            # end-of-discharge capacity; late/curvature values zeroed when fewer than 50 cycles.
            def masked_slope(wm):
                nn_ = wm.sum(1).clamp_min(1.0)
                tm_ = (wm * t).sum(1) / nn_
                cm_ = (wm * cap).sum(1) / nn_
                vt_ = (wm * (t - tm_.unsqueeze(1)) ** 2).sum(1).clamp_min(1e-6)
                return (wm * (t - tm_.unsqueeze(1)) * (cap - cm_.unsqueeze(1))).sum(1) / vt_
            s_early = masked_slope(w * (t < 40).to(w.dtype))
            s_late = masked_slope(w * (t >= 40).to(w.dtype))
            late_valid = (count >= 50).to(feats.dtype)
            curv = s_late - s_early
            late = torch.stack([
                self._logstat(s_early, 1e-7), torch.sign(s_early),
                self._logstat(s_late, 1e-7) * late_valid, torch.sign(s_late) * late_valid,
                self._logstat(curv, 1e-7) * late_valid, torch.sign(curv) * late_valid,
                late_valid,
            ], dim=1)
            feats = torch.cat([feats, late], dim=1)
        return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("DeltaQFingerprintAnchorTCN requires a cycle mask")
        feats = self._fingerprint(curves, mask)
        z = ((feats - self.fp_mean) / self.fp_std).clamp(-6.0, 6.0)
        self._fp = self.fp_mlp(z.to(curves.dtype))
        trunk_curves = curves[:, :, :3]
        if self._dual_q is not None:
            trunk_curves = torch.cat([trunk_curves, self._dual_q.to(trunk_curves.dtype)], dim=2)
        out = super().forward(trunk_curves, mask)
        if self.unbounded_head:
            out = out * torch.exp(self.head_log_gain) + self.fp_direct(z.to(out.dtype)).squeeze(-1)
        return out

    @torch.no_grad()
    def set_reference_bank(self, dataset, max_items=4096):
        gen = torch.Generator().manual_seed(0)  # private RNG: global seed/loader order untouched
        n = len(dataset)
        order = torch.randperm(n, generator=gen)[: min(n, max_items)].tolist()
        feats = []
        for start in range(0, len(order), 256):
            items = [dataset[i] for i in order[start : start + 256]]
            curves = torch.stack([torch.as_tensor(it["cycle_curve_data"]) for it in items]).float()
            mask = torch.stack([torch.as_tensor(it["curve_attn_mask"]) for it in items])
            feats.append(self._fingerprint(curves, mask))
        feats = torch.cat(feats)
        self.fp_mean.copy_(feats.mean(0).to(self.fp_mean.device))
        self.fp_std.copy_(feats.std(0).clamp_min(1e-3).to(self.fp_std.device))

    def _condition_summary(self, summary):
        if self._fp is None:
            return summary
        return summary + self.fp_projection(self._fp)

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["fp_mlp"] = self.fp_mlp
        modules["fp_projection"] = self.fp_projection
        if self.unbounded_head:
            modules["fp_direct"] = self.fp_direct
        return modules


class MetaFingerprintAnchorTCN(DeltaQFingerprintAnchorTCN):
    """R4-E073: E072 fingerprint stream + A7 test-condition metadata (as MetaConditionedAnchorTCN).
    fp_projection starts at N(0, 1e-3) instead of exactly zero (E072 flagged the zero-init layer
    EXPLODING); meta_projection is zero-initialised as in E062."""

    META_DIM = 4

    def __init__(self, *args, fp_init_std=1e-3, meta_dropout=0.0, meta_dropout_eval_scale=False,
                 dual_capacity_view=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_dropout = float(meta_dropout)
        self.meta_dropout_eval_scale = bool(meta_dropout_eval_scale)
        self.dual_capacity_view = bool(dual_capacity_view)
        self._meta_salt = 0
        if self.dual_capacity_view:
            # R4-E085 (Codex brainstorm #1): phase encoders take [V, I, Q_soc, Q_nominal, position];
            # weights from Q_nominal start at zero, so the model starts as R4-E079. The replacement
            # convs are fully overwritten, so the global RNG is restored afterwards (Codex C1): the
            # retained blocks then get the same random initialisation as R4-E079.
            _rng_state = torch.get_rng_state()
            for enc in (self.charge_encoder, self.discharge_encoder):
                old_conv = enc[0]
                new_conv = nn.Conv1d(5, old_conv.out_channels, old_conv.kernel_size[0],
                                     stride=old_conv.stride[0], padding=old_conv.padding[0])
                with torch.no_grad():
                    new_conv.weight.zero_()
                    new_conv.weight[:, :3] = old_conv.weight[:, :3]
                    new_conv.weight[:, 4] = old_conv.weight[:, 3]
                    new_conv.bias.copy_(old_conv.bias)
                enc[0] = new_conv
            torch.set_rng_state(_rng_state)
        with torch.no_grad():
            self.fp_projection.weight.normal_(0.0, fp_init_std)
        self.meta_projection = nn.Linear(self.META_DIM, self.inter_summary.in_features)
        with torch.no_grad():
            self.meta_projection.weight.zero_(); self.meta_projection.bias.zero_()
        self._meta = None

    def train(self, mode=True):
        if mode:
            self._meta_salt += 1  # new meta-dropout pattern each epoch
        return super().train(mode)

    def _meta_keep(self, curves):
        """Per-sample keep mask (training only). Keyed on the Cycle-1 curve and the epoch salt, so
        the full and truncated branches of one sample (prefix-drift) share the same mask."""
        key = (curves[:, 0, :3].float().abs().sum(dim=(1, 2)) * 9973.0 + 7.31 * self._meta_salt)
        u = torch.frac(torch.sin(key) * 43758.5453).abs()
        return (u >= self.meta_dropout).to(curves.dtype).unsqueeze(-1)

    def forward(self, curves, mask):
        self._meta = curves[:, 0, 3, : self.META_DIM] if curves.shape[2] == 4 else None
        self._dual_q = None
        if self.dual_capacity_view:
            if curves.shape[2] != 4:
                raise ValueError("dual_capacity_view needs the v3 metadata channel (log SOC width)")
            width = torch.exp(curves[:, 0, 3, 0]).view(-1, 1, 1, 1)  # unaffected by meta dropout
            self._dual_q = curves[:, :, 2:3] * width                 # nominal-capacity view (v1)
        if self._meta is not None and self.meta_dropout > 0:
            if self.training:
                self._meta = self._meta * self._meta_keep(curves)
            elif self.meta_dropout_eval_scale:  # R4-E084: non-inverted dropout -> expectation at eval
                self._meta = self._meta * (1.0 - self.meta_dropout)
        return super().forward(curves, mask)

    def _condition_summary(self, summary):
        summary = super()._condition_summary(summary)
        if self._meta is None:
            return summary
        return summary + self.meta_projection(self._meta.to(summary.dtype))

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["meta_projection"] = self.meta_projection
        return modules


class DualPhaseCoordConvDetachedAnchorTCN(DualPhaseCoordConvAnchorTCN):
    """Anchor TCN with the shared Cycle-1 subtraction gradient isolated."""

    def _anchor_relative(self, values, weights):
        first_relative = torch.zeros_like(values[:, :1])
        later_relative = (
            values[:, 1:] - values[:, :1].detach()
        ) * weights[:, 1:]
        return torch.cat([first_relative, later_relative], dim=1)


class DualPhaseCoordConvLocalDeltaTCN(DualPhaseCoordConvAnchorTCN):
    """Dual-phase TCN with adjacent-cycle differences instead of an anchor."""

    def _anchor_relative(self, values, weights):
        first_delta = torch.zeros_like(values[:, :1])
        later_delta = (values[:, 1:] - values[:, :-1]) * weights[:, 1:]
        return torch.cat([first_delta, later_delta], dim=1)

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["local_delta_projection"] = modules.pop("anchor_projection")
        return modules


class DualPhaseCoordConvSmoothedDeltaTCN(DualPhaseCoordConvLocalDeltaTCN):
    """Local-delta TCN using a fixed trailing-state mean as its reference."""

    def __init__(self, *args, delta_window=3, **kwargs):
        super().__init__(*args, **kwargs)
        if delta_window < 1:
            raise ValueError("delta_window must be positive")
        self.delta_window = int(delta_window)

    def _anchor_relative(self, values, weights):
        steps = values.shape[1]
        prefix = torch.cat(
            [torch.zeros_like(values[:, :1]), values.cumsum(dim=1)], dim=1
        )
        end = torch.arange(steps, device=values.device)
        start = (end - self.delta_window).clamp_min(0)
        trailing_sum = prefix[:, end] - prefix[:, start]
        counts = (end - start).clamp_min(1).to(values.dtype).view(1, steps, 1)
        trailing_mean = trailing_sum / counts
        later_delta = (values[:, 1:] - trailing_mean[:, 1:]) * weights[:, 1:]
        return torch.cat([torch.zeros_like(values[:, :1]), later_delta], dim=1)

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["smoothed_delta_projection"] = modules.pop("local_delta_projection")
        return modules


class DualPhaseResidualCoordConvCycleAttention(
    DualPhaseResidualCoordConvTemporalTCN
):
    """Dual-phase residual curve encoder with masked additive cycle pooling."""

    def __init__(self, *args, attention_hidden=32, **kwargs):
        super().__init__(*args, **kwargs)
        del self.temporal_tcn
        del self.inter_summary
        self.cycle_attention = nn.Sequential(
            nn.Linear(self.d_model, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        self.capture_attention = False
        self._attention_diagnostics = {}

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError(
                "DualPhaseResidualCoordConvCycleAttention requires a cycle mask"
            )
        if curves.shape[-1] != 2 * self.phase_split:
            raise ValueError("Dual-phase curve input must contain exactly 300 points")
        batch, cycles = curves.shape[:2]
        flat_curves = curves.reshape(batch * cycles, 3, curves.shape[-1])
        charge = flat_curves[:, :, : self.phase_split]
        discharge = flat_curves[:, :, self.phase_split :]
        charge_state = self._encode_phase(charge, self.charge_encoder)
        discharge_state = self._encode_phase(discharge, self.discharge_encoder)
        values = torch.cat([charge_state, discharge_state], dim=-1)
        values = values.reshape(batch, cycles, self.d_model)

        observed = mask.bool()
        scores = self.cycle_attention(values).squeeze(-1)
        scores = scores.masked_fill(~observed, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        summary = torch.sum(attention.unsqueeze(-1) * values, dim=1)
        values = summary
        for block in self.inter_mlp:
            values = block(values)

        if self.capture_attention:
            padding_mass = (attention * (~observed).to(attention.dtype)).sum(dim=1)
            normalized_entropy = -(
                attention.clamp_min(1e-12).log() * attention
            ).sum(dim=1) / observed.sum(dim=1).clamp_min(2).to(attention.dtype).log()
            last_index = observed.sum(dim=1).long().clamp_min(1) - 1
            last_weight = attention.gather(1, last_index[:, None]).squeeze(1)
            self._attention_diagnostics = {
                "max_padding_attention": float(padding_mass.max().detach().cpu()),
                "normalized_entropy_mean": float(
                    normalized_entropy.mean().detach().cpu()
                ),
                "normalized_entropy_min": float(
                    normalized_entropy.min().detach().cpu()
                ),
                "maximum_weight_mean": float(
                    attention.max(dim=1).values.mean().detach().cpu()
                ),
                "first_cycle_weight_mean": float(
                    attention[:, 0].mean().detach().cpu()
                ),
                "last_observed_weight_mean": float(
                    last_weight.mean().detach().cpu()
                ),
            }
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "charge_encoder": self.charge_encoder,
            "discharge_encoder": self.discharge_encoder,
            "cycle_attention": self.cycle_attention,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }

    def attention_diagnostics(self):
        return dict(self._attention_diagnostics)


class ResidualConvAnchorTemporalTCN(ResidualConvTemporalTCNMaskedSummary):
    """R4-E006 with one Cycle-1 relative comparator before the retained TCN."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        temporal_dilations=(1, 2, 4),
        d_ff=256,
        mlp_layers=7,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            temporal_dilations=temporal_dilations,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        self.anchor_projection = nn.Linear(2 * d_model, d_model)
        with torch.no_grad():
            self.anchor_projection.weight.zero_()
            self.anchor_projection.weight[:, :d_model].copy_(torch.eye(d_model))
            self.anchor_projection.bias.zero_()

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ResidualConvAnchorTemporalTCN requires a cycle mask")
        observed = mask.bool()
        if not bool(observed[:, 0].all()):
            raise ValueError("Cycle-1 anchor must be observed for every sample")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

        weights = observed.unsqueeze(-1).to(values.dtype)
        values = values * weights
        anchor = values[:, :1]
        relative = (values - anchor) * weights
        values = self.anchor_projection(torch.cat([values, relative], dim=-1))
        values = values * weights

        temporal_mask = weights.transpose(1, 2)
        values = values.transpose(1, 2)
        for block in self.temporal_tcn:
            values = block(values, temporal_mask)
        values = values.transpose(1, 2) * weights

        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = values.sum(dim=1) / counts
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        values = self.inter_summary(summary)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "anchor_projection": self.anchor_projection,
            "temporal_tcn_first": self.temporal_tcn[0],
            "temporal_tcn_last": self.temporal_tcn[-1],
            "inter_summary": self.inter_summary,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }


class ResidualConvDualBranchTemporalTCN(ResidualConvTemporalTCNMaskedSummary):
    """R4-E006 dynamic path plus a Cycle-1 static branch fused after summary."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        temporal_dilations=(1, 2, 4),
        d_ff=256,
        mlp_layers=7,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            temporal_dilations=temporal_dilations,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        self.late_fusion = nn.Linear(2 * d_model, d_model)
        with torch.no_grad():
            self.late_fusion.weight.zero_()
            self.late_fusion.weight[:, :d_model].copy_(torch.eye(d_model))
            self.late_fusion.bias.zero_()

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ResidualConvDualBranchTemporalTCN requires a cycle mask")
        observed = mask.bool()
        if not bool(observed[:, 0].all()):
            raise ValueError("Cycle-1 static branch must be observed for every sample")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

        weights = observed.unsqueeze(-1).to(values.dtype)
        values = values * weights
        static_state = values[:, 0]

        temporal_mask = weights.transpose(1, 2)
        dynamic_values = values.transpose(1, 2)
        for block in self.temporal_tcn:
            dynamic_values = block(dynamic_values, temporal_mask)
        dynamic_values = dynamic_values.transpose(1, 2) * weights

        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = dynamic_values.sum(dim=1) / counts
        first = dynamic_values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = dynamic_values.gather(
            1,
            last_index[:, None, None].expand(-1, 1, dynamic_values.shape[-1]),
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        dynamic_state = self.inter_summary(summary)
        values = self.late_fusion(torch.cat([dynamic_state, static_state], dim=-1))
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "temporal_tcn_first": self.temporal_tcn[0],
            "temporal_tcn_last": self.temporal_tcn[-1],
            "inter_summary": self.inter_summary,
            "late_fusion": self.late_fusion,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }


class ResidualConvTemporalTCNCapacityLifecycle(
    ResidualConvTemporalTCNMaskedSummary
):
    """R4-E006 plus an explicit input-derived capacity lifecycle path."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        temporal_dilations=(1, 2, 4),
        d_ff=256,
        mlp_layers=7,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            temporal_dilations=temporal_dilations,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        self.lifecycle_projection = nn.Sequential(
            nn.Linear(7, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._lifecycle_diagnostics = {}

    def _lifecycle_features(self, curves, observed, lengths, last_index):
        capacity = curves[:, :, 2, :].amax(dim=-1)
        q_first = capacity[:, 0].clamp_min(0.01)
        q_last = capacity.gather(1, last_index[:, None]).squeeze(1)
        ratio = (q_last / q_first).clamp(0.0, 2.0)
        fade = ((q_first - q_last) / q_first).clamp(-1.0, 1.0)

        positions = torch.arange(
            capacity.shape[1], device=capacity.device, dtype=capacity.dtype
        )[None, :]
        weights = observed.to(capacity.dtype)
        counts = weights.sum(dim=1).clamp_min(1.0)
        x_mean = (positions * weights).sum(dim=1) / counts
        y_mean = (capacity * weights).sum(dim=1) / counts
        centered_x = (positions - x_mean[:, None]) * weights
        slope = (
            centered_x * (capacity - y_mean[:, None]) * weights
        ).sum(dim=1) / centered_x.square().sum(dim=1).clamp_min(1.0)
        slope_100 = (100.0 * slope / q_first).clamp(-2.0, 2.0)

        indices = torch.arange(capacity.shape[1], device=capacity.device)[None, :]
        window = lengths.clamp_max(10)
        early_mask = indices < window[:, None]
        recent_start = lengths - window
        recent_mask = (indices >= recent_start[:, None]) & (
            indices < lengths[:, None]
        )
        early_mean = (
            capacity * early_mask.to(capacity.dtype)
        ).sum(dim=1) / window.to(capacity.dtype)
        recent_mean = (
            capacity * recent_mask.to(capacity.dtype)
        ).sum(dim=1) / window.to(capacity.dtype)
        window_fade = (
            (early_mean - recent_mean) / early_mean.clamp_min(0.01)
        ).clamp(-1.0, 1.0)
        prefix_fraction = lengths.to(capacity.dtype) / capacity.shape[1]
        return torch.stack(
            [
                q_first.clamp(0.0, 2.0),
                q_last.clamp(0.0, 2.0),
                ratio,
                fade,
                slope_100,
                window_fade,
                prefix_fraction,
            ],
            dim=-1,
        )

    def _lifecycle_scale(self, lifecycle_features, lengths):
        return torch.ones(
            lifecycle_features.shape[0],
            1,
            device=lifecycle_features.device,
            dtype=lifecycle_features.dtype,
        )

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError(
                "ResidualConvTemporalTCNCapacityLifecycle requires a cycle mask"
            )
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)
        temporal_mask = weights.transpose(1, 2)
        values = (values * weights).transpose(1, 2)
        for block in self.temporal_tcn:
            values = block(values, temporal_mask)
        values = values.transpose(1, 2) * weights

        lengths = observed.sum(dim=1).long().clamp_min(1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = values.sum(dim=1) / counts
        first = values[:, 0]
        last_index = lengths - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        learned_summary = self.inter_summary(summary)
        lifecycle_features = self._lifecycle_features(
            curves, observed, lengths, last_index
        )
        lifecycle_scale = self._lifecycle_scale(lifecycle_features, lengths)
        lifecycle_state = self.lifecycle_projection(lifecycle_features) * lifecycle_scale
        with torch.no_grad():
            self._lifecycle_diagnostics = {
                "feature_mean": lifecycle_features.detach().mean(dim=0).cpu().tolist(),
                "feature_std": lifecycle_features.detach()
                .std(dim=0, unbiased=False)
                .cpu()
                .tolist(),
                "projection_norm_mean": float(
                    lifecycle_state.detach().norm(dim=-1).mean().cpu()
                ),
                "learned_summary_norm_mean": float(
                    learned_summary.detach().norm(dim=-1).mean().cpu()
                ),
                "projection_to_summary_norm_ratio": float(
                    lifecycle_state.detach().norm(dim=-1).mean().cpu()
                    / learned_summary.detach().norm(dim=-1).mean().clamp_min(1e-12).cpu()
                ),
                "reliability_scale_mean": float(
                    lifecycle_scale.detach().mean().cpu()
                ),
                "reliability_scale_min": float(
                    lifecycle_scale.detach().min().cpu()
                ),
                "reliability_scale_max": float(
                    lifecycle_scale.detach().max().cpu()
                ),
            }
        values = learned_summary + lifecycle_state
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def lifecycle_diagnostics(self):
        return dict(self._lifecycle_diagnostics)

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["lifecycle_projection_first"] = self.lifecycle_projection[0]
        modules["lifecycle_projection_last"] = self.lifecycle_projection[-1]
        return modules


class ResidualConvTemporalTCNReliabilityGatedLifecycle(
    ResidualConvTemporalTCNCapacityLifecycle
):
    """E012 lifecycle residual scaled by observed-history reliability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reliability_gate = nn.Linear(7, 1)
        nn.init.zeros_(self.reliability_gate.weight)
        nn.init.zeros_(self.reliability_gate.bias)

    def _lifecycle_scale(self, lifecycle_features, lengths):
        history_reliability = ((lengths.to(lifecycle_features.dtype) - 1.0) / 99.0)
        history_reliability = history_reliability.clamp(0.0, 1.0).unsqueeze(-1)
        learned_correction = 2.0 * torch.sigmoid(
            self.reliability_gate(lifecycle_features)
        )
        return history_reliability * learned_correction

    def diagnostic_modules(self):
        modules = super().diagnostic_modules()
        modules["reliability_gate"] = self.reliability_gate
        return modules


class ResidualConvTemporalTCNAdaptiveMixture(
    ResidualConvTemporalTCNMaskedSummary
):
    """R4-E006 with a matched-depth, input-routed two-expert predictor tail."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        temporal_dilations=(1, 2, 4),
        d_ff=256,
        mlp_layers=7,
        shared_layers=5,
        num_experts=2,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        if num_experts != 2:
            raise ValueError("The controlled E011 ablation requires two experts")
        if mlp_layers - shared_layers != num_experts:
            raise ValueError(
                "Shared plus one block per expert must match the parent depth"
            )
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            temporal_dilations=temporal_dilations,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        self.shared_predictor = nn.ModuleList(
            list(self.inter_mlp[:shared_layers])
        )
        self.expert_predictors = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(num_experts)]
        )
        self.mixture_gate = nn.Linear(d_model, num_experts)
        self.expert_heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(num_experts)]
        )
        del self.inter_mlp
        del self.head
        self._gate_diagnostics = {}

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError(
                "ResidualConvTemporalTCNAdaptiveMixture requires a cycle mask"
            )
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)
        temporal_mask = weights.transpose(1, 2)
        values = (values * weights).transpose(1, 2)
        for block in self.temporal_tcn:
            values = block(values, temporal_mask)
        values = values.transpose(1, 2) * weights

        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = values.sum(dim=1) / counts
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        values = self.inter_summary(summary)
        for block in self.shared_predictor:
            values = block(values)

        gate = torch.softmax(self.mixture_gate(values), dim=-1)
        expert_outputs = []
        for expert, head in zip(self.expert_predictors, self.expert_heads):
            expert_values = expert(values)
            expert_outputs.append(head(F.relu(expert_values)).squeeze(-1))
        predictions = torch.stack(expert_outputs, dim=-1)
        with torch.no_grad():
            detached_gate = gate.detach()
            detached_predictions = predictions.detach()
            entropy = -(
                detached_gate * detached_gate.clamp_min(1e-8).log()
            ).sum(dim=-1)
            self._gate_diagnostics = {
                "expert_0_mean": float(detached_gate[:, 0].mean().item()),
                "expert_0_std": float(
                    detached_gate[:, 0].std(unbiased=False).item()
                ),
                "expert_0_min": float(detached_gate[:, 0].min().item()),
                "expert_0_max": float(detached_gate[:, 0].max().item()),
                "expert_1_mean": float(detached_gate[:, 1].mean().item()),
                "maximum_assignment_mean": float(
                    detached_gate.max(dim=-1).values.mean().item()
                ),
                "normalized_entropy_mean": float(
                    (entropy / math.log(detached_gate.shape[-1])).mean().item()
                ),
                "expert_prediction_absolute_difference_mean": float(
                    (detached_predictions[:, 0] - detached_predictions[:, 1])
                    .abs()
                    .mean()
                    .item()
                ),
            }
        return (predictions * gate).sum(dim=-1)

    def gate_diagnostics(self):
        return dict(self._gate_diagnostics)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "temporal_tcn_first": self.temporal_tcn[0],
            "temporal_tcn_last": self.temporal_tcn[-1],
            "inter_summary": self.inter_summary,
            "shared_predictor": self.shared_predictor[-1],
            "expert_0": self.expert_predictors[0],
            "expert_1": self.expert_predictors[1],
            "mixture_gate": self.mixture_gate,
            "expert_head_0": self.expert_heads[0],
            "expert_head_1": self.expert_heads[1],
        }


class MultiScaleResidualConvTemporalTCNMaskedSummary(
    ResidualConvTemporalTCNMaskedSummary
):
    """R4-E006 with one parallel multi-scale within-cycle residual block."""

    def __init__(
        self,
        d_model=32,
        temporal_dilations=(1, 2, 4),
        d_ff=256,
        mlp_layers=7,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=0,
            temporal_dilations=temporal_dilations,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        self.residual_blocks = nn.Sequential(
            MultiScaleResidualCurveBlock(d_model)
        )

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "multiscale_curve_block": self.residual_blocks[0],
            "temporal_tcn_first": self.temporal_tcn[0],
            "temporal_tcn_last": self.temporal_tcn[-1],
            "inter_summary": self.inter_summary,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }


class ResidualConvTemporalPyramidTCN(ResidualConvTemporalTCNMaskedSummary):
    """R4-E006 with mask-safe temporal-pyramid summary pooling."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        temporal_dilations=(1, 2, 4),
        pyramid_levels=(1, 2, 4),
        d_ff=256,
        mlp_layers=7,
        dropout=0.0,
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            temporal_dilations=temporal_dilations,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode="mean_last_delta",
            dropout=dropout,
        )
        self.temporal_pyramid = MaskedTemporalPyramidPool(pyramid_levels)
        self.inter_summary = nn.Linear(
            self.temporal_pyramid.output_bins * d_model,
            d_model,
        )

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ResidualConvTemporalPyramidTCN requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)
        temporal_mask = weights.transpose(1, 2)
        values = (values * weights).transpose(1, 2)
        for block in self.temporal_tcn:
            values = block(values, temporal_mask)
        values = values.transpose(1, 2) * weights

        values = self.inter_summary(self.temporal_pyramid(values, observed))
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "temporal_tcn_first": self.temporal_tcn[0],
            "temporal_tcn_last": self.temporal_tcn[-1],
            "temporal_pyramid": self.temporal_pyramid,
            "pyramid_projection": self.inter_summary,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }


class ResidualConvCycleAttentionMaskedSummary(ResidualConvMaskedSummary):
    """R4-E005 curve CNN with one masked learned cycle-attention block."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        n_heads=4,
        attention_d_ff=128,
        d_ff=256,
        mlp_layers=7,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        self.position = nn.Parameter(torch.zeros(1, 100, d_model))
        self.attention_norm1 = nn.LayerNorm(d_model)
        self.cycle_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm2 = nn.LayerNorm(d_model)
        self.attention_ffn = nn.Sequential(
            nn.Linear(d_model, attention_d_ff),
            nn.GELU(),
            nn.Linear(attention_d_ff, d_model),
        )
        self.capture_attention = False
        self._attention_diagnostics = {}
        nn.init.normal_(self.position, std=0.02)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError(
                "ResidualConvCycleAttentionMaskedSummary requires a cycle mask"
            )
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)
        values = (values + self.position[:, :cycles]) * weights
        normalized = self.attention_norm1(values)
        attended, attention = self.cycle_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~observed,
            need_weights=self.capture_attention,
            average_attn_weights=False,
        )
        values = (values + attended) * weights
        values = (
            values + self.attention_ffn(self.attention_norm2(values))
        ) * weights

        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = values.sum(dim=1) / counts
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        values = self.inter_summary(summary)
        for block in self.inter_mlp:
            values = block(values)

        if self.capture_attention:
            last_attention = attention.gather(
                2,
                last_index[:, None, None, None].expand(
                    -1, attention.shape[1], 1, attention.shape[-1]
                ),
            ).squeeze(2)
            padding_mass = (
                last_attention * (~observed)[:, None, :].to(last_attention.dtype)
            ).sum(dim=-1)
            observed_attention = (
                last_attention * observed[:, None, :].to(last_attention.dtype)
            )
            observed_attention = observed_attention / observed_attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            entropy = -(
                observed_attention.clamp_min(1e-12).log() * observed_attention
            ).sum(dim=-1)
            normalizer = (
                observed.sum(dim=1).clamp_min(2).to(entropy.dtype).log()[:, None]
            )
            self._attention_diagnostics = {
                "max_padding_attention": float(padding_mass.max().detach().cpu()),
                "mean_normalized_last_entropy": float(
                    (entropy / normalizer).mean().detach().cpu()
                ),
            }

        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "cycle_attention": self.cycle_attention,
            "attention_ffn": self.attention_ffn,
            "inter_summary": self.inter_summary,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }

    def attention_diagnostics(self):
        return dict(self._attention_diagnostics)


class ResidualConvGRUMaskedSummary(ResidualConvMaskedSummary):
    """R4-E005 curve CNN with one gated recurrent cycle-state layer."""

    def __init__(
        self,
        d_model=32,
        residual_layers=1,
        d_ff=256,
        mlp_layers=7,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__(
            d_model=d_model,
            residual_layers=residual_layers,
            d_ff=d_ff,
            mlp_layers=mlp_layers,
            summary_mode=summary_mode,
            dropout=dropout,
        )
        self.temporal_gru = nn.GRU(
            d_model, d_model, num_layers=1, batch_first=True
        )

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("ResidualConvGRUMaskedSummary requires a cycle mask")
        batch, cycles = curves.shape[:2]
        values = self.cycle_stem(curves.reshape(batch * cycles, 3, 300))
        values = self.residual_blocks(values)
        values = self.curve_pool(values).squeeze(-1).reshape(batch, cycles, -1)

        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)
        values, _ = self.temporal_gru(values * weights)
        values = values * weights

        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = values.sum(dim=1) / counts
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        values = self.inter_summary(summary)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "cycle_stem": self.cycle_stem,
            "residual_blocks": self.residual_blocks,
            "temporal_gru": self.temporal_gru,
            "inter_summary": self.inter_summary,
            "inter_mlp": self.inter_mlp[-1],
            "head": self.head,
        }


class MultiScaleConvMaskedSummary(nn.Module):
    """Multi-Scale Grouped CNN curve encoder fused with raw embedding + Masked Summary Pooling."""

    def __init__(
        self,
        d_model=64,
        d_ff=256,
        trunk_layers=4,
        summary_mode="mean_last_delta",
        head_mode="linear",
        feature_mode="raw",
        enable_capacity_ratio=False,
        enable_cycle_conv=False,
        conv_groups=3,
        context_layers=2,
        n_heads=4,
        enable_aux_head=False,
        dropout=0.0,
    ):
        super().__init__()
        self.summary_mode = summary_mode
        self.head_mode = head_mode
        self.feature_mode = feature_mode
        self.enable_capacity_ratio = enable_capacity_ratio
        self.enable_cycle_conv = enable_cycle_conv
        self.enable_aux_head = enable_aux_head
        self.conv_groups = conv_groups
        if enable_capacity_ratio:
            self.deg_proj = nn.Sequential(
                nn.Linear(3, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        if enable_cycle_conv:
            self.cycle_conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.GELU(),
            )
        self.conv_fine = nn.Sequential(
            nn.Conv1d(3, 24, kernel_size=3, stride=2, padding=1, groups=conv_groups),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.conv_med = nn.Sequential(
            nn.Conv1d(3, 24, kernel_size=7, stride=3, padding=3, groups=conv_groups),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.conv_coarse = nn.Sequential(
            nn.Conv1d(3, 24, kernel_size=15, stride=5, padding=7, groups=conv_groups),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.raw_embed = nn.Linear(900, d_model)
        fusion_width = 240 * 3 + d_model
        if feature_mode == "raw_plus_reference_delta":
            self.relative_embed = nn.Linear(900, d_model)
            fusion_width += d_model
        elif feature_mode != "raw":
            raise ValueError(f"Unknown feature mode: {feature_mode}")
        self.fusion = nn.Sequential(
            nn.Linear(fusion_width, d_model),
            nn.GELU(),
        )
        if summary_mode == "mean":
            summary_width = d_model
        elif summary_mode == "hybrid_sharp_window":
            summary_width = 6 * d_model
        elif summary_mode in (
            "multi_resolution",
            "dual_sharp_curvature",
            "dual_sharp_context_residual",
        ):
            summary_width = 5 * d_model
        elif summary_mode == "gru":
            summary_width = d_model
            self.temporal_gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        else:
            summary_width = 3 * d_model
        if summary_mode == "gru":
            self.inter_summary = nn.Identity()
        else:
            self.inter_summary = nn.Linear(summary_width, d_model)
        self.inter_trunk = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(trunk_layers)]
        )
        if head_mode == "linear":
            self.head = nn.Linear(d_model, 1)
        elif head_mode == "scale_mixture":
            self.short_scale_head = nn.Linear(d_model, 1)
            self.long_scale_head = nn.Linear(d_model, 1)
            self.scale_selector = nn.Linear(d_model, 1)
            self._scale_diagnostics = {}
        elif head_mode == "additive_scale_residual":
            self.head = nn.Linear(d_model, 1)
            self.scale_residual = nn.Linear(d_model, 1)
            nn.init.zeros_(self.scale_residual.weight)
            nn.init.zeros_(self.scale_residual.bias)
            self._scale_diagnostics = {}
        else:
            raise ValueError(f"Unknown head mode: {head_mode}")
        if enable_aux_head:
            # Initialized after the complete E007 prediction path so the same
            # seed preserves every shared parameter and initial EOL output.
            self.aux_degradation_head = nn.Linear(d_model, 1)
        if summary_mode == "dual_sharp_context_residual":
            # These modules are initialized after the unchanged E012 path so
            # matching seeds retain identical static-path parameters.
            self.context_pos_embed = nn.Parameter(torch.zeros(1, 100, d_model))
            nn.init.normal_(self.context_pos_embed, std=0.02)
            self.context_transformer = nn.ModuleList(
                [
                    TransformerCycleBlock(d_model, n_heads, d_ff, dropout)
                    for _ in range(context_layers)
                ]
            )
            self.context_projection = nn.Linear(3 * d_model, d_model)
            # Start from the exact E012 static representation. The contextual
            # branch must earn a non-zero residual from training evidence.
            nn.init.zeros_(self.context_projection.weight)
            nn.init.zeros_(self.context_projection.bias)

    def forward(self, curves, mask, return_aux=False):
        if mask is None:
            raise ValueError("MultiScaleConvMaskedSummary requires a cycle mask")
        batch, cycles = curves.shape[:2]
        observed = mask.bool()
        flat_curves = curves.reshape(batch * cycles, 3, 300)
        f_fine = self.conv_fine(flat_curves).reshape(batch * cycles, 240)
        f_med = self.conv_med(flat_curves).reshape(batch * cycles, 240)
        f_coarse = self.conv_coarse(flat_curves).reshape(batch * cycles, 240)
        conv_feat = torch.cat([f_fine, f_med, f_coarse], dim=-1)
        raw_feat = self.raw_embed(flat_curves.reshape(batch * cycles, 900))
        fusion_parts = [conv_feat, raw_feat]
        if self.feature_mode == "raw_plus_reference_delta":
            relative_curves = curves - curves[:, :1]
            relative_feat = self.relative_embed(
                relative_curves.reshape(batch * cycles, 900)
            )
            fusion_parts.append(relative_feat)
        fused = self.fusion(torch.cat(fusion_parts, dim=-1)).reshape(batch, cycles, -1)
        if self.enable_cycle_conv:
            weights_t = observed.unsqueeze(1).to(fused.dtype)
            conv_in = fused.transpose(1, 2) * weights_t
            conv_out = self.cycle_conv(conv_in) * weights_t
            fused = (fused + conv_out.transpose(1, 2)) * observed.unsqueeze(-1).to(fused.dtype)

        weights = observed.unsqueeze(-1).to(fused.dtype)
        lengths = observed.sum(dim=1).long().clamp_min(1)
        mean = (fused * weights).sum(dim=1) / lengths.unsqueeze(-1).to(fused.dtype)
        first = fused[:, 0]
        last_index = lengths - 1
        last = fused.gather(1, last_index[:, None, None].expand(-1, 1, fused.shape[-1])).squeeze(1)

        if self.summary_mode == "gru":
            masked_fused = fused * observed.unsqueeze(-1).to(fused.dtype)
            gru_out, _ = self.temporal_gru(masked_fused)
            summary = gru_out.gather(1, last_index[:, None, None].expand(-1, 1, gru_out.shape[-1])).squeeze(1)
        elif self.summary_mode in (
            "multi_resolution",
            "dual_sharp_curvature",
            "dual_sharp_context_residual",
            "hybrid_sharp_window",
        ):
            early_len = lengths.clamp_max(10)
            idx = torch.arange(cycles, device=fused.device).unsqueeze(0).expand(batch, -1)
            early_mask = (idx < early_len.unsqueeze(1)).unsqueeze(-1).to(fused.dtype)
            mean_early = (fused * early_mask).sum(dim=1) / early_len.unsqueeze(-1).to(fused.dtype)
            recent_start = (lengths - 10).clamp_min(0)
            recent_mask = ((idx >= recent_start.unsqueeze(1)) & (idx < lengths.unsqueeze(1))).unsqueeze(-1).to(fused.dtype)
            recent_len = (lengths - recent_start).clamp_min(1)
            mean_recent = (fused * recent_mask).sum(dim=1) / recent_len.unsqueeze(-1).to(fused.dtype)
            delta_velocity = mean_recent - mean_early
            mid_start = early_len
            mid_end = recent_start
            has_mid = (mid_end > mid_start).unsqueeze(-1)
            mid_mask = ((idx >= mid_start.unsqueeze(1)) & (idx < mid_end.unsqueeze(1))).unsqueeze(-1).to(fused.dtype)
            mid_len = (mid_end - mid_start).clamp_min(1).unsqueeze(-1).to(fused.dtype)
            mean_mid = torch.where(has_mid, (fused * mid_mask).sum(dim=1) / mid_len, mean)
            curvature = mean_recent - 2.0 * mean_mid + mean_early
            if self.summary_mode == "hybrid_sharp_window":
                # Synthesizes instantaneous point-sharpness (last, last-first) with smooth window temporal descriptors (mean_recent, delta_velocity, curvature)
                summary = torch.cat([mean, last, last - first, mean_recent, delta_velocity, curvature], dim=-1)
            elif self.summary_mode in (
                "dual_sharp_curvature",
                "dual_sharp_context_residual",
            ):
                # Combines instantaneous last state (c_T) and total change (c_T - c_1) from E007
                # with continuous velocity and acceleration curvature from E011:
                summary = torch.cat([mean, last, last - first, delta_velocity, curvature], dim=-1)
            else:
                summary = torch.cat([mean, mean_early, mean_recent, delta_velocity, curvature], dim=-1)
        elif self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        else:
            summary = mean
        values = self.inter_summary(summary)
        if self.summary_mode == "dual_sharp_context_residual":
            contextual = (fused + self.context_pos_embed[:, :cycles]) * weights
            for block in self.context_transformer:
                contextual = block(contextual, key_padding_mask=~observed) * weights
            context_mean = (contextual * weights).sum(dim=1) / lengths.unsqueeze(-1).to(
                contextual.dtype
            )
            context_first = contextual[:, 0]
            context_last = contextual.gather(
                1,
                last_index[:, None, None].expand(-1, 1, contextual.shape[-1]),
            ).squeeze(1)
            context_summary = torch.cat(
                [context_mean, context_last, context_last - context_first], dim=-1
            )
            values = values + self.context_projection(context_summary)
        if self.enable_capacity_ratio:
            q_all = curves[:, :, 2, :].max(dim=-1).values
            q_1 = q_all[:, 0].clamp_min(0.01)
            q_last = q_all.gather(1, last_index.unsqueeze(1)).squeeze(1)
            ratio = (q_last / q_1).clamp(0.0, 2.0).unsqueeze(-1)
            fade = ((q_1 - q_last) / q_1).clamp(-1.0, 1.0).unsqueeze(-1)
            prefix_len = lengths.unsqueeze(-1).to(fused.dtype)
            fade_rate = (fade / prefix_len).clamp(-0.1, 0.1)
            deg_feats = torch.cat([ratio, fade, fade_rate], dim=-1)
            deg_repr = self.deg_proj(deg_feats)
            values = values + deg_repr
        for block in self.inter_trunk:
            values = block(values)
        if self.enable_capacity_ratio:
            values = values + deg_repr
        values = F.relu(values)
        if self.head_mode == "scale_mixture":
            selector = torch.sigmoid(self.scale_selector(values))
            short_prediction = self.short_scale_head(values)
            long_prediction = self.long_scale_head(values)
            prediction = (1.0 - selector) * short_prediction + selector * long_prediction
            self._scale_diagnostics = {
                "selector_mean": float(selector.detach().mean().cpu()),
                "selector_std": float(selector.detach().std(unbiased=False).cpu()),
                "selector_min": float(selector.detach().min().cpu()),
                "selector_max": float(selector.detach().max().cpu()),
            }
            return prediction.squeeze(-1)
        if self.head_mode == "additive_scale_residual":
            base_prediction = self.head(values)
            scale_correction = self.scale_residual(mean)
            self._scale_diagnostics = {
                "correction_mean": float(scale_correction.detach().mean().cpu()),
                "correction_std": float(scale_correction.detach().std(unbiased=False).cpu()),
                "correction_min": float(scale_correction.detach().min().cpu()),
                "correction_max": float(scale_correction.detach().max().cpu()),
            }
            return (base_prediction + scale_correction).squeeze(-1)
        prediction = self.head(values).squeeze(-1)
        if return_aux:
            if not self.enable_aux_head:
                raise RuntimeError("Auxiliary degradation head is disabled")
            degradation = self.aux_degradation_head(fused).squeeze(-1)
            degradation = degradation * observed.to(degradation.dtype)
            return prediction, degradation
        return prediction

    def gate_diagnostics(self):
        return dict(getattr(self, "_scale_diagnostics", {}))

    def diagnostic_modules(self):
        mods = {
            "conv_fine": self.conv_fine[0],
            "conv_med": self.conv_med[0],
            "conv_coarse": self.conv_coarse[0],
            "raw_embed": self.raw_embed,
            "fusion": self.fusion[0],
            "inter_trunk_first": self.inter_trunk[0],
            "inter_trunk_last": self.inter_trunk[-1],
        }
        if self.feature_mode == "raw_plus_reference_delta":
            mods["relative_embed"] = self.relative_embed
        if self.head_mode == "scale_mixture":
            mods["short_scale_head"] = self.short_scale_head
            mods["long_scale_head"] = self.long_scale_head
            mods["scale_selector"] = self.scale_selector
        elif self.head_mode == "additive_scale_residual":
            mods["head"] = self.head
            mods["scale_residual"] = self.scale_residual
        else:
            mods["head"] = self.head
        if self.enable_aux_head:
            mods["aux_degradation_head"] = self.aux_degradation_head
        if self.summary_mode == "gru":
            mods["temporal_gru"] = self.temporal_gru
        else:
            mods["inter_summary"] = self.inter_summary
        if self.summary_mode == "dual_sharp_context_residual":
            mods["context_transformer_first"] = self.context_transformer[0]
            mods["context_transformer_last"] = self.context_transformer[-1]
            mods["context_projection"] = self.context_projection
        if self.enable_capacity_ratio:
            mods["deg_proj"] = self.deg_proj[0]
        if self.enable_cycle_conv:
            mods["cycle_conv"] = self.cycle_conv[0]
        return mods


class TransformerCycleBlock(nn.Module):

    """Pre-LN Multi-Head Self-Attention + FFN Transformer Block."""

    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values, key_padding_mask):
        normalized = self.norm1(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        values = values + attended
        values = values + self.ffn(self.norm2(values))
        return values


class CyclePatchTransformer(nn.Module):
    """CPMLP cycle-as-token encoder with a mask-safe inter-cycle Transformer."""

    def __init__(
        self,
        d_model=32,
        d_ff=256,
        intra_layers=12,
        inter_mlp_layers=7,
        transformer_layers=2,
        n_heads=4,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__()
        if summary_mode not in {"mean_last_delta", "last"}:
            raise ValueError(f"Unknown CyclePatch summary mode: {summary_mode}")
        self.summary_mode = summary_mode
        self.intra_embed = nn.Linear(300 * 3, d_model)
        self.intra_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(intra_layers)]
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerCycleBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(transformer_layers)
            ]
        )
        summary_width = d_model if summary_mode == "last" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.inter_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(inter_mlp_layers)]
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("CyclePatchTransformer requires a cycle mask")
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(curves.dtype)
        values = self.intra_embed(torch.flatten(curves, start_dim=2))
        for block in self.intra_mlp:
            values = block(values)
        values = (values + self.pos_embed[:, : curves.shape[1]]) * weights
        for block in self.transformer_blocks:
            values = block(values, key_padding_mask=~observed) * weights

        lengths = observed.sum(dim=1).long().clamp_min(1)
        mean = (values * weights).sum(dim=1) / lengths.unsqueeze(-1).to(values.dtype)
        first = values[:, 0]
        last_index = lengths - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = last if self.summary_mode == "last" else torch.cat(
            [mean, last, last - first], dim=-1
        )
        values = self.inter_summary(summary)
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "intra_embed": self.intra_embed,
            "intra_mlp_first": self.intra_mlp[0],
            "intra_mlp_middle": self.intra_mlp[len(self.intra_mlp) // 2],
            "intra_mlp_last": self.intra_mlp[-1],
            "inter_transformer_first": self.transformer_blocks[0].ffn[-2],
            "inter_transformer_last": self.transformer_blocks[-1].ffn[-2],
            "inter_summary": self.inter_summary,
            "inter_mlp_first": self.inter_mlp[0],
            "inter_mlp_last": self.inter_mlp[-1],
            "head": self.head,
        }


class CyclePatchTCN(nn.Module):
    """CPMLP cycle-as-token encoder with a local causal inter-cycle TCN."""

    def __init__(
        self,
        d_model=32,
        d_ff=256,
        intra_layers=12,
        inter_mlp_layers=7,
        temporal_dilations=(1, 2, 4, 8),
        dropout=0.0,
    ):
        super().__init__()
        self.intra_embed = nn.Linear(300 * 3, d_model)
        self.intra_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(intra_layers)]
        )
        self.temporal_tcn = nn.ModuleList(
            [CausalTCNBlock(d_model, dilation) for dilation in temporal_dilations]
        )
        self.inter_summary = nn.Linear(3 * d_model, d_model)
        self.inter_mlp = nn.ModuleList(
            [MLPBlock(d_model, d_ff, d_model, dropout) for _ in range(inter_mlp_layers)]
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("CyclePatchTCN requires a cycle mask")
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(curves.dtype)
        values = self.intra_embed(torch.flatten(curves, start_dim=2))
        for block in self.intra_mlp:
            values = block(values)
        values = values * weights
        for block in self.temporal_tcn:
            values = block(values) * weights

        lengths = observed.sum(dim=1).long().clamp_min(1)
        mean = (values * weights).sum(dim=1) / lengths.unsqueeze(-1).to(values.dtype)
        first = values[:, 0]
        last_index = lengths - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        values = self.inter_summary(torch.cat([mean, last, last - first], dim=-1))
        for block in self.inter_mlp:
            values = block(values)
        return self.head(F.relu(values)).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "intra_embed": self.intra_embed,
            "intra_mlp_first": self.intra_mlp[0],
            "intra_mlp_middle": self.intra_mlp[len(self.intra_mlp) // 2],
            "intra_mlp_last": self.intra_mlp[-1],
            "inter_tcn_first": self.temporal_tcn[0],
            "inter_tcn_last": self.temporal_tcn[-1],
            "inter_summary": self.inter_summary,
            "inter_mlp_first": self.inter_mlp[0],
            "inter_mlp_last": self.inter_mlp[-1],
            "head": self.head,
        }


class CycleTransformerSummary(nn.Module):
    """Transformer Multi-Head Self-Attention over Cycles with Masked Degradation Summary."""

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        summary_mode="mean_last_delta",
        dropout=0.0,
        enable_aux_head=False,
    ):
        super().__init__()
        self.summary_mode = summary_mode
        self.intra_embed = nn.Linear(300 * 3, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerCycleBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        summary_width = d_model if summary_mode == "mean" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.capture_attention = False
        self._attention_diagnostics = {}
        if enable_aux_head:
            self.aux_degradation_head = nn.Linear(d_model, 1)

    def forward(self, curves, mask, return_aux=False):
        if mask is None:
            raise ValueError("CycleTransformerSummary requires a cycle mask")
        values = self.intra_embed(torch.flatten(curves, start_dim=2))
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)

        values = (values + self.pos_embed[:, : curves.shape[1]]) * weights

        for block in self.transformer_blocks:
            values = block(values, key_padding_mask=~observed) * weights

        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        summary = self.inter_summary(summary)
        pred = self.head(summary).squeeze(-1)
        if return_aux:
            if not hasattr(self, "aux_degradation_head"):
                raise RuntimeError("Auxiliary degradation head is disabled")
            aux_deg = self.aux_degradation_head(values).squeeze(-1) * weights.squeeze(-1)
            return pred, aux_deg
        return pred

    def diagnostic_modules(self):
        modules = {
            "intra_embed": self.intra_embed,
            "transformer": self.transformer_blocks[-1].ffn[-2],
            "inter_summary": self.inter_summary,
            "head": self.head[-1],
        }
        if hasattr(self, "aux_degradation_head"):
            modules["aux_head"] = self.aux_degradation_head
        return modules


class CycleTransformerAttentiveSummary(nn.Module):
    """Transformer Encoder + Learned Query Attentive Pooling + Degradation Deltas."""

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.intra_embed = nn.Linear(300 * 3, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, d_model))
        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_token, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerCycleBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.inter_summary = nn.Linear(3 * d_model, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.capture_attention = False
        self._attention_diagnostics = {}

    def forward(self, curves, mask):
        if mask is None:
            raise ValueError("CycleTransformerAttentiveSummary requires a cycle mask")
        values = self.intra_embed(torch.flatten(curves, start_dim=2))
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)

        values = (values + self.pos_embed[:, : curves.shape[1]]) * weights

        for block in self.transformer_blocks:
            values = block(values, key_padding_mask=~observed) * weights

        batch = values.shape[0]
        queries = self.query_token.expand(batch, -1, -1)
        normalized_values = self.pool_norm(values)
        attended_summary, attention = self.pool_attention(
            queries,
            normalized_values,
            normalized_values,
            key_padding_mask=~observed,
            need_weights=self.capture_attention,
            average_attn_weights=False,
        )
        attn_pool = attended_summary.squeeze(1)

        if self.capture_attention:
            query_weights = attention.squeeze(2)
            padding_mass = (
                query_weights * (~observed)[:, None, :].to(query_weights.dtype)
            ).sum(dim=-1)
            observed_weights = query_weights * observed[:, None, :].to(
                query_weights.dtype
            )
            observed_weights = observed_weights / observed_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            entropy = -(
                observed_weights.clamp_min(1e-12).log() * observed_weights
            ).sum(dim=-1)
            normalizer = observed.sum(dim=1).clamp_min(2).to(entropy.dtype).log()[
                :, None
            ]
            self._attention_diagnostics = {
                "max_padding_attention": float(padding_mass.max().detach().cpu()),
                "mean_normalized_query_entropy": float(
                    (entropy / normalizer).mean().detach().cpu()
                ),
            }

        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)

        summary = torch.cat([attn_pool, last, last - first], dim=-1)
        summary = self.inter_summary(summary)
        return self.head(summary).squeeze(-1)

    def diagnostic_modules(self):
        return {
            "intra_embed": self.intra_embed,
            "transformer": self.transformer_blocks[-1].ffn[-2],
            "pool_attention": self.pool_attention,
            "inter_summary": self.inter_summary,
            "head": self.head[-1],
        }

    def attention_diagnostics(self):
        return dict(self._attention_diagnostics)


class ConvCycleTransformerSummary(nn.Module):
    """Fast 1D Conv Stem + 2-Layer Transformer + Masked Summary Pooling."""

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__()
        self.summary_mode = summary_mode
        self.conv_stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(32, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerCycleBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        summary_width = d_model if summary_mode == "mean" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.aux_degradation_head = nn.Linear(d_model, 1)

    def forward(self, curves, mask, return_aux=False):
        if mask is None:
            raise ValueError("ConvCycleTransformerSummary requires a cycle mask")
        batch, cycles = curves.shape[:2]
        flat_curves = curves.reshape(batch * cycles, 3, 300)
        stem_feats = self.conv_stem(flat_curves).squeeze(-1)
        values = stem_feats.reshape(batch, cycles, -1)

        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(values.dtype)

        values = (values + self.pos_embed[:, :cycles]) * weights

        for block in self.transformer_blocks:
            values = block(values, key_padding_mask=~observed) * weights

        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        summary = self.inter_summary(summary)
        pred = self.head(summary).squeeze(-1)
        if return_aux:
            aux_deg = self.aux_degradation_head(values).squeeze(-1) * weights.squeeze(-1)
            return pred, aux_deg
        return pred

    def diagnostic_modules(self):
        return {
            "conv_stem": self.conv_stem[2],
            "transformer": self.transformer_blocks[-1].ffn[-2],
            "inter_summary": self.inter_summary,
            "aux_head": self.aux_degradation_head,
            "head": self.head[-1],
        }


class DualStreamDeltaQTransformer(nn.Module):
    """Dual-Stream Joint Curves + Delta-Q(V) Electrochemical Feature Fusion Transformer."""

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__()
        self.summary_mode = summary_mode
        self.intra_embed = nn.Linear(300 * 3, d_model)
        self.delta_embed = nn.Linear(300, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerCycleBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        summary_width = d_model if summary_mode == "mean" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.aux_degradation_head = nn.Linear(d_model, 1)

    def forward(self, curves, mask, return_aux=False):
        if mask is None:
            raise ValueError("DualStreamDeltaQTransformer requires a cycle mask")
        batch, cycles = curves.shape[:2]
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(curves.dtype)

        raw_embed = self.intra_embed(torch.flatten(curves, start_dim=2))

        q_1 = curves[:, 0:1, 2, :]  # [B, 1, 300]
        delta_q = curves[:, :, 2, :] - q_1  # [B, 100, 300]
        delta_embed = self.delta_embed(delta_q)

        fused = self.fusion(torch.cat([raw_embed, delta_embed], dim=-1))
        values = (fused + self.pos_embed[:, :cycles]) * weights

        for block in self.transformer_blocks:
            values = block(values, key_padding_mask=~observed) * weights

        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        summary = self.inter_summary(summary)
        pred = self.head(summary).squeeze(-1)
        if return_aux:
            aux_deg = self.aux_degradation_head(values).squeeze(-1) * weights.squeeze(-1)
            return pred, aux_deg
        return pred

    def diagnostic_modules(self):
        return {
            "intra_embed": self.intra_embed,
            "delta_embed": self.delta_embed,
            "fusion": self.fusion[0],
            "transformer": self.transformer_blocks[-1].ffn[-2],
            "inter_summary": self.inter_summary,
            "aux_head": self.aux_degradation_head,
            "head": self.head[-1],
        }


class ElectroTransformer(nn.Module):
    """Channel-Specific Feature Extraction + Multi-Head Self-Attention Cycle Transformer."""

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__()
        self.summary_mode = summary_mode
        self.v_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=3, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.i_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=3, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.q_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=3, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.fusion = nn.Sequential(
            nn.Linear(480 + 900, d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerCycleBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        summary_width = d_model if summary_mode == "mean" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.aux_degradation_head = nn.Linear(d_model, 1)

    def forward(self, curves, mask, return_aux=False):
        if mask is None:
            raise ValueError("ElectroTransformer requires a cycle mask")
        batch, cycles = curves.shape[:2]
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(curves.dtype)

        flat_curves = curves.reshape(batch * cycles, 3, 300)
        v_feat = self.v_conv(flat_curves[:, 0:1, :]).reshape(batch * cycles, -1)
        i_feat = self.i_conv(flat_curves[:, 1:2, :]).reshape(batch * cycles, -1)
        q_feat = self.q_conv(flat_curves[:, 2:3, :]).reshape(batch * cycles, -1)
        raw_feat = flat_curves.reshape(batch * cycles, 900)

        fused_tokens = self.fusion(torch.cat([v_feat, i_feat, q_feat, raw_feat], dim=-1))
        values = fused_tokens.reshape(batch, cycles, -1)

        values = (values + self.pos_embed[:, :cycles]) * weights

        for block in self.transformer_blocks:
            values = block(values, key_padding_mask=~observed) * weights

        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        summary = self.inter_summary(summary)
        pred = self.head(summary).squeeze(-1)
        if return_aux:
            aux_deg = self.aux_degradation_head(values).squeeze(-1) * weights.squeeze(-1)
            return pred, aux_deg
        return pred

    def diagnostic_modules(self):
        return {
            "v_conv": self.v_conv[0],
            "fusion": self.fusion[0],
            "transformer": self.transformer_blocks[-1].ffn[-2],
            "inter_summary": self.inter_summary,
            "aux_head": self.aux_degradation_head,
            "head": self.head[-1],
        }


class FastElectroTransformer(nn.Module):
    """Vectorized Grouped-Conv Channel Feature Extraction + Multi-Head Self-Attention Transformer."""

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        summary_mode="mean_last_delta",
        dropout=0.0,
    ):
        super().__init__()
        self.summary_mode = summary_mode
        self.grouped_channel_conv = nn.Sequential(
            nn.Conv1d(3, 48, kernel_size=7, stride=3, padding=3, groups=3),
            nn.GELU(),
            nn.Conv1d(48, 48, kernel_size=5, stride=2, padding=2, groups=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.raw_embed = nn.Linear(900, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(480 + d_model, d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerCycleBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        summary_width = d_model if summary_mode == "mean" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.aux_degradation_head = nn.Linear(d_model, 1)

    def forward(self, curves, mask, return_aux=False):
        if mask is None:
            raise ValueError("FastElectroTransformer requires a cycle mask")
        batch, cycles = curves.shape[:2]
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(curves.dtype)

        flat_curves = curves.reshape(batch * cycles, 3, 300)
        conv_feat = self.grouped_channel_conv(flat_curves).reshape(batch * cycles, 480)
        raw_feat = self.raw_embed(flat_curves.reshape(batch * cycles, 900))

        fused_tokens = self.fusion(torch.cat([conv_feat, raw_feat], dim=-1))
        values = fused_tokens.reshape(batch, cycles, -1)

        values = (values + self.pos_embed[:, :cycles]) * weights

        for block in self.transformer_blocks:
            values = block(values, key_padding_mask=~observed) * weights

        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)
        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        summary = self.inter_summary(summary)
        pred = self.head(summary).squeeze(-1)
        if return_aux:
            aux_deg = self.aux_degradation_head(values).squeeze(-1) * weights.squeeze(-1)
            return pred, aux_deg
        return pred

    def diagnostic_modules(self):
        return {
            "grouped_conv": self.grouped_channel_conv[0],
            "raw_embed": self.raw_embed,
            "fusion": self.fusion[0],
            "transformer": self.transformer_blocks[-1].ffn[-2],
            "inter_summary": self.inter_summary,
            "aux_head": self.aux_degradation_head,
            "head": self.head[-1],
        }


class MultiScaleElectroTransformer(nn.Module):
    """Multi-Scale Grouped-Conv Channel Feature Extraction (Fine k=3, Med k=7, Coarse k=15) + Cycle Transformer."""

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=3,
        summary_mode="mean_last_delta",
        dropout=0.0,
        enable_aux_head=False,
    ):
        super().__init__()
        self.summary_mode = summary_mode
        self.conv_fine = nn.Sequential(
            nn.Conv1d(3, 24, kernel_size=3, stride=2, padding=1, groups=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.conv_med = nn.Sequential(
            nn.Conv1d(3, 24, kernel_size=7, stride=3, padding=3, groups=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.conv_coarse = nn.Sequential(
            nn.Conv1d(3, 24, kernel_size=15, stride=5, padding=7, groups=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10),
        )
        self.raw_embed = nn.Linear(900, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(240 * 3 + d_model, d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerCycleBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        summary_width = d_model if summary_mode == "mean" else 3 * d_model
        self.inter_summary = nn.Linear(summary_width, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        if enable_aux_head:
            self.aux_degradation_head = nn.Linear(d_model, 1)

    def forward(self, curves, mask, return_aux=False):
        if mask is None:
            raise ValueError("MultiScaleElectroTransformer requires a cycle mask")
        batch, cycles = curves.shape[:2]
        observed = mask.bool()
        weights = observed.unsqueeze(-1).to(curves.dtype)

        flat_curves = curves.reshape(batch * cycles, 3, 300)
        f_fine = self.conv_fine(flat_curves).reshape(batch * cycles, 240)
        f_med = self.conv_med(flat_curves).reshape(batch * cycles, 240)
        f_coarse = self.conv_coarse(flat_curves).reshape(batch * cycles, 240)
        conv_feat = torch.cat([f_fine, f_med, f_coarse], dim=-1)
        raw_feat = self.raw_embed(flat_curves.reshape(batch * cycles, 900))

        fused_tokens = self.fusion(torch.cat([conv_feat, raw_feat], dim=-1))
        values = fused_tokens.reshape(batch, cycles, -1)

        values = (values + self.pos_embed[:, :cycles]) * weights

        for block in self.transformer_blocks:
            values = block(values, key_padding_mask=~observed) * weights

        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        first = values[:, 0]
        last_index = observed.sum(dim=1).long().clamp_min(1) - 1
        last = values.gather(
            1, last_index[:, None, None].expand(-1, 1, values.shape[-1])
        ).squeeze(1)

        summary = mean
        if self.summary_mode == "mean_last_delta":
            summary = torch.cat([mean, last, last - first], dim=-1)
        summary = self.inter_summary(summary)

        pred = self.head(summary).squeeze(-1)
        if return_aux and hasattr(self, "aux_degradation_head"):
            aux_deg = self.aux_degradation_head(values).squeeze(-1)
            return pred, aux_deg
        elif return_aux:
            return pred, None
        return pred

    def diagnostic_modules(self):
        modules = {
            "conv_fine": self.conv_fine[0],
            "conv_med": self.conv_med[0],
            "conv_coarse": self.conv_coarse[0],
            "raw_embed": self.raw_embed,
            "fusion": self.fusion[0],
            "transformer": self.transformer_blocks[-1].ffn[-2],
            "inter_summary": self.inter_summary,
            "head": self.head[-1],
        }
        if hasattr(self, "aux_degradation_head"):
            modules["aux_head"] = self.aux_degradation_head
        return modules



class ResidualConvTemporalTCNSocWindowView(ResidualConvTemporalTCNPhysicalDifferential):
    """D1 (amendment A4): R4-E006 plus a 4th channel = Q / q_ref.

    q_ref is the cell's own early discharge capacity (max over cycles 1..5).
    BatteryLife labels EOL relative to the SOC window
    (Qd / (nominal * SOC_interval) <= 0.8), while the locked input divides Q by
    nominal only. Unobserved cycles are zero-filled and cannot raise the max, so
    q_ref uses observed cycles only. `view="absolute_duplicate"` is the matched
    control (4th channel = absolute Q).
    """

    def __init__(self, view="own_initial", ref_cycles=5, **kwargs):
        super().__init__(**kwargs)
        if view not in {"own_initial", "absolute_duplicate"}:
            raise ValueError("view must be 'own_initial' or 'absolute_duplicate'")
        self.view = view
        self.ref_cycles = ref_cycles
        self._mask = None

    def forward(self, curves, mask):
        self._mask = mask
        try:
            return super().forward(curves, mask)
        finally:
            self._mask = None

    def augment_curves(self, curves):
        capacity = curves[..., 2, :]
        if self.view == "absolute_duplicate":
            extra = capacity
        else:
            early = curves[:, : self.ref_cycles, 2, self.phase_split :]
            if self._mask is not None:
                early = early * self._mask[:, : self.ref_cycles, None].to(early.dtype)
            q_ref = early.amax(dim=(1, 2))
            extra = capacity / q_ref.clamp_min(1e-3)[:, None, None]
        return torch.cat([curves, extra.unsqueeze(-2)], dim=-2)


class BatteryLifeModelWrapper(nn.Module):
    """Run an unchanged BatteryLife model (../BatteryLife/models/<name>.py) in the v4 runner.

    Program.md §2b: the input is passed unchanged ([B,100,3,300], BatteryLife's
    native layout) and the [B,1] output is squeezed to [B]. Architecture hyperparameters come from
    the proposal (Selected_hyperparameters.md, Li-ion row).
    """

    def __init__(self, model_name, model_config):
        super().__init__()
        import importlib
        import sys
        from types import SimpleNamespace

        bl_root = str(ROOT.parent / "BatteryLife")
        added = bl_root not in sys.path
        if added:
            sys.path.insert(0, bl_root)
        try:
            module = importlib.import_module(f"models.{model_name}")
        finally:
            if added:
                sys.path.remove(bl_root)
        # run_main.py defaults, then the proposal's Li-ion values
        cfg = dict(d_model=16, n_heads=4, e_layers=2, d_layers=1, d_ff=32, factor=1,
                   dropout=0.1, activation="relu", lstm_layers=1, moving_avg=25,
                   enc_in=3, dec_in=3, c_out=1, patch_len=16, stride=8, embed="timeF",
                   freq="h", output_attention=False, label_len=0, pred_len=0)
        cfg.update(model_config)
        if "lstm_layers" not in model_config and "d_layers" in model_config:
            cfg["lstm_layers"] = model_config["d_layers"]
        cfg.update(charge_discharge_length=300, early_cycle_threshold=100, output_num=1, seq_len=1)
        self.model_name = model_name
        self.inner = module.Model(SimpleNamespace(**cfg))

    def forward(self, curves, mask=None):
        if mask is None:
            mask = torch.ones(curves.shape[:2], device=curves.device, dtype=curves.dtype)
        out = self.inner(curves, mask)  # [B,100,3,300] is BatteryLife's native loader layout
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(out.shape[0], -1)[:, 0]

    def diagnostic_modules(self):
        """Early, middle and late hookable children of the wrapped model."""
        candidates = []
        for name, module in self.inner.named_children():
            if not any(p.requires_grad for p in module.parameters()):
                continue
            if isinstance(module, nn.ModuleList):
                if len(module) > 0:
                    candidates.append((f"{name}_0", module[0]))
                    if len(module) > 1:
                        candidates.append((f"{name}_{len(module) - 1}", module[-1]))
            else:
                candidates.append((name, module))
        if not candidates:
            return {"inner": self.inner}
        picks = {
            candidates[0][0]: candidates[0][1],
            candidates[len(candidates) // 2][0]: candidates[len(candidates) // 2][1],
            candidates[-1][0]: candidates[-1][1],
        }
        return {f"bl_{k}": v for k, v in picks.items()}


def build_model(config):
    variant = config.get("variant", "cpmlp")
    if variant == "residual_conv_temporal_tcn_soc_window_view":
        return ResidualConvTemporalTCNSocWindowView(
            view=config.get("view", "own_initial"),
            ref_cycles=config.get("ref_cycles", 5),
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "reference_pair_tcn":
        return ReferencePairTemporalTCN(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            k_references=config.get("num_references", config.get("k_references", 16)),
            shuffle_reference_labels=config.get("shuffle_reference_labels", False),
        )
    if variant == "batterylife":
        return BatteryLifeModelWrapper(config["batterylife_model"], config["batterylife_config"])
    if variant == "cpmlp":
        # v5/v6: paper hyper-parameters per dataset come from the proposal (Selected_hyperparameters.md);
        # missing keys keep the Li-ion defaults, so old proposals build the same model.
        return CPMLP(**{k: config[k] for k in ("d_model", "d_ff", "e_layers", "d_layers", "dropout") if k in config})
    if variant == "multi_scale_electro_transformer":
        return MultiScaleElectroTransformer(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 256),
            n_layers=config.get("n_layers", 3),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "fast_electro_transformer":
        return FastElectroTransformer(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 256),
            n_layers=config.get("n_layers", 2),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "electro_transformer":
        return ElectroTransformer(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 256),
            n_layers=config.get("n_layers", 2),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "dual_stream_delta_q_transformer":
        return DualStreamDeltaQTransformer(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 256),
            n_layers=config.get("n_layers", 2),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "conv_cycle_transformer_summary":
        return ConvCycleTransformerSummary(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 256),
            n_layers=config.get("n_layers", 2),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "cycle_transformer_attentive_summary":
        return CycleTransformerAttentiveSummary(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 256),
            n_layers=config.get("n_layers", 2),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "cycle_transformer_summary":
        return CycleTransformerSummary(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 256),
            n_layers=config.get("n_layers", 2),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            enable_aux_head=config.get("enable_aux_head", False),
        )
    if variant == "cycle_patch_transformer":
        return CyclePatchTransformer(
            d_model=config.get("d_model", 32),
            d_ff=config.get("d_ff", 256),
            intra_layers=config.get("intra_layers", 12),
            inter_mlp_layers=config.get("inter_mlp_layers", 7),
            transformer_layers=config.get("transformer_layers", 2),
            n_heads=config.get("n_heads", 4),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "cycle_patch_tcn":
        return CyclePatchTCN(
            d_model=config.get("d_model", 32),
            d_ff=config.get("d_ff", 256),
            intra_layers=config.get("intra_layers", 12),
            inter_mlp_layers=config.get("inter_mlp_layers", 7),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4, 8))),
            dropout=config.get("dropout", 0.0),
        )

    if variant == "residual_conv_masked_summary":

        return ResidualConvMaskedSummary(
            d_model=config.get("d_model", 64),
            residual_layers=config.get("residual_layers", 1),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 4),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
        )
    if variant == "residual_conv_temporal_tcn_masked_summary":
        return ResidualConvTemporalTCNMaskedSummary(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "residual_conv_temporal_tcn_physical_differential":
        return ResidualConvTemporalTCNPhysicalDifferential(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
            derivative_epsilon=config.get("derivative_epsilon", 1e-4),
            derivative_transform=config.get("derivative_transform", "none"),
        )
    if variant == "dual_phase_coordconv_temporal_tcn":
        return DualPhaseCoordConvTemporalTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
        )
    if variant == "dual_phase_residual_coordconv_temporal_tcn":
        return DualPhaseResidualCoordConvTemporalTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
        )
    if variant == "dual_phase_coordconv_phase_contrast_tcn":
        return DualPhaseCoordConvPhaseContrastTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
        )
    if variant == "dual_phase_coordconv_residual_contrast_tcn":
        return DualPhaseCoordConvResidualContrastTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
        )
    if variant == "meta_fingerprint_anchor_tcn":
        return MetaFingerprintAnchorTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
            fp_init_std=config.get("fp_init_std", 1e-3),
            meta_dropout=config.get("meta_dropout", 0.0),
            charge_features=config.get("charge_features", False),
            unbounded_head=config.get("unbounded_head", False),
            late_fade_features=config.get("late_fade_features", False),
            meta_dropout_eval_scale=config.get("meta_dropout_eval_scale", False),
            dual_capacity_view=config.get("dual_capacity_view", False),
        )
    if variant == "deltaq_fingerprint_anchor_tcn":
        return DeltaQFingerprintAnchorTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
        )
    if variant == "meta_conditioned_anchor_tcn":
        return MetaConditionedAnchorTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
        )
    if variant == "dual_phase_coordconv_anchor_tcn":
        return DualPhaseCoordConvAnchorTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
        )
    if variant == "dual_phase_coordconv_detached_anchor_tcn":
        return DualPhaseCoordConvDetachedAnchorTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
        )
    if variant == "dual_phase_coordconv_local_delta_tcn":
        return DualPhaseCoordConvLocalDeltaTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
        )
    if variant == "dual_phase_coordconv_smoothed_delta_tcn":
        return DualPhaseCoordConvSmoothedDeltaTCN(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
            delta_window=config.get("delta_window", 3),
        )
    if variant == "dual_phase_residual_coordconv_cycle_attention":
        return DualPhaseResidualCoordConvCycleAttention(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
            phase_split=config.get("phase_split", 150),
            attention_hidden=config.get("attention_hidden", 32),
        )
    if variant == "residual_conv_anchor_temporal_tcn":
        return ResidualConvAnchorTemporalTCN(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "residual_conv_dual_branch_temporal_tcn":
        return ResidualConvDualBranchTemporalTCN(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "residual_conv_temporal_tcn_capacity_lifecycle":
        return ResidualConvTemporalTCNCapacityLifecycle(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "residual_conv_temporal_tcn_reliability_gated_lifecycle":
        return ResidualConvTemporalTCNReliabilityGatedLifecycle(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "residual_conv_temporal_tcn_adaptive_mixture":
        return ResidualConvTemporalTCNAdaptiveMixture(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            shared_layers=config.get("shared_layers", 5),
            num_experts=config.get("num_experts", 2),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "multiscale_residual_conv_temporal_tcn_masked_summary":
        return MultiScaleResidualConvTemporalTCNMaskedSummary(
            d_model=config.get("d_model", 32),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "residual_conv_temporal_pyramid_tcn":
        return ResidualConvTemporalPyramidTCN(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2, 4))),
            pyramid_levels=tuple(config.get("pyramid_levels", (1, 2, 4))),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "residual_conv_cycle_attention_masked_summary":
        return ResidualConvCycleAttentionMaskedSummary(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            n_heads=config.get("n_heads", 4),
            attention_d_ff=config.get("attention_d_ff", 128),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "residual_conv_gru_masked_summary":
        return ResidualConvGRUMaskedSummary(
            d_model=config.get("d_model", 32),
            residual_layers=config.get("residual_layers", 1),
            d_ff=config.get("d_ff", 256),
            mlp_layers=config.get("mlp_layers", 7),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            dropout=config.get("dropout", 0.0),
        )
    if variant == "masked_summary":
        return MaskedSummaryCPMLP(
            d_model=config.get("d_model", 32),
            d_ff=config.get("d_ff", 256),
            e_layers=config.get("e_layers", 12),
            d_layers=config.get("d_layers", 7),
            dropout=config.get("dropout", 0.0),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
        )
    if variant == "global_gated_innovation":
        return GlobalGatedInnovationCPMLP()
    if variant == "conv_cycle_attention":

        return ConvCycleAttention(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            layers=config.get("layers", 6),
        )
    if variant == "conv_cycle_gru":
        return ConvCycleGRU(
            d_model=config.get("d_model", 64),
            hidden_size=config.get("hidden_size", 128),
            layers=config.get("layers", 2),
        )
    if variant == "simple_conv_mean":
        return SimpleConvMean(d_model=config.get("d_model", 64))
    if variant == "residual_conv_mean":
        return ResidualConvMean(
            d_model=config.get("d_model", 64),
            residual_layers=config.get("residual_layers", 3),
            residual_dilation=config.get("residual_dilation", 1),
        )
    if variant == "residual_conv_attention_last":
        return ResidualConvAttentionLast(
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 128),
        )
    if variant == "normalized_conv_mean":
        return NormalizedConvMean(
            d_model=config.get("d_model", 64),
            layers=config.get("layers", 1),
        )
    if variant == "multiscale_residual_conv_mean":
        return MultiScaleResidualConvMean(d_model=config.get("d_model", 64))
    if variant == "residual_conv_mean_max":
        return ResidualConvMeanMax(d_model=config.get("d_model", 64))
    if variant == "residual_conv_temporal_tcn_mean":
        return ResidualConvTemporalTCNMean(
            d_model=config.get("d_model", 64),
            temporal_dilations=tuple(config.get("temporal_dilations", (1, 2))),
        )
    if variant == "multiscale_curve_tcn":
        return MultiScaleCurveTCN(
            curve_channels=config.get("curve_channels", 64),
            temporal_channels=config.get("temporal_channels", 128),
            temporal_dilations=tuple(
                config.get("temporal_dilations", (1, 2, 4, 8, 16, 32, 64))
            ),
        )
    if variant == "multiscale_conv_masked_summary":
        return MultiScaleConvMaskedSummary(
            d_model=config.get("d_model", 64),
            d_ff=config.get("d_ff", 256),
            trunk_layers=config.get("trunk_layers", 4),
            summary_mode=config.get("summary_mode", "mean_last_delta"),
            head_mode=config.get("head_mode", "linear"),
            feature_mode=config.get("feature_mode", "raw"),
            enable_capacity_ratio=config.get("enable_capacity_ratio", False),
            enable_cycle_conv=config.get("enable_cycle_conv", False),
            conv_groups=config.get("conv_groups", 3),
            context_layers=config.get("context_layers", 2),
            n_heads=config.get("n_heads", 4),
            enable_aux_head=config.get("enable_aux_head", False),
            dropout=config.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown model variant: {variant}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def block_name(parameter_name):
    if parameter_name.startswith("inner."):
        inner_name = parameter_name[len("inner."):]
        parts = inner_name.split(".")
        child = parts[0]
        if len(parts) > 1 and parts[1].isdigit():
            return f"bl_{child}.{parts[1]}"
        if len(parts) > 2 and parts[1] in {"attn_layers", "layers"} and parts[2].isdigit():
            return f"bl_{child}.{parts[2]}"
        return f"bl_{child}"
    parts = parameter_name.split(".")
    if parts[0] in {"transformer_blocks", "attention_blocks"} and len(parts) > 1:
        block = f"{parts[0]}.{parts[1]}"
        if any(part in {"attention", "attn", "self_attn"} for part in parts[2:]):
            return f"{block}.attention"
        if "ffn" in parts[2:]:
            return f"{block}.ffn"
        return block
    if parts[0] in {
        "intra_mlp",
        "inter_mlp",
        "residual_blocks",
        "normalized_blocks",
        "shared_predictor",
    }:
        layer = parts[1] if len(parts) > 1 else "root"
        return f"{parts[0]}.{layer}"
    if parts[0] in {"expert_predictors", "expert_heads"}:
        expert = parts[1] if len(parts) > 1 else "root"
        return f"{parts[0]}.{expert}"
    if parameter_name.startswith("mixture_gate"):
        return "mixture_gate"
    if parameter_name.startswith("raw_embed"):
        return "raw_embed"
    if parameter_name.startswith("delta_embed"):
        return "delta_embed"
    if parameter_name.startswith("fusion"):
        return "fusion"
    if parameter_name.startswith(("pos_embed", "position", "cls_token", "query_token")):
        return "position_tokens"
    if parameter_name.startswith("pool_attention"):
        return "pool_attention"
    if parameter_name.startswith("pool_norm"):
        return "pool_norm"
    if parameter_name.startswith("conv_stem"):
        return "conv_stem"
    if parameter_name.startswith("stem_proj"):
        return "stem_proj"
    if parameter_name.startswith("aux_degradation_head"):
        return "aux_head"
    if parameter_name.startswith("cycle_stem"):
        return "cycle_stem"
    if parameter_name.startswith("multiscale_block"):
        return "multiscale_block"
    if parameter_name.startswith("curve_encoder"):
        return "curve_encoder"
    if parameter_name.startswith("temporal_tcn"):
        return "temporal_tcn"
    if parameter_name.startswith("cycle_encoder"):
        return "cycle_encoder"
    if parameter_name.startswith("attention_blocks"):
        return "cycle_attention"
    if parameter_name.startswith(("cycle_attention", "attention_norm")):
        return "cycle_attention"
    if parameter_name.startswith("attention_ffn"):
        return "attention_ffn"
    if parameter_name.startswith("temporal_gru"):
        return "temporal_gru"
    if parameter_name.startswith("intra_embed"):
        return "intra_embed"
    if parameter_name.startswith("intra_mlp"):
        return "intra_mlp"
    if parameter_name.startswith("inter_flatten"):
        return "inter_flatten"
    if parameter_name.startswith("inter_summary"):
        return "inter_summary"
    if parameter_name.startswith("innovation_gate"):
        return "innovation_gate"
    if parameter_name.startswith("inter_mlp"):
        return "inter_mlp"
    if parameter_name.startswith("head"):
        return "head"
    return parts[0]


def gradient_snapshot(model):
    values = {}
    grouped = {}
    for name, parameter in model.named_parameters():
        group = grouped.setdefault(
            block_name(name),
            {"grad_sq": 0.0, "param_sq": 0.0, "near_zero": 0, "n": 0, "nonfinite": 0},
        )
        parameter_value = parameter.detach()
        finite_parameter = torch.isfinite(parameter_value)
        safe_parameter = torch.where(
            finite_parameter, parameter_value, torch.zeros_like(parameter_value)
        )
        group["param_sq"] += float(torch.sum(safe_parameter * safe_parameter).item())
        gradient = parameter.grad
        if gradient is None:
            group["near_zero"] += parameter.numel()
            group["n"] += parameter.numel()
            continue
        detached = gradient.detach()
        finite = torch.isfinite(detached)
        group["nonfinite"] += int((~finite).sum().item())
        safe = torch.where(finite, detached, torch.zeros_like(detached))
        group["grad_sq"] += float(torch.sum(safe * safe).item())
        group["near_zero"] += int((safe.abs() <= 1e-12).sum().item())
        group["n"] += safe.numel()
    for name, item in grouped.items():
        grad_norm = math.sqrt(item["grad_sq"])
        parameter_norm = math.sqrt(item["param_sq"])
        values[name] = {
            "grad_norm": grad_norm,
            "parameter_norm": parameter_norm,
            "grad_to_parameter_ratio": grad_norm / (parameter_norm + 1e-12),
            "near_zero_gradient_fraction": item["near_zero"] / max(1, item["n"]),
            "nonfinite_gradients": item["nonfinite"],
        }
    return values


def parameter_snapshot(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def update_snapshot(model, before):
    grouped = {}
    for name, parameter in model.named_parameters():
        group = grouped.setdefault(block_name(name), {"delta_sq": 0.0, "weight_sq": 0.0})
        delta = parameter.detach() - before[name]
        group["delta_sq"] += float(torch.sum(delta * delta).item())
        group["weight_sq"] += float(torch.sum(before[name] * before[name]).item())
    return {
        name: math.sqrt(item["delta_sq"]) / (math.sqrt(item["weight_sq"]) + 1e-12)
        for name, item in grouped.items()
    }


def summarize_diagnostics(diagnostics):
    """Turn sampled layer diagnostics into explicit flags and repair guidance."""
    samples = diagnostics.get("samples", [])
    thresholds = {
        "disconnected_grad_norm_max": 1e-12,
        "disconnected_update_ratio_max": 1e-8,
        "near_zero_fraction_warning": 0.50,
        "vanishing_grad_ratio_median": 1e-6,
        "exploding_grad_ratio_max": 100.0,
        "collapsed_activation_std_max": 1e-8,
        "collapsed_activation_zero_fraction": 0.999,
        "physics_gradient_ratio_min": 0.01,
        "physics_gradient_ratio_max": 1.0,
        "physics_loss_ratio_max": 1.0,
        "physics_conflict_cosine_min": -0.3,
    }
    if not samples:
        return {
            "overall_gradient_health": "NOT_AVAILABLE",
            "thresholds": thresholds,
            "layer_health": {},
            "activation_health": {},
            "mask_health": {"status": "NOT_AVAILABLE"},
            "branch_gradient_imbalance": {"status": "NOT_AVAILABLE"},
            "agent_action": "Do not make a gradient-based architecture claim; obtain a valid diagnostic run.",
        }

    gradient_layers = sorted(
        {name for sample in samples for name in sample.get("gradients", {})}
    )
    layer_health = {}
    overall = "HEALTHY"
    for name in gradient_layers:
        gradient_items = [
            sample["gradients"][name]
            for sample in samples
            if name in sample.get("gradients", {})
        ]
        update_values = [
            sample["update_ratios"][name]
            for sample in samples
            if name in sample.get("update_ratios", {})
        ]
        grad_norms = [item["grad_norm"] for item in gradient_items]
        grad_ratios = [item["grad_to_parameter_ratio"] for item in gradient_items]
        near_zero = [item["near_zero_gradient_fraction"] for item in gradient_items]
        nonfinite = sum(item["nonfinite_gradients"] for item in gradient_items)
        flags = []
        actions = []
        if nonfinite > 0:
            flags.append("NONFINITE_GRADIENT")
            actions.append("Inspect the loss/normalization path and lower unstable scale before rerunning.")
        if max(grad_norms, default=0.0) <= thresholds["disconnected_grad_norm_max"]:
            flags.append("DISCONNECTED_OR_ZERO_GRADIENT")
            actions.append("Trace the forward graph and mask; remove an unused layer or reconnect it, then run a matched repair.")
        if update_values and update_values[-1] <= thresholds["disconnected_update_ratio_max"]:
            flags.append("NO_PARAMETER_UPDATE")
            actions.append("Check requires_grad, optimizer membership, graph connectivity, and loss reachability.")
        if near_zero and near_zero[-1] > thresholds["near_zero_fraction_warning"]:
            flags.append("NEAR_ZERO_DOMINANT")
            actions.append("Inspect activation saturation and masking before considering a layer replacement.")
        if (
            grad_ratios[-1] < thresholds["vanishing_grad_ratio_median"]
            and "DISCONNECTED_OR_ZERO_GRADIENT" not in flags
        ):
            flags.append("VANISHING_GRADIENT_CANDIDATE")
            actions.append("Compare early/late ratios; test normalization or a residual path with a matched ablation.")
        if grad_ratios[-1] > thresholds["exploding_grad_ratio_max"]:
            flags.append("EXPLODING_GRADIENT_CANDIDATE")
            actions.append("Inspect normalization, initialization, loss scale, learning rate, and clipping need.")
        status = "UNHEALTHY" if any(
            flag in flags
            for flag in ("NONFINITE_GRADIENT", "DISCONNECTED_OR_ZERO_GRADIENT", "NO_PARAMETER_UPDATE")
        ) else ("WARNING" if flags else "HEALTHY")
        if status == "UNHEALTHY":
            overall = "UNHEALTHY"
        elif status == "WARNING" and overall == "HEALTHY":
            overall = "WARNING"
        layer_health[name] = {
            "status": status,
            "flags": flags,
            "grad_norm": {"first": grad_norms[0], "last": grad_norms[-1], "min": min(grad_norms), "max": max(grad_norms), "median": float(np.median(grad_norms))},
            "grad_to_parameter_ratio": {"first": grad_ratios[0], "last": grad_ratios[-1], "min": min(grad_ratios), "max": max(grad_ratios), "median": float(np.median(grad_ratios))},
            "near_zero_gradient_fraction": {"first": near_zero[0], "last": near_zero[-1], "max": max(near_zero), "median": float(np.median(near_zero))},
            "update_ratio": {"first": update_values[0], "last": update_values[-1], "min": min(update_values), "max": max(update_values), "median": float(np.median(update_values))},
            "nonfinite_gradient_count": nonfinite,
            "recommended_action": actions or ["No gradient-driven repair is indicated; decide with metrics and matched controls."],
        }

    last_gradient_ratios = {
        name: item["grad_to_parameter_ratio"]["last"]
        for name, item in layer_health.items()
        if item["grad_to_parameter_ratio"]["last"] > 0
    }
    branch_gradient_imbalance = {"status": "NOT_AVAILABLE"}
    if last_gradient_ratios:
        smallest_name = min(last_gradient_ratios, key=last_gradient_ratios.get)
        largest_name = max(last_gradient_ratios, key=last_gradient_ratios.get)
        ratio = last_gradient_ratios[largest_name] / last_gradient_ratios[smallest_name]
        branch_gradient_imbalance = {
            "status": "WARNING" if ratio > 1000.0 else "HEALTHY",
            "largest_block": largest_name,
            "smallest_block": smallest_name,
            "largest_to_smallest_ratio": ratio,
            "warning_threshold": 1000.0,
        }
        if ratio > 1000.0 and overall == "HEALTHY":
            overall = "WARNING"

    activation_names = sorted(
        {name for sample in samples for name in sample.get("activations", {})}
    )
    activation_health = {}
    for name in activation_names:
        items = [sample["activations"][name] for sample in samples if name in sample.get("activations", {})]
        flags = []
        if sum(item["nonfinite"] for item in items) > 0:
            flags.append("NONFINITE_ACTIVATION")
        if max((item["std"] for item in items), default=0.0) <= thresholds["collapsed_activation_std_max"]:
            flags.append("CONSTANT_ACTIVATION")
        if float(np.median([item["zero_fraction"] for item in items])) >= thresholds["collapsed_activation_zero_fraction"]:
            flags.append("ZERO_ACTIVATION_COLLAPSE")
        activation_health[name] = {
            "status": "UNHEALTHY" if flags else "HEALTHY",
            "flags": flags,
            "std_min": min(item["std"] for item in items),
            "std_max": max(item["std"] for item in items),
            "zero_fraction_max": max(item["zero_fraction"] for item in items),
            "mean_l2_norm_min": min(item["mean_l2_norm"] for item in items),
            "mean_l2_norm_max": max(item["mean_l2_norm"] for item in items),
            "active_feature_fraction_min": min(
                item["active_feature_fraction"] for item in items
            ),
            "nonfinite_count": sum(item["nonfinite"] for item in items),
        }
        if flags:
            overall = "UNHEALTHY"

    empty_masks_max = max(sample.get("empty_masks", 0) for sample in samples)
    mask_health = {
        "status": "HEALTHY" if empty_masks_max == 0 else "UNHEALTHY",
        "empty_masks_max": empty_masks_max,
        "observed_fraction_min": min(sample.get("mask_observed_fraction", 0.0) for sample in samples),
        "observed_fraction_max": max(sample.get("mask_observed_fraction", 0.0) for sample in samples),
        "note": "This checks empty-mask incidence only; architecture-specific padding leakage needs its own invariance/attention diagnostic.",
    }
    if empty_masks_max:
        overall = "UNHEALTHY"

    physics_items = [
        sample["physics_loss"]
        for sample in samples
        if "physics_loss" in sample
    ]
    physics_balance = {"status": "NOT_APPLICABLE"}
    if physics_items:
        loss_ratios = [item["weighted_to_main_loss_ratio"] for item in physics_items]
        gradient_ratios = [
            item["weighted_to_main_gradient_ratio"] for item in physics_items
        ]
        cosines = [
            item["gradient_cosine"]
            for item in physics_items
            if item["gradient_cosine"] is not None
        ]
        flags = []
        median_loss_ratio = float(np.median(loss_ratios))
        median_gradient_ratio = float(np.median(gradient_ratios))
        median_cosine = float(np.median(cosines)) if cosines else None
        if median_gradient_ratio < thresholds["physics_gradient_ratio_min"]:
            flags.append("PHYSICS_GRADIENT_INEFFECTIVE")
        if median_gradient_ratio > thresholds["physics_gradient_ratio_max"]:
            flags.append("PHYSICS_GRADIENT_DOMINATES")
        if median_loss_ratio > thresholds["physics_loss_ratio_max"]:
            flags.append("PHYSICS_LOSS_DOMINATES")
        if (
            median_cosine is not None
            and median_cosine < thresholds["physics_conflict_cosine_min"]
        ):
            flags.append("PHYSICS_GRADIENT_CONFLICT")
        is_whatif = any(
            str(item.get("type", "")).startswith("whatif_") for item in physics_items
        )
        if is_whatif:
            # loss.md §5 exception: one-sided what-if pushes against memorization,
            # so negative cosine is a reported flag only and not a gating failure.
            severe = {
                "PHYSICS_GRADIENT_DOMINATES",
                "PHYSICS_LOSS_DOMINATES",
            }
        else:
            severe = {
                "PHYSICS_GRADIENT_DOMINATES",
                "PHYSICS_LOSS_DOMINATES",
                "PHYSICS_GRADIENT_CONFLICT",
            }
        physics_status = (
            "UNHEALTHY"
            if severe.intersection(flags)
            else ("WARNING" if flags else "HEALTHY")
        )
        if physics_status == "UNHEALTHY":
            overall = "UNHEALTHY"
        elif physics_status == "WARNING" and overall == "HEALTHY":
            overall = "WARNING"
        physics_balance = {
            "status": physics_status,
            "flags": flags,
            "sample_count": len(physics_items),
            "weighted_to_main_loss_ratio": {
                "min": min(loss_ratios),
                "max": max(loss_ratios),
                "median": median_loss_ratio,
            },
            "weighted_to_main_gradient_ratio": {
                "min": min(gradient_ratios),
                "max": max(gradient_ratios),
                "median": median_gradient_ratio,
            },
            "gradient_cosine": {
                "min": min(cosines) if cosines else None,
                "max": max(cosines) if cosines else None,
                "median": median_cosine,
            },
        }

    loss_balance = {"status": "NOT_ASSESSED"}
    if physics_items:
        balance_samples = []
        r_values = []
        consecutive_above_5 = 0
        max_consecutive_above_5 = 0
        all_active_fractions = []
        for sample in samples:
            if "physics_loss" in sample:
                pl = sample["physics_loss"]
                r_val = float(pl.get("weighted_to_main_gradient_ratio", 0.0))
                act_frac = float(pl.get("active_fraction", 0.0))
                all_active_fractions.append(act_frac)
                r_values.append(r_val)
                if r_val > 5.0:
                    consecutive_above_5 += 1
                    max_consecutive_above_5 = max(max_consecutive_above_5, consecutive_above_5)
                else:
                    consecutive_above_5 = 0
                balance_samples.append({
                    "step": sample.get("step"),
                    "l_main": float(sample.get("main_loss", {}).get("raw", 0.0)),
                    "weighted_l_phys": float(pl.get("weighted", 0.0)),
                    "R": r_val,
                    "gradient_cosine": pl.get("gradient_cosine"),
                    "active_fraction": act_frac,
                })
        if not any(f > 0 for f in all_active_fractions):
            balance_status = "NOT_ASSESSED"
        else:
            median_r = float(np.median(r_values)) if r_values else 0.0
            if median_r < 0.05:
                balance_status = "BALANCE_FAIL_SMALL"
            elif median_r > 2.0 or max_consecutive_above_5 >= 2:
                balance_status = "BALANCE_FAIL_LARGE"
            else:
                balance_status = "PASS"
        loss_balance = {
            "status": balance_status,
            "thresholds": {"r_min": 0.05, "r_max": 2.0, "consecutive_above_5_max": 2},
            "median_R": float(np.median(r_values)) if r_values else 0.0,
            "samples": balance_samples,
        }

    epoch_history = diagnostics.get("epoch_history", [])
    learning_dynamics = {"status": "NOT_AVAILABLE"}
    if epoch_history:
        mapes = [item["validation"]["mape"] for item in epoch_history]
        losses = [item["train_loss_mean"] for item in epoch_history]
        minimum_index = int(np.argmin(mapes))
        learning_dynamics = {
            "status": "AVAILABLE",
            "epochs_observed": len(epoch_history),
            "minimum_mape_epoch": epoch_history[minimum_index]["epoch"],
            "minimum_validation_mape": mapes[minimum_index],
            "first_to_minimum_mape_change": mapes[0] - mapes[minimum_index],
            "last_minus_minimum_mape": mapes[-1] - mapes[minimum_index],
            "first_to_last_train_loss_ratio": losses[-1] / max(losses[0], 1e-12),
            "still_improving_at_boundary": minimum_index == len(epoch_history) - 1,
        }
    return {
        "overall_gradient_health": overall,
        "thresholds": thresholds,
        "layer_health": layer_health,
        "activation_health": activation_health,
        "mask_health": mask_health,
        "branch_gradient_imbalance": branch_gradient_imbalance,
        "physics_balance": physics_balance,
        "loss_balance": loss_balance,
        "learning_dynamics": learning_dynamics,
        "padding_invariance": diagnostics.get(
            "padding_invariance", {"status": "NOT_AVAILABLE"}
        ),
        "agent_action": (
            "Mark the run INVALID and make no KEEP/DISCARD decision; apply the smallest matched repair before comparison."
            if overall == "UNHEALTHY"
            else "Do not replace a layer from gradients alone; use metrics, time trends, and a matched ablation."
        ),
    }


def physics_loss_fn(
    aux_degradation,
    mask,
    curves=None,
    predictions=None,
    physics_type="monotonicity",
    monotonic_tolerance=0.0,
    monotonic_component_weight=0.25,
    return_components=False,
):
    mask = mask.to(aux_degradation.dtype)
    diff = aux_degradation[:, :-1] - aux_degradation[:, 1:]
    pair_mask = mask[:, 1:] * mask[:, :-1]
    mono_violation = F.relu(diff - monotonic_tolerance).square() * pair_mask
    mono_loss = mono_violation.sum() / pair_mask.sum().clamp_min(1.0)

    if physics_type == "capacity_health":
        if curves is None:
            raise ValueError("capacity_health physics requires observed curves")
        capacity = curves[:, :, 2, :].max(dim=-1).values
        initial_capacity = capacity[:, :1].clamp_min(0.01)
        target_degradation = (1.0 - capacity / initial_capacity).clamp(-0.5, 1.5)
        reconstruction_values = F.smooth_l1_loss(
            aux_degradation,
            target_degradation,
            beta=0.02,
            reduction="none",
        )
        reconstruction_loss = (reconstruction_values * mask).sum() / mask.sum().clamp_min(1.0)
        total = reconstruction_loss + monotonic_component_weight * mono_loss
        violation_count = ((diff - monotonic_tolerance) > 0).to(mask.dtype) * pair_mask
        components = {
            "total": total,
            "reconstruction": reconstruction_loss,
            "monotonicity": mono_loss,
            "monotonic_violation_rate": violation_count.sum()
            / pair_mask.sum().clamp_min(1.0),
        }
        return components if return_components else total

    sei_loss = torch.tensor(0.0, device=aux_degradation.device)
    if aux_degradation.shape[1] >= 3:
        accel = (aux_degradation[:, 2:] - aux_degradation[:, 1:-1]) - (
            aux_degradation[:, 1:-1] - aux_degradation[:, :-2]
        )
        triplet_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
        accel_violation = F.relu(accel).square() * triplet_mask
        sei_loss = accel_violation.sum() / triplet_mask.sum().clamp_min(1.0)

    extrap_loss = torch.tensor(0.0, device=aux_degradation.device)
    if (
        physics_type in {"extrapolation", "full_physics"}
        and curves is not None
        and predictions is not None
    ):
        q_curves = curves[:, :, 2, :]
        q_max = q_curves.max(dim=-1).values
        q_1 = q_max[:, 0].clamp_min(0.1)
        last_idx = mask.sum(dim=1).long().clamp_min(1) - 1
        q_last = q_max.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        delta_q = (q_1 - q_last).clamp_min(1e-4)
        prefix_len = last_idx.float().clamp_min(1.0)
        eol_extrap = (prefix_len * 0.2 * q_1 / delta_q).clamp(min=100.0, max=5000.0)
        log_eol_extrap = torch.log(eol_extrap)
        log_pred = predictions.clamp(min=4.0, max=9.0)
        extrap_loss = F.smooth_l1_loss(log_pred, log_eol_extrap, beta=0.2)

    if physics_type == "combined":
        total = mono_loss + 0.5 * sei_loss
    elif physics_type in {"extrapolation", "full_physics"}:
        total = mono_loss + 0.5 * sei_loss + 0.3 * extrap_loss
    else:
        total = mono_loss
    if return_components:
        return {
            "total": total,
            "reconstruction": torch.zeros_like(total),
            "monotonicity": mono_loss,
            "monotonic_violation_rate": (
                ((diff - monotonic_tolerance) > 0).to(mask.dtype) * pair_mask
            ).sum()
            / pair_mask.sum().clamp_min(1.0),
        }
    return total


def physics_monotonicity_loss(aux_degradation, mask):
    return physics_loss_fn(aux_degradation, mask, physics_type="monotonicity")


WHATIF_PHYSICS_TYPES = {
    "whatif_fade_p0",
    "whatif_fade_p0_flip",
    "whatif_fade_p0b",
    "prefix_drift",
    "whatif_trend_p4",
    "whatif_trend_p4b",
    "whatif_knee_p0c",
    "whatif_vsag_p2",
}


def compute_whatif_fade_ramp(device=None, dtype=torch.float32):
    """Compute ramp_n = clamp((n - 10)/90, 0, 1) for n = 1..100 (t = n - 1)."""
    n = torch.arange(1, 101, device=device, dtype=dtype)
    return torch.clamp((n - 10.0) / 90.0, 0.0, 1.0)


def apply_whatif_fade(curves, masks=None, alpha=None):
    """
    Apply what-if fade to curves out of place:
    x̃ = curves.clone()
    Multiply only the discharge half of the Q channel (x̃[:, :, 2, 150:])
    by s_n = 1 - α * ramp_n * mask_n, with ramp_n = clamp((n - 10)/90, 0, 1) and n = 1..100.
    Leave V, I, charge Q and masked cycles unchanged (masked cycles stay zero).
    """
    x_tilde = curves.clone()
    ramp = compute_whatif_fade_ramp(device=curves.device, dtype=curves.dtype)
    if alpha is None:
        return x_tilde
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=curves.device, dtype=curves.dtype)
    else:
        alpha = alpha.to(device=curves.device, dtype=curves.dtype)
    if alpha.ndim == 0:
        alpha = alpha.view(1, 1)
    elif alpha.ndim == 1:
        alpha = alpha.view(-1, 1)
    if masks is not None:
        s = 1.0 - (alpha * ramp.view(1, -1)) * masks.to(dtype=curves.dtype)
    else:
        s = 1.0 - (alpha * ramp.view(1, -1))
    x_tilde[:, :, 2, 150:] = x_tilde[:, :, 2, 150:] * s.unsqueeze(-1)
    if masks is not None:
        x_tilde = x_tilde * masks.unsqueeze(-1).unsqueeze(-1)
    return x_tilde


def apply_whatif_knee(curves, masks=None, alpha=None):
    """
    P0c knee (loss.md §4.6):
    x̃_{n,2,150:300} = x_{n,2,150:300} * (1 - α * r_n^2).
    """
    x_tilde = curves.clone()
    ramp = compute_whatif_fade_ramp(device=curves.device, dtype=curves.dtype)
    if alpha is None:
        return x_tilde
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=curves.device, dtype=curves.dtype)
    else:
        alpha = alpha.to(device=curves.device, dtype=curves.dtype)
    if alpha.ndim == 0:
        alpha = alpha.view(1, 1)
    elif alpha.ndim == 1:
        alpha = alpha.view(-1, 1)
    ramp2 = ramp.square()
    if masks is not None:
        s = 1.0 - (alpha * ramp2.view(1, -1)) * masks.to(dtype=curves.dtype)
    else:
        s = 1.0 - (alpha * ramp2.view(1, -1))
    x_tilde[:, :, 2, 150:] = x_tilde[:, :, 2, 150:] * s.unsqueeze(-1)
    if masks is not None:
        x_tilde = x_tilde * masks.unsqueeze(-1).unsqueeze(-1)
    return x_tilde


def apply_whatif_vsag(curves, masks=None, beta=None):
    """
    P2 voltage sag (loss.md §4.6):
    x̃_{n,0,150:300} = x_{n,0,150:300} - β * r_n.
    """
    x_tilde = curves.clone()
    ramp = compute_whatif_fade_ramp(device=curves.device, dtype=curves.dtype)
    if beta is None:
        return x_tilde
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, device=curves.device, dtype=curves.dtype)
    else:
        beta = beta.to(device=curves.device, dtype=curves.dtype)
    if beta.ndim == 0:
        beta = beta.view(1, 1)
    elif beta.ndim == 1:
        beta = beta.view(-1, 1)
    delta = (beta * ramp.view(1, -1)).unsqueeze(-1)
    x_tilde[:, :, 0, 150:] = x_tilde[:, :, 0, 150:] - delta
    if masks is not None:
        x_tilde = x_tilde * masks.unsqueeze(-1).unsqueeze(-1)
    return x_tilde


def apply_whatif_trend(curves, masks=None, kappa=None, prefix_lengths=None, falling_only=False):
    """
    P4 own-trend acceleration (loss.md §4.6):
    x̃_{n,2,150:300} = x_{10,2,150:300} + (1 + κ * r_n) * (x_{n,2,150:300} - x_{10,2,150:300}) for n >= 11.
    Applies only to samples with prefix >= 20.
    """
    x_tilde = curves.clone()
    if kappa is None:
        return x_tilde
    if not torch.is_tensor(kappa):
        kappa = torch.tensor(kappa, device=curves.device, dtype=curves.dtype)
    else:
        kappa = kappa.to(device=curves.device, dtype=curves.dtype)
    if kappa.ndim == 0:
        kappa = kappa.view(1, 1)
    elif kappa.ndim == 1:
        kappa = kappa.view(-1, 1)

    if prefix_lengths is None:
        if masks is not None:
            prefix_lengths = masks.sum(dim=-1).long()
        else:
            prefix_lengths = torch.full((curves.size(0),), 100, device=curves.device, dtype=torch.long)
    elif not torch.is_tensor(prefix_lengths):
        prefix_lengths = torch.tensor(prefix_lengths, device=curves.device, dtype=torch.long)
    else:
        prefix_lengths = prefix_lengths.to(device=curves.device, dtype=torch.long)

    valid_mask = prefix_lengths >= 20
    if not valid_mask.any():
        if masks is not None:
            x_tilde = x_tilde * masks.unsqueeze(-1).unsqueeze(-1)
        return x_tilde

    ramp = compute_whatif_fade_ramp(device=curves.device, dtype=curves.dtype)
    factor = 1.0 + kappa * ramp.view(1, -1)
    x10 = curves[:, 9:10, 2, 150:]
    delta = curves[:, :, 2, 150:] - x10
    if falling_only:  # P4b (loss.md §4.6): accelerate only the capacity-loss part; rising (break-in) parts unchanged
        new_q = x10 + delta + (factor.unsqueeze(-1) - 1.0) * delta.clamp(max=0.0)
    else:
        new_q = x10 + factor.unsqueeze(-1) * delta

    valid_expanded = valid_mask.view(-1, 1, 1)
    x_tilde[:, :, 2, 150:] = torch.where(valid_expanded, new_q, x_tilde[:, :, 2, 150:])
    if masks is not None:
        x_tilde = x_tilde * masks.unsqueeze(-1).unsqueeze(-1)
    return x_tilde


def apply_whatif(curves, masks=None, param=None, physics_type="whatif_fade_p0", prefix_lengths=None):
    if physics_type in {"whatif_fade_p0", "whatif_fade_p0_flip", "whatif_fade_p0b"}:
        return apply_whatif_fade(curves, masks=masks, alpha=param)
    elif physics_type == "whatif_knee_p0c":
        return apply_whatif_knee(curves, masks=masks, alpha=param)
    elif physics_type == "whatif_vsag_p2":
        return apply_whatif_vsag(curves, masks=masks, beta=param)
    elif physics_type in ("whatif_trend_p4", "whatif_trend_p4b"):
        return apply_whatif_trend(curves, masks=masks, kappa=param, prefix_lengths=prefix_lengths,
                                  falling_only=(physics_type == "whatif_trend_p4b"))
    else:
        raise ValueError(f"Unknown what-if physics_type: {physics_type}")


def compute_alpha_max_per_domain(train_cells):
    """
    α_max[domain] = 90th percentile over training cells of that domain of
    (Q10 - Q100) / Q10, where Q_n = max discharge Q of cycle n.
    Fall back to global training 90th percentile for domains with < 5 cells.
    """
    domain_ratios = {}
    all_ratios = []
    for cell in train_cells:
        tensor = cell["tensor"]
        # n = 10 -> index 9; n = 100 -> index 99
        # Channel 2 (Q), discharge half is index 150:
        q10 = float(tensor[9, 2, 150:].max())
        q100 = float(tensor[99, 2, 150:].max())
        denom = max(q10, 1e-6)
        ratio = (q10 - q100) / denom
        domain = cell["domain"]
        domain_ratios.setdefault(domain, []).append(ratio)
        all_ratios.append(ratio)

    global_p90 = float(np.percentile(all_ratios, 90))
    alpha_max_by_domain = {"__global__": max(0.0, global_p90)}
    for domain, ratios in domain_ratios.items():
        if len(ratios) >= 5:
            p90 = float(np.percentile(ratios, 90))
            alpha_max_by_domain[domain] = max(0.0, p90)
        else:
            alpha_max_by_domain[domain] = max(0.0, global_p90)
    return alpha_max_by_domain


def compute_beta_max_per_domain(train_cells):
    """
    β_max per domain = 90th percentile over training cells of that domain of
    the training drop in mean discharge V (normalized channel 0, points 150–299),
    cycle 10 → 100. Drop = v10 - v100.
    Fall back to global training 90th percentile for domains with < 5 cells.
    """
    domain_drops = {}
    all_drops = []
    for cell in train_cells:
        tensor = cell["tensor"]
        # n = 10 -> index 9; n = 100 -> index 99
        # Channel 0 (V), discharge half is index 150:300
        v10 = float(tensor[9, 0, 150:].mean())
        v100 = float(tensor[99, 0, 150:].mean())
        drop = v10 - v100
        domain = cell["domain"]
        domain_drops.setdefault(domain, []).append(drop)
        all_drops.append(drop)

    global_p90 = float(np.percentile(all_drops, 90))
    beta_max_by_domain = {"__global__": max(0.0, global_p90)}
    for domain, drops in domain_drops.items():
        if len(drops) >= 5:
            p90 = float(np.percentile(drops, 90))
            beta_max_by_domain[domain] = max(0.0, p90)
        else:
            beta_max_by_domain[domain] = max(0.0, global_p90)
    return beta_max_by_domain


def compute_fade_slope_per_domain(train_cells, min_cells=5):
    """P0b (loss.md §4.6): k_d = -slope of ln(EOL) vs relative fade phi=(Q10-Q100)/Q10,
    ordinary least squares on training cells of domain d (global fit if < min_cells). k >= 0."""
    import numpy as _np
    per = {}
    for cell in train_cells:
        tensor = cell.get("tensor") if isinstance(cell, dict) else None
        if tensor is None or float(cell.get("eol", 0)) <= 0:
            continue
        q10 = float(tensor[9, 2, 150:].max()); q100 = float(tensor[99, 2, 150:].max())
        if q10 <= 1e-6 or q100 <= 0:
            continue
        per.setdefault(cell["domain"], []).append(((q10 - q100) / q10, math.log(float(cell["eol"]))))
    def fit(pairs):
        if len(pairs) < 2:
            return 0.0
        x = _np.array([a for a, _ in pairs]); y = _np.array([b for _, b in pairs])
        if _np.var(x) < 1e-12:
            return 0.0
        return max(0.0, float(-_np.cov(x, y, bias=True)[0, 1] / _np.var(x)))
    allp = [p for v in per.values() for p in v]
    out = {"__global__": fit(allp)}
    for d, pairs in per.items():
        out[d] = fit(pairs) if len(pairs) >= min_cells else out["__global__"]
    return out


def compute_bounds_per_domain(train_cells, physics_type="whatif_fade_p0"):
    if physics_type in {"whatif_fade_p0", "whatif_fade_p0_flip", "whatif_knee_p0c", "whatif_fade_p0b"}:
        return compute_alpha_max_per_domain(train_cells)
    elif physics_type == "whatif_vsag_p2":
        return compute_beta_max_per_domain(train_cells)
    elif physics_type in ("whatif_trend_p4", "whatif_trend_p4b"):
        return {"__global__": 1.0}
    else:
        return {}


def sample_whatif_alpha(domains, alpha_max_by_domain, rng):
    """
    Sample α ~ U(0, α_max[domain]) per sample using a dedicated CPU RNG.
    Never touches loader/global RNG, so step-1 batches stay identical to Phase 1.
    """
    global_p90 = alpha_max_by_domain.get("__global__", 0.0) if alpha_max_by_domain else 0.0
    batch_size = len(domains)
    u = torch.rand(batch_size, generator=rng, dtype=torch.float32)
    upper_bounds = torch.tensor(
        [max(0.0, alpha_max_by_domain.get(dom, global_p90)) for dom in domains],
        dtype=torch.float32,
    )
    return u * upper_bounds


def sample_whatif_beta(domains, beta_max_by_domain, rng):
    """
    Sample β ~ U(0, β_max[domain]) per sample using a dedicated CPU RNG.
    """
    global_p90 = beta_max_by_domain.get("__global__", 0.0) if beta_max_by_domain else 0.0
    batch_size = len(domains)
    u = torch.rand(batch_size, generator=rng, dtype=torch.float32)
    upper_bounds = torch.tensor(
        [max(0.0, beta_max_by_domain.get(dom, global_p90)) for dom in domains],
        dtype=torch.float32,
    )
    return u * upper_bounds


def sample_whatif_kappa(batch_size, rng):
    """
    Sample κ ~ U(0, 1) per sample using a dedicated CPU RNG.
    """
    return torch.rand(batch_size, generator=rng, dtype=torch.float32)



def log_eol_ratio(f_a, f_b, mu_over_s, floor=1e-3):
    """log(yhat_a / yhat_b) for standardized predictions f (yhat = scale * (f + mu/s)); scale cancels."""
    a = (f_a + mu_over_s).clamp_min(floor)
    b = (f_b + mu_over_s).clamp_min(floor)
    return torch.log(a) - torch.log(b)


def truncate_prefix_view(curves, masks, rng, low=0.5):
    """Prefix-drift (loss.md §4.2, label-free): a shorter view of the same sample.
    i = observed cycles; j = clamp(floor(u*i), 1, i-1), u ~ U(low, 1); cycles >= j are zeroed and masked."""
    i = masks.sum(dim=-1).long()
    u = torch.rand(curves.size(0), generator=rng).to(curves.device) * (1.0 - low) + low
    j = torch.where(i >= 2, (u * i.float()).floor().long().clamp(min=1), i)
    j = torch.minimum(j, (i - 1).clamp(min=1))
    idx = torch.arange(curves.size(1), device=curves.device)[None, :]
    new_mask = (idx < j[:, None]).to(masks.dtype)
    return curves * new_mask[:, :, None, None].to(curves.dtype), new_mask


# ---------------- Degradation-law physics loss (user-approved 2026-11-03) ----------------
# q_hat(n) = q_ref - (q_ref - q_eol) * (n / y_hat)^b  (q_eol = smoothed q at the label EOL, ~0.8), y_hat = predicted EOL (cycles), b = softplus(law_head(h)) + 0.3
# L_law = mean_k |q_hat(n_k) - q(n_k)| / (q_ref - q_eol), on K points of the FULL train trajectory (train cells only).
LAW_K = 32


def load_law_targets(path=None, k=LAW_K):
    path = Path(path) if path else Path(__file__).resolve().parent / ".cache" / "qtraj_train_v1.pt"
    raw = torch.load(path, weights_only=False)
    out = {}
    for fn, d in raw.items():
        q = d["q"].double().numpy()
        n_end = int(min(len(q), math.ceil(d["eol"])))
        if n_end < 12 or not np.isfinite(q[:n_end]).any():
            continue
        qs = q[:n_end].copy()
        good = np.isfinite(qs)
        qs = np.interp(np.arange(n_end), np.where(good)[0], qs[good])
        # rolling median (window 9) removes single-cycle capacity spikes
        pad = np.pad(qs, 4, mode="edge")
        qs = np.median(np.lib.stride_tricks.sliding_window_view(pad, 9), axis=1)
        q_ref = float(np.median(qs[: min(10, n_end)]))
        q_eol = float(qs[n_end - 1])  # label-consistent threshold (ISU-ILCC labels cross ~0.8 slightly off)
        if q_ref - q_eol < 0.02:
            continue
        n = np.unique(np.round(np.linspace(1, n_end, k)).astype(int))
        n = np.resize(n, k)
        out[fn] = (torch.tensor(n, dtype=torch.float32), torch.tensor(qs[n - 1], dtype=torch.float32), q_ref, q_eol)
    return out


def attach_law_head(model, out_dim=1):
    if getattr(model, "law_head", None) is None:
        width = model.head.in_features
        gen = torch.Generator().manual_seed(777)
        head = nn.Linear(width, out_dim)
        with torch.no_grad():
            head.weight.copy_(torch.randn(head.weight.shape, generator=gen) * 0.01)
            head.bias.fill_(0.5)
            if out_dim == 4:  # [log-EOL offset, log b1, log b2, logit w]: init b1 ~ 0.6 (SEI), b2 ~ 4 (knee), w 0.5
                head.bias.copy_(torch.tensor([0.0, -0.5, 1.4, 0.0]))
        model.law_head = head
        model._law_h = None
        model.head.register_forward_hook(lambda m, i, o: setattr(model, "_law_h", i[0]))
    return model


def degradation_law_loss(model, predictions, cell_ids, law_targets, scaler, target_transform="standard"):
    h = model._law_h
    idx = [k for k, c in enumerate(cell_ids) if c in law_targets]
    if not idx or h is None:
        z = predictions.sum() * 0.0
        return z, {"total": z, "active_fraction": 0.0, "monotonic_violation_rate": 0.0, "monotonicity": z, "reconstruction": z}
    dev = predictions.device
    sel = torch.tensor(idx, device=dev)
    pred = predictions[sel]
    y_hat = torch.exp(pred.clamp(max=12.0)) if target_transform == "log" else pred * scaler.scale + scaler.mean
    y_hat = y_hat.clamp_min(20.0)
    b = F.softplus(model.law_head(h[sel]).squeeze(-1)) + 0.3
    n = torch.stack([law_targets[cell_ids[k]][0] for k in idx]).to(dev)
    q = torch.stack([law_targets[cell_ids[k]][1] for k in idx]).to(dev)
    q_ref = torch.tensor([law_targets[cell_ids[k]][2] for k in idx], device=dev, dtype=q.dtype)
    q_eol = torch.tensor([law_targets[cell_ids[k]][3] for k in idx], device=dev, dtype=q.dtype)
    budget = (q_ref - q_eol)[:, None]
    q_hat = q_ref[:, None] - budget * (n / y_hat[:, None]).pow(b[:, None])
    err = (q_hat - q).abs() / budget
    loss = err.mean()
    active = float((err.mean(dim=1) > 0.05).float().mean().item())
    return loss, {"total": loss, "active_fraction": active, "monotonic_violation_rate": active,
                  "monotonicity": loss, "reconstruction": torch.tensor(0.0, device=dev),
                  "b_mean": float(b.mean().item()), "coverage": len(idx) / max(len(cell_ids), 1)}



def degradation_law2_loss(model, predictions, cell_ids, law_targets, scaler, target_transform="standard"):
    """Two-term law with its OWN EOL parameter (R4-E048):
    y_law = stopgrad(y_hat) * exp(d);  q_hat(n) = q_ref - (q_ref - q_eol) * [w (n/y_law)^b1 + (1-w) (n/y_law)^b2]
    The law head (4 outputs) and the shared trunk learn from the full train trajectory; the EOL head gets no law gradient."""
    h = model._law_h
    idx = [k for k, c in enumerate(cell_ids) if c in law_targets]
    if not idx or h is None:
        z = predictions.sum() * 0.0
        return z, {"total": z, "active_fraction": 0.0, "monotonic_violation_rate": 0.0, "monotonicity": z, "reconstruction": z}
    dev = predictions.device
    sel = torch.tensor(idx, device=dev)
    pred = predictions[sel].detach()
    y_hat = torch.exp(pred.clamp(max=12.0)) if target_transform == "log" else pred * scaler.scale + scaler.mean
    y_hat = y_hat.clamp_min(20.0)
    out = model.law_head(h[sel])
    y_law = y_hat * torch.exp(out[:, 0].clamp(-2.0, 2.0))
    b1 = torch.exp(out[:, 1].clamp(-3.0, 3.0)); b2 = torch.exp(out[:, 2].clamp(-3.0, 3.0)); w = torch.sigmoid(out[:, 3])
    n = torch.stack([law_targets[cell_ids[k]][0] for k in idx]).to(dev)
    q = torch.stack([law_targets[cell_ids[k]][1] for k in idx]).to(dev)
    q_ref = torch.tensor([law_targets[cell_ids[k]][2] for k in idx], device=dev, dtype=q.dtype)
    q_eol = torch.tensor([law_targets[cell_ids[k]][3] for k in idx], device=dev, dtype=q.dtype)
    budget = (q_ref - q_eol)[:, None]
    x = (n / y_law[:, None]).clamp(max=4.0)
    shape = w[:, None] * x.pow(b1[:, None]) + (1 - w[:, None]) * x.pow(b2[:, None])
    q_hat = q_ref[:, None] - budget * shape
    err = (q_hat - q).abs() / budget
    loss = err.mean()
    active = float((err.mean(dim=1) > 0.05).float().mean().item())
    true_eol = n.max(dim=1).values
    return loss, {"total": loss, "active_fraction": active, "monotonic_violation_rate": active,
                  "monotonicity": loss, "reconstruction": torch.tensor(0.0, device=dev),
                  "law_eol_rel_err": float(((y_law - true_eol).abs() / true_eol).mean().item())}



def timewarp_aug_loss(model, batch, scaler, config, rng, device, target_transform="standard"):
    """Physics time-warp augmentation (R4-E049): same degradation mechanism run kappa times faster.
    x_k(cycle m) = x(cycle floor(kappa*m)), prefix i' = min(ceil(i/kappa), 100, ceil(EOL/kappa)-1), EOL' = EOL/kappa,
    kappa log-uniform in [kmin, kmax]. L_aug = RelHuber(f(x_k), EOL') (same main loss, exact physical label)."""
    curves = batch["cycle_curve_data"].to(device); masks = batch["curve_attn_mask"].to(device)
    eol = batch["raw_eol"].to(device)
    B, C = curves.shape[:2]
    kmin, kmax = float(config.get("timewarp_kmin", 0.6)), float(config.get("timewarp_kmax", 1.6))
    u = torch.rand(B, generator=rng).to(device)
    kappa = torch.exp(math.log(kmin) + u * (math.log(kmax) - math.log(kmin)))
    i = masks.sum(dim=-1)
    eol_new = eol / kappa
    i_new = torch.minimum(torch.ceil(i / kappa), torch.full_like(i, C))
    i_new = torch.minimum(i_new, torch.ceil(eol_new) - 1).clamp_min(1.0)
    m = torch.arange(C, device=device, dtype=curves.dtype)[None, :]
    src = torch.floor(m * kappa[:, None]).long()
    src = torch.minimum(src, (i.long() - 1).clamp_min(0)[:, None])
    valid = (m < i_new[:, None]).to(curves.dtype)
    x = torch.gather(curves, 1, src[:, :, None, None].expand(-1, -1, curves.shape[2], curves.shape[3]))
    x = x * valid[:, :, None, None]
    pred = model(x, valid)
    target = torch.log(eol_new.clamp_min(1.0)) if target_transform == "log" else (eol_new - float(scaler.mean)) / float(scaler.scale)
    loss = compute_main_loss(pred, target, eol_new, scaler, target_transform, config)
    return loss, {"total": loss, "active_fraction": 1.0, "monotonic_violation_rate": 0.0, "monotonicity": loss,
                  "reconstruction": torch.tensor(0.0, device=device), "kappa_mean": float(kappa.mean().item())}



def budget_bound_loss(predictions, curves, masks, scaler, target_transform="standard", gate=0.05, slack=2.0):
    """Remaining-budget physics bound (v2 SOC-normalized input, R4-E06x):
    q(n) = max discharge capacity channel of cycle n; for prefix i >= 20: rate = -slope of q over the last <=40 observed
    cycles, b = mean q of the last 5 observed cycles - 0.8. Active only if b < gate and rate > 1e-5:
    EOL_phys = i + b/rate;  L = mean_active ReLU(y_hat - slack*EOL_phys) / s."""
    q = curves[:, :, 2, 150:].amax(dim=-1)
    i = masks.sum(dim=-1).long()
    B, C = q.shape
    idx = torch.arange(C, device=q.device)[None, :]
    win = ((idx < i[:, None]) & (idx >= (i - 40).clamp_min(0)[:, None])).to(q.dtype)
    nwin = win.sum(dim=1).clamp_min(1.0)
    x = idx.to(q.dtype).expand(B, -1)
    mx = (x * win).sum(1) / nwin; my = (q * win).sum(1) / nwin
    slope = (((x - mx[:, None]) * (q - my[:, None]) * win).sum(1) / (((x - mx[:, None]) ** 2) * win).sum(1).clamp_min(1e-6))
    last5 = ((idx < i[:, None]) & (idx >= (i - 5).clamp_min(0)[:, None])).to(q.dtype)
    q_i = (q * last5).sum(1) / last5.sum(1).clamp_min(1.0)
    b = q_i - 0.8; rate = -slope
    active = (i >= 20) & (b < gate) & (rate > 1e-5)
    y_hat = torch.exp(predictions.clamp(max=12.0)) if target_transform == "log" else predictions * scaler.scale + scaler.mean
    eol_phys = i.to(q.dtype) + b.clamp_min(0.0) / rate.clamp_min(1e-5)
    viol = F.relu(y_hat - slack * eol_phys) / float(scaler.scale)
    loss = (viol * active.to(viol.dtype)).sum() / active.sum().clamp_min(1)
    return loss, float(active.float().mean().item())


def stat_tail_balance_loss(predictions, raw_eol, scaler, target_transform="standard", tau=0.55, scale_max=3.0,
                           delta=0.15, d_over=0.5, d_under=0.25):
    """Statistical tail term (loss.md §4.7 stat_tail_balance): r = (yhat - y)/max(y,1); e+ = ReLU(r - delta),
    e- = ReLU(-r - delta), w = clip(y/mu, 1, scale_max);
    L = mean[tau*Huber_{d_over}(e+) + (1 - tau)*w*Huber_{d_under}(e-)]. Uses labels (statistical, not physics)."""
    y_hat = torch.exp(predictions.clamp(max=12.0)) if target_transform == "log" else predictions * scaler.scale + scaler.mean
    y = raw_eol.clamp_min(1.0)
    r = (y_hat - y) / y
    e_pos = F.relu(r - delta); e_neg = F.relu(-r - delta)
    w = (y / float(scaler.mean)).clamp(1.0, scale_max)
    z = torch.zeros_like(r)
    l_pos = F.huber_loss(e_pos, z, delta=d_over, reduction="none")
    l_neg = F.huber_loss(e_neg, z, delta=d_under, reduction="none")
    loss = (tau * l_pos + (1.0 - tau) * w * l_neg).mean()
    return loss, {"active_fraction": float(((e_pos > 0) | (e_neg > 0)).float().mean().item()),
                  "over_fraction": float((e_pos > 0).float().mean().item()),
                  "under_fraction": float((e_neg > 0).float().mean().item())}


def secondary_physics_loss(kind, model, predictions, batch, curves, masks, scaler, config, rng, device, target_transform,
                           unsigned=False, return_meta=False):
    if kind == "budget_bound":
        return budget_bound_loss(predictions, curves, masks, scaler, target_transform,
                                 gate=float(config.get("budget_gate", 0.05)), slack=float(config.get("budget_slack", 2.0)))[0]
    if kind == "stat_tail_balance":
        l, meta = stat_tail_balance_loss(predictions, batch["raw_eol"].to(predictions.device), scaler, target_transform,
                                         tau=float(config.get("tail_tau", 0.55)), scale_max=float(config.get("tail_scale_max", 3.0)))
        return (l, meta) if return_meta else l
    if kind == "timewarp_aug":
        return timewarp_aug_loss(model, batch, scaler, config, rng, device, target_transform)[0]
    if kind in WHATIF_PHYSICS_TYPES:
        l, meta = whatif_loss(model, curves, masks, batch["domain"], config.get("_phys2_bounds"), rng, physics_type=kind,
                              predictions=predictions, prefix_lengths=batch.get("prefix_length"),
                              anchor_gradient=("both" if kind == "prefix_drift" else "stop"),  # loss.md §4.6: what-if anchor is sg[f(x)]
                              unsigned=(unsigned and kind != "prefix_drift"))
        return (l, meta) if return_meta else l
    raise ValueError(f"unsupported physics2_type {kind}")



def build_rarity_weights(train_cells, alpha=0.5, max_w=5.0, bw=0.25, grid_n=256):
    """Statistical rarity weighting (R4-E060): w(EOL) = clip(density(log EOL)^-alpha / Z, <= max_w),
    density = Gaussian KDE (bandwidth bw in log units) over training cells, each cell weighted by its prefix count;
    Z normalizes the mean weight over training samples to 1."""
    eol = np.array([c["eol"] for c in train_cells], dtype=np.float64)
    cnt = np.array([min(100, int(c["observed_cycles"]), math.ceil(c["eol"]) - 1) for c in train_cells], dtype=np.float64)
    le = np.log(eol)
    grid = np.linspace(le.min() - 0.5, le.max() + 0.5, grid_n)
    dens = (cnt[None, :] * np.exp(-0.5 * ((grid[:, None] - le[None, :]) / bw) ** 2)).sum(1)
    dens = dens / dens.max()
    raw = np.power(np.maximum(dens, 1e-6), -alpha)
    w_cells = np.interp(le, grid, raw)
    z = (w_cells * cnt).sum() / cnt.sum()
    w = np.minimum(raw / z, max_w)
    return {"grid": torch.tensor(grid, dtype=torch.float32), "w": torch.tensor(w, dtype=torch.float32),
            "summary": {"alpha": alpha, "max_w": max_w, "bw": bw,
                        "w_at_eol": {str(e): float(np.interp(np.log(e), grid, w)) for e in (150, 500, 1000, 2000, 3000, 4000)}}}


def rarity_weighted_relative_huber(predictions, raw_eol, scaler, target_transform, config, table):
    pred = torch.exp(predictions.clamp(max=12.0)) if target_transform == "log" else predictions * scaler.scale + scaler.mean
    rel = (pred - raw_eol) / raw_eol.clamp_min(1.0)
    per = F.huber_loss(rel, torch.zeros_like(rel), delta=config.get("relative_huber_delta", 0.15), reduction="none")
    grid = table["grid"].to(raw_eol.device); wt = table["w"].to(raw_eol.device)
    idx = torch.bucketize(torch.log(raw_eol.clamp_min(1.0)), grid).clamp(0, len(grid) - 1)
    return (per * wt[idx]).mean()


def sample_whatif_param(domains, bounds_by_domain, rng, physics_type="whatif_fade_p0"):
    if physics_type in {"whatif_fade_p0", "whatif_fade_p0_flip", "whatif_knee_p0c", "whatif_fade_p0b"}:
        return sample_whatif_alpha(domains, bounds_by_domain, rng)
    elif physics_type == "whatif_vsag_p2":
        return sample_whatif_beta(domains, bounds_by_domain, rng)
    elif physics_type in ("whatif_trend_p4", "whatif_trend_p4b"):
        return sample_whatif_kappa(len(domains), rng)
    else:
        raise ValueError(f"Unknown physics_type for sampling: {physics_type}")


def whatif_loss(
    model,
    curves,
    masks,
    domains,
    bounds_by_domain=None,
    rng=None,
    physics_type="whatif_fade_p0",
    predictions=None,
    prefix_lengths=None,
    alpha_max_by_domain=None,
    anchor_gradient="stop",
    k_by_domain=None,
    mu_over_s=None,
    unsigned=False,
    drift_log_mu_over_s=None,
):
    """
    Computes what-if physics loss:
    L_phys = mean ReLU(f(x̃) - stopgrad f(x)) for whatif_fade_p0, whatif_knee_p0c, whatif_vsag_p2, whatif_trend_p4
    L_phys = mean ReLU(stopgrad f(x) - f(x̃)) for whatif_fade_p0_flip
    """
    if bounds_by_domain is None:
        bounds_by_domain = alpha_max_by_domain
    if predictions is None:
        predictions = model(curves, masks)
    f_x = predictions if anchor_gradient == "both" else predictions.detach()  # loss.md §4.6 P0-sym
    if physics_type == "prefix_drift":
        x_tilde, masks_t = truncate_prefix_view(curves, masks, rng)
        f_tilde = model(x_tilde, masks_t)
        drift = f_tilde - f_x
        if drift_log_mu_over_s is not None:  # loss.md: prefix_drift in log-EOL space (scale-free)
            drift = log_eol_ratio(f_tilde, f_x, drift_log_mu_over_s)
        phys_loss = drift.abs().mean()
        active_fraction = float((drift.abs() > 0.05).float().mean().item())
        return phys_loss, {"total": phys_loss, "active_fraction": active_fraction,
                           "monotonic_violation_rate": active_fraction, "monotonicity": phys_loss,
                           "reconstruction": torch.tensor(0.0, device=curves.device)}
    base_type = "whatif_fade_p0" if physics_type == "whatif_fade_p0b" else physics_type
    param = sample_whatif_param(domains, bounds_by_domain, rng, physics_type=base_type).to(curves.device)
    x_tilde = apply_whatif(curves, masks, param, physics_type=base_type, prefix_lengths=prefix_lengths)
    f_tilde = model(x_tilde, masks)

    if physics_type == "whatif_fade_p0_flip":
        diff = f_x - f_tilde
    else:
        diff = f_tilde - f_x
    if physics_type == "whatif_fade_p0b" and k_by_domain is not None and mu_over_s is not None:
        # P0b (loss.md §4.6): y_cf <= y_x * exp(-tau * k_d * alpha)
        #   <=> f_cf - f_x + (mu/s + f_x) * (1 - exp(-tau k_d alpha)) <= 0   (standardized units)
        k_vec = torch.tensor([k_by_domain.get(d, k_by_domain.get("__global__", 0.0)) for d in domains],
                             device=curves.device, dtype=f_x.dtype)
        tau = float(k_by_domain.get("__tau__", 0.5))
        drop = 1.0 - torch.exp(-tau * k_vec * param.to(f_x.dtype).reshape(-1))
        diff = diff + (float(mu_over_s) + f_x) * drop

    violation = diff.abs() if unsigned else torch.relu(diff)  # unsigned: lambda calibration only (loss.md §5.2)
    if physics_type in ("whatif_trend_p4", "whatif_trend_p4b"):
        if prefix_lengths is None:
            if masks is not None:
                prefix_lengths = masks.sum(dim=-1).long()
            else:
                prefix_lengths = torch.full((curves.size(0),), 100, device=curves.device, dtype=torch.long)
        elif not torch.is_tensor(prefix_lengths):
            prefix_lengths = torch.tensor(prefix_lengths, device=curves.device, dtype=torch.long)
        else:
            prefix_lengths = prefix_lengths.to(device=curves.device, dtype=torch.long)
        valid_mask = prefix_lengths >= 20
        if valid_mask.any():
            valid_f = valid_mask.float()
            phys_loss = (violation * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            active_fraction = float(((diff > 0) & valid_mask).float().sum().item() / valid_f.sum().clamp_min(1.0).item())
            violation_rate = active_fraction
        else:
            phys_loss = (f_tilde * 0.0).sum()
            active_fraction = 0.0
            violation_rate = 0.0
    else:
        phys_loss = violation.mean()
        active_fraction = float((diff > 0).float().mean().item())
        violation_rate = active_fraction

    return phys_loss, {
        "total": phys_loss,
        "active_fraction": active_fraction,
        "monotonic_violation_rate": violation_rate,
        "monotonicity": phys_loss,
        "reconstruction": torch.tensor(0.0, device=curves.device),
    }


whatif_fade_loss = whatif_loss


def compute_p5_horizon_loss(predictions, prefix_lengths, scaler, target_transform="standard"):
    """
    P5 horizon validity (loss.md §4.6):
    L_P5 = mean ReLU((i - y_hat) / s)
    where i is prefix length and y_hat is predicted EOL in raw cycles.
    """
    if target_transform == "log":
        pred_eol = torch.exp(predictions.clamp(max=12.0))
    else:
        pred_eol = predictions * scaler.scale + scaler.mean
    s = scaler.scale
    if not torch.is_tensor(prefix_lengths):
        prefix_lengths = torch.tensor(
            prefix_lengths, device=predictions.device, dtype=predictions.dtype
        )
    else:
        prefix_lengths = prefix_lengths.to(
            device=predictions.device, dtype=predictions.dtype
        )
    violation = F.relu((prefix_lengths - pred_eol) / s)
    return violation.mean()


_GROUP_DRO_Q = {}


def group_dro_relative_huber(predictions, raw_eol, scaler, domains, config):
    """loss.md §4.4 online Group-DRO (Sagawa et al., 2020) on Relative Huber:
    q_g <- q_g * exp(eta * L_g) (normalized over all seen domains); L = sum_g q_g L_g / sum_g q_g over
    domains present in the batch. Same per-sample loss and units as the locked main loss."""
    eta = float(config.get("group_dro_eta", 0.1))
    pred_unscaled = predictions * scaler.scale + scaler.mean
    rel = (pred_unscaled - raw_eol) / raw_eol.clamp_min(1.0)
    per = F.huber_loss(rel, torch.zeros_like(rel), delta=config.get("relative_huber_delta", 0.15), reduction="none")
    groups = {}
    for i, d in enumerate(domains):
        groups.setdefault(d, []).append(i)
    losses = {d: per[idx].mean() for d, idx in groups.items()}
    for d, l in losses.items():
        _GROUP_DRO_Q[d] = _GROUP_DRO_Q.get(d, 1.0) * math.exp(eta * float(l.detach()))
    total = sum(_GROUP_DRO_Q.values())
    for d in _GROUP_DRO_Q:
        _GROUP_DRO_Q[d] /= total
    weights = {d: _GROUP_DRO_Q[d] for d in losses}
    wsum = sum(weights.values())
    return sum(weights[d] * losses[d] for d in losses) / wsum


def compute_main_loss(
    predictions,
    target,
    raw_eol,
    scaler,
    target_transform,
    config,
):
    main_loss_name = config.get("main_loss", "mse")
    if main_loss_name == "mse":
        return F.mse_loss(predictions, target)
    elif main_loss_name == "relative_huber":
        if target_transform == "log":
            pred_unscaled = torch.exp(predictions.clamp(max=12.0))
        else:
            pred_unscaled = predictions * scaler.scale + scaler.mean
        relative_error = (pred_unscaled - raw_eol) / raw_eol.clamp_min(1.0)
        return F.huber_loss(
            relative_error,
            torch.zeros_like(relative_error),
            delta=config.get("relative_huber_delta", 0.15),
        )
    elif main_loss_name == "huber":
        return F.huber_loss(
            predictions,
            target,
            delta=config.get("huber_delta", 1.0),
        )
    elif main_loss_name in {"relative_l1", "smooth_mape", "mape"}:
        if target_transform == "log":
            pred_unscaled = torch.exp(predictions.clamp(max=12.0))
        else:
            pred_unscaled = predictions * scaler.scale + scaler.mean
        if main_loss_name == "smooth_mape":
            abs_diff = F.smooth_l1_loss(
                pred_unscaled,
                raw_eol,
                beta=config.get("beta", 15.0),
                reduction="none",
            )
        else:
            abs_diff = (pred_unscaled - raw_eol).abs()
        return (abs_diff / raw_eol.clamp_min(1.0)).mean()
    elif main_loss_name == "log_cosh_mape":
        if target_transform == "log":
            pred_unscaled = torch.exp(predictions.clamp(max=12.0))
        else:
            pred_unscaled = predictions * scaler.scale + scaler.mean
        rel_err = (pred_unscaled - raw_eol) / raw_eol.clamp_min(1.0)
        return (
            rel_err.abs()
            + F.softplus(-2.0 * rel_err.abs())
            - math.log(2.0)
        ).mean()
    elif main_loss_name == "compound_mape_logmse":
        if target_transform == "log":
            pred_unscaled = torch.exp(predictions.clamp(max=12.0))
        else:
            pred_unscaled = predictions * scaler.scale + scaler.mean
        rel_err = (pred_unscaled - raw_eol) / raw_eol.clamp_min(1.0)
        log_cosh_term = (
            rel_err.abs()
            + F.softplus(-2.0 * rel_err.abs())
            - math.log(2.0)
        ).mean()
        log_mse_term = F.mse_loss(predictions, target)
        alpha = config.get("compound_loss_alpha", 0.5)
        return alpha * log_cosh_term + (1.0 - alpha) * log_mse_term
    else:
        raise ValueError(f"Unknown main loss: {main_loss_name}")


def calibrate_whatif_lambda(
    model,
    cal_batches,
    bounds_by_domain=None,
    physics_rng=None,
    config=None,
    device=None,
    scaler=None,
    target_transform="standard",
    physics_type="whatif_fade_p0",
    alpha_max_by_domain=None,
):
    """
    Calibrate λ before step 1 on 3 fixed training batches:
    λ = ρ · median(‖∇L_main‖ / ‖∇L_cal‖), with ρ = config["physics_rho"] (default 0.5)
    and unsigned L_cal = mean |f(x̃) - stopgrad f(x)|.
    Use full-parameter float64 norms with a floor of 1e-12. Log λ.
    """
    if bounds_by_domain is None:
        bounds_by_domain = alpha_max_by_domain
    if config is None:
        config = {}
    model.train()
    saved_state = copy.deepcopy(model.state_dict())
    ratios = []
    ratios_p5 = []
    has_p5 = config.get("physics_horizon_weight", 0.0) > 0

    for batch in cal_batches:
        curves = batch["cycle_curve_data"].to(device)
        masks = batch["curve_attn_mask"].to(device)
        labels = batch["label"].to(device)
        raw_eol = batch["raw_eol"].to(device)
        domains = batch["domain"]
        prefix_lengths = batch.get("prefix_length")

        model.zero_grad(set_to_none=True)
        predictions = model(curves, masks)
        target = (
            torch.log(raw_eol.clamp_min(1.0))
            if target_transform == "log"
            else labels
        )
        main_loss = compute_main_loss(
            predictions, target, raw_eol, scaler, target_transform, config
        )

        # physics_calibration_mode="match" calibrates in the training gradient mode (loss.md §5.2 repair, R4-E042)
        f_x = predictions if (config.get("physics_calibration_mode", "stop") == "match" and config.get("physics_anchor_gradient", "stop") == "both") else predictions.detach()
        if physics_type == "prefix_drift":
            x_tilde, masks_t = truncate_prefix_view(curves, masks, physics_rng)
            param = torch.zeros(curves.size(0), device=device)
            f_tilde = model(x_tilde, masks_t)
        else:
            param = sample_whatif_param(domains, bounds_by_domain, physics_rng, physics_type=physics_type).to(device)
            x_tilde = apply_whatif(curves, masks, param, physics_type=physics_type, prefix_lengths=prefix_lengths)
            f_tilde = model(x_tilde, masks)
        raw_diff = f_tilde - f_x
        if physics_type == "prefix_drift" and config.get("physics_drift_space", "standard") == "log" and scaler is not None:
            raw_diff = log_eol_ratio(f_tilde, f_x, float(scaler.mean) / float(scaler.scale))
        k_cal = config.get("_p0b_k_by_domain") if physics_type == "whatif_fade_p0b" else None
        if k_cal is not None and scaler is not None:
            # R4-E044 fix: calibrate lambda on the actual P0b loss (including its margin term)
            k_vec = torch.tensor([k_cal.get(d, k_cal.get("__global__", 0.0)) for d in domains], device=device, dtype=f_x.dtype)
            drop = 1.0 - torch.exp(-float(k_cal.get("__tau__", 0.5)) * k_vec * param.to(f_x.dtype).reshape(-1))
            raw_diff = raw_diff + (float(scaler.mean) / float(scaler.scale) + f_x) * drop
        abs_diff = torch.abs(raw_diff)
        if physics_type in ("whatif_trend_p4", "whatif_trend_p4b"):
            if prefix_lengths is None:
                prefix_lengths = masks.sum(dim=-1).long()
            elif not torch.is_tensor(prefix_lengths):
                prefix_lengths = torch.tensor(prefix_lengths, device=device, dtype=torch.long)
            else:
                prefix_lengths = prefix_lengths.to(device=device, dtype=torch.long)
            valid_mask = prefix_lengths >= 20
            if valid_mask.any():
                L_cal = (abs_diff * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
            else:
                L_cal = torch.mean(abs_diff)
        else:
            L_cal = torch.mean(abs_diff)

        params = [p for p in model.parameters() if p.requires_grad]
        g_main = torch.autograd.grad(
            main_loss, params, retain_graph=True, allow_unused=True
        )
        g_cal = torch.autograd.grad(
            L_cal, params, retain_graph=has_p5, allow_unused=True
        )

        norm_main_sq = 0.0
        for g in g_main:
            if g is not None:
                g_d = g.detach().cpu().double()
                norm_main_sq += float(torch.sum(g_d * g_d).cpu())
        norm_main = math.sqrt(norm_main_sq)

        norm_cal_sq = 0.0
        for g in g_cal:
            if g is not None:
                g_d = g.detach().cpu().double()
                norm_cal_sq += float(torch.sum(g_d * g_d).cpu())
        norm_cal = math.sqrt(norm_cal_sq)

        norm_main = max(norm_main, 1e-12)
        norm_cal = max(norm_cal, 1e-12)
        ratios.append(norm_main / norm_cal)

        if has_p5 and "prefix_length" in batch:
            prefix_lengths = batch["prefix_length"]
            if target_transform == "log":
                pred_eol = torch.exp(predictions.clamp(max=12.0))
            else:
                pred_eol = predictions * scaler.scale + scaler.mean
            if not torch.is_tensor(prefix_lengths):
                pref_t = torch.tensor(prefix_lengths, device=predictions.device, dtype=predictions.dtype)
            else:
                pref_t = prefix_lengths.to(device=predictions.device, dtype=predictions.dtype)
            L_cal_p5 = torch.mean(torch.abs(pref_t - pred_eol) / scaler.scale)
            g_cal_p5 = torch.autograd.grad(
                L_cal_p5, params, retain_graph=False, allow_unused=True
            )
            norm_cal_p5_sq = 0.0
            for g in g_cal_p5:
                if g is not None:
                    g_d = g.detach().cpu().double()
                    norm_cal_p5_sq += float(torch.sum(g_d * g_d).cpu())
            norm_cal_p5 = max(math.sqrt(norm_cal_p5_sq), 1e-12)
            ratios_p5.append(norm_main / norm_cal_p5)

    model.zero_grad(set_to_none=True)
    model.load_state_dict(saved_state)

    rho = float(config.get("physics_rho", 0.5))
    median_ratio = float(np.median(ratios))
    calibrated_lambda = float(rho * median_ratio)
    print(
        f"[physics] Calibrated lambda = {calibrated_lambda:.6e} "
        f"(rho={rho}, median_ratio={median_ratio:.6e}, batch_ratios={[round(r, 6) for r in ratios]})",
        flush=True,
    )
    calibrated_lambda_p5 = None
    if has_p5 and ratios_p5:
        median_ratio_p5 = float(np.median(ratios_p5))
        calibrated_lambda_p5 = float(rho * median_ratio_p5)
        print(
            f"[physics] Calibrated lambda_p5 = {calibrated_lambda_p5:.6e} "
            f"(rho={rho}, median_ratio={median_ratio_p5:.6e})",
            flush=True,
        )
    return calibrated_lambda, {
        "rho": rho,
        "median_ratio": median_ratio,
        "batch_ratios": ratios,
        "lambda": calibrated_lambda,
        "lambda_p5": calibrated_lambda_p5,
    }


@torch.no_grad()
def evaluate_whatif_violation_rate(
    model,
    val_loader,
    device,
    bounds_by_domain=None,
    physics_type="whatif_fade_p0",
    multipliers=(0.25, 0.5, 1.0),
    alpha_max_by_domain=None,
    mu_over_s=None,
):
    if physics_type == "prefix_drift":
        # held-out inconsistency: share of samples whose half-to-full prefix prediction differs > 15% (cycles)
        model.eval(); rng = torch.Generator().manual_seed(12345); n = 0; bad = 0; rel_sum = 0.0; rel_std_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                curves = batch["cycle_curve_data"].to(device); masks = batch["curve_attn_mask"].to(device)
                f_x = model(curves, masks); xt, mt = truncate_prefix_view(curves, masks, rng); f_t = model(xt, mt)
                rel_std = (f_t - f_x).abs() / f_x.abs().clamp_min(1e-3)
                if mu_over_s is not None:  # Codex R4-E083: relative drift of unscaled EOL (cycles)
                    rel = torch.expm1(log_eol_ratio(f_t, f_x, mu_over_s)).abs()
                else:
                    rel = rel_std
                bad += int((rel > 0.15).sum()); n += int(rel.numel()); rel_sum += float(rel.sum()); rel_std_sum += float(rel_std.sum())
        return {"total_evaluations": n, "total_violations": bad, "pooled_violation_rate": bad / max(n, 1),
                "mean_relative_drift": rel_sum / max(n, 1), "mean_relative_drift_std_units": rel_std_sum / max(n, 1),
                "drift_space": "eol_cycles" if mu_over_s is not None else "standardized",
                "note": "prefix_drift: share of held-out samples whose shorter-prefix EOL prediction differs > 15%"}
    if bounds_by_domain is None:
        bounds_by_domain = alpha_max_by_domain
    model.eval()
    global_p90 = bounds_by_domain.get("__global__", 0.0) if bounds_by_domain else 0.0
    total_evals = 0
    total_violations = 0
    by_multiplier = {m: {"evals": 0, "violations": 0} for m in multipliers}
    by_domain = {}

    for batch in val_loader:
        curves = batch["cycle_curve_data"].to(device)
        masks = batch["curve_attn_mask"].to(device)
        domains = batch["domain"]
        prefix_lengths = batch.get("prefix_length")
        if prefix_lengths is None and masks is not None:
            prefix_lengths = masks.sum(dim=-1).long()
        elif prefix_lengths is not None and not torch.is_tensor(prefix_lengths):
            prefix_lengths = torch.tensor(prefix_lengths, device=device, dtype=torch.long)
        elif prefix_lengths is not None and torch.is_tensor(prefix_lengths):
            prefix_lengths = prefix_lengths.to(device=device, dtype=torch.long)

        f_x = model(curves, masks)
        B = curves.size(0)

        if physics_type in ("whatif_trend_p4", "whatif_trend_p4b"):
            valid_samples = (prefix_lengths >= 20).cpu().numpy()
        else:
            valid_samples = np.ones(B, dtype=bool)

        valid_count = int(valid_samples.sum())
        if valid_count == 0:
            continue

        for m in multipliers:
            if physics_type in ("whatif_trend_p4", "whatif_trend_p4b"):
                param_vals = torch.full((B,), float(m), dtype=torch.float32, device=device)
            else:
                param_vals = torch.tensor(
                    [
                        m * max(0.0, bounds_by_domain.get(dom, global_p90))
                        for dom in domains
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            x_tilde = apply_whatif(curves, masks, param_vals, physics_type=physics_type, prefix_lengths=prefix_lengths)
            f_tilde = model(x_tilde, masks)

            if physics_type == "whatif_fade_p0_flip":
                viol = ((f_x > f_tilde).cpu().numpy()) & valid_samples
            else:
                viol = ((f_tilde > f_x).cpu().numpy()) & valid_samples

            viol_count = int(viol.sum())
            total_evals += valid_count
            total_violations += viol_count
            by_multiplier[m]["evals"] += valid_count
            by_multiplier[m]["violations"] += viol_count

            for i, dom in enumerate(domains):
                if valid_samples[i]:
                    if dom not in by_domain:
                        by_domain[dom] = {"evals": 0, "violations": 0}
                    by_domain[dom]["evals"] += 1
                    if bool(viol[i]):
                        by_domain[dom]["violations"] += 1

    pooled_rate = total_violations / max(total_evals, 1)
    multiplier_rates = {
        f"violation_rate_alpha_{m}": (
            by_multiplier[m]["violations"] / max(by_multiplier[m]["evals"], 1)
        )
        for m in multipliers
    }
    domain_rates = {
        dom: stats["violations"] / max(stats["evals"], 1)
        for dom, stats in sorted(by_domain.items())
    }
    return {
        "pooled_violation_rate": float(pooled_rate),
        "total_evaluations": total_evals,
        "total_violations": total_violations,
        "by_multiplier": multiplier_rates,
        "by_domain": domain_rates,
    }


def evaluate_checkpoint_violation(experiment_id, device=None, physics_type=None):
    if device is None:
        device = choose_device()
    exp_dir = ROOT / "experiments" / experiment_id
    ckpt_path = exp_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    config = ckpt.get("config", {})
    if physics_type is None:
        physics_type = config.get("physics_type", "whatif_fade_p0")

    model = build_model(ckpt["model"]).to(device)
    sd = ckpt["state_dict"]
    if any(k.startswith("law_head.") for k in sd):
        attach_law_head(model, sd["law_head.weight"].shape[0]).to(device)
    model.load_state_dict(sd)
    model.eval()

    train_loader, val_loader, scaler = prepare.load_train_val_data(32, 2021)
    bounds_by_domain = compute_bounds_per_domain(train_loader.dataset.cells, physics_type)

    res = evaluate_whatif_violation_rate(
        model,
        val_loader,
        device,
        bounds_by_domain,
        physics_type=physics_type,
    )
    print(f"\n===== PHYSICS VIOLATION RATE EVALUATION: {experiment_id} ({physics_type}) =====", flush=True)
    print(f"Pooled violation rate: {res['pooled_violation_rate']:.4f}", flush=True)
    print(f"Total evaluations: {res['total_evaluations']} (violations: {res['total_violations']})", flush=True)
    print("By alpha multiplier:", flush=True)
    for mult, rate in (res.get("by_multiplier") or {}).items():
        print(f"  {mult}: {rate:.4f}", flush=True)
    print("By domain:", flush=True)
    for dom, rate in (res.get("by_domain") or {}).items():
        print(f"  {dom}: {rate:.4f}", flush=True)
    return res


def prefix_consistency_loss(predictions, cell_ids):
    groups = {}
    for index, cell_id in enumerate(cell_ids):
        groups.setdefault(cell_id, []).append(index)
    active_groups = [indices for indices in groups.values() if len(indices) >= 2]
    if not active_groups:
        return predictions.sum() * 0.0, {
            "active_fraction": 0.0,
            "duplicate_groups": 0,
        }
    deviations = []
    active_samples = 0
    for indices in active_groups:
        values = predictions[indices]
        deviations.append((values - values.mean()).square())
        active_samples += len(indices)
    return torch.cat(deviations).mean(), {
        "active_fraction": active_samples / len(cell_ids),
        "duplicate_groups": len(active_groups),
    }


def loss_gradient_interaction(model, main_loss, weighted_auxiliary):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    main_gradients = torch.autograd.grad(
        main_loss, parameters, retain_graph=True, allow_unused=True
    )
    auxiliary_gradients = torch.autograd.grad(
        weighted_auxiliary, parameters, retain_graph=True, allow_unused=True
    )
    main_sq = auxiliary_sq = dot = 0.0
    shared_main_sq = shared_auxiliary_sq = shared_dot = 0.0
    for main_gradient, auxiliary_gradient, parameter in zip(
        main_gradients, auxiliary_gradients, parameters
    ):
        main_value = torch.zeros_like(parameter) if main_gradient is None else main_gradient
        auxiliary_value = (
            torch.zeros_like(parameter)
            if auxiliary_gradient is None
            else auxiliary_gradient
        )
        main_sq += float(torch.sum(main_value * main_value).detach().cpu())
        auxiliary_sq += float(
            torch.sum(auxiliary_value * auxiliary_value).detach().cpu()
        )
        dot += float(torch.sum(main_value * auxiliary_value).detach().cpu())
        if main_gradient is not None and auxiliary_gradient is not None:
            shared_main_sq += float(
                torch.sum(main_gradient * main_gradient).detach().cpu()
            )
            shared_auxiliary_sq += float(
                torch.sum(auxiliary_gradient * auxiliary_gradient).detach().cpu()
            )
            shared_dot += float(
                torch.sum(main_gradient * auxiliary_gradient).detach().cpu()
            )
    main_norm = math.sqrt(main_sq)
    auxiliary_norm = math.sqrt(auxiliary_sq)
    total_cosine = None
    if main_norm > 0 and auxiliary_norm > 0:
        total_cosine = dot / (main_norm * auxiliary_norm)
    shared_main_norm = math.sqrt(shared_main_sq)
    shared_auxiliary_norm = math.sqrt(shared_auxiliary_sq)
    shared_cosine = None
    if shared_main_norm > 0 and shared_auxiliary_norm > 0:
        shared_cosine = shared_dot / (shared_main_norm * shared_auxiliary_norm)
    return {
        "main_gradient_norm": main_norm,
        "weighted_auxiliary_gradient_norm": auxiliary_norm,
        "gradient_cosine": shared_cosine,
        "all_parameter_gradient_cosine": total_cosine,
        "shared_main_gradient_norm": shared_main_norm,
        "shared_weighted_auxiliary_gradient_norm": shared_auxiliary_norm,
    }


class ActivationCollector:
    def __init__(self, model):
        self.enabled = False
        self.current = {}
        if hasattr(model, "diagnostic_modules"):
            modules = model.diagnostic_modules()
        else:
            modules = {
                "intra_embed": model.intra_embed,
                "intra_mlp": model.intra_mlp[-1],
                "inter_mlp": model.inter_mlp[-1],
                "head": model.head,
            }
            if hasattr(model, "inter_flatten"):
                modules["inter_flatten"] = model.inter_flatten
            if hasattr(model, "inter_summary"):
                modules["inter_summary"] = model.inter_summary
        self.handles = [module.register_forward_hook(self._hook(name)) for name, module in modules.items()]

    def _hook(self, name):
        def collect(_module, _inputs, output):
            if not self.enabled:
                return
            value = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(value, torch.nn.utils.rnn.PackedSequence):
                value = value.data
            elif hasattr(value, "data") and isinstance(getattr(value, "data"), torch.Tensor) and not isinstance(value, torch.Tensor):
                value = value.data
            if not isinstance(value, torch.Tensor):
                return
            value = value.detach()
            flattened = value.reshape(-1, value.shape[-1]) if value.ndim > 1 else value[:, None]
            feature_std = flattened.float().std(dim=0, unbiased=False)
            self.current[name] = {
                "mean": float(value.mean().item()),
                "std": float(value.std(unbiased=False).item()),
                "zero_fraction": float((value == 0).float().mean().item()),
                "nonfinite": int((~torch.isfinite(value)).sum().item()),
                "mean_l2_norm": float(
                    flattened.float().norm(dim=-1).mean().item()
                ),
                "active_feature_fraction": float((feature_std > 1e-6).float().mean().item()),
            }
        return collect

    def close(self):
        for handle in self.handles:
            handle.remove()


def _group_metrics(predictions, targets, groups, cell_ids):
    output = {}
    groups = np.asarray(groups)
    cell_ids = np.asarray(cell_ids)
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        values = prepare.compute_metrics(predictions[selected], targets[selected])
        signed_error = predictions[selected] - targets[selected]
        signed_relative_error = signed_error / np.maximum(targets[selected], 1.0)
        values["bias"] = float(np.mean(signed_error))
        values["relative_bias"] = float(np.mean(signed_relative_error))
        values["prediction_mean"] = float(np.mean(predictions[selected]))
        values["target_mean"] = float(np.mean(targets[selected]))
        values["sample_count"] = int(selected.sum())
        values["unique_cell_count"] = int(len(set(cell_ids[selected].tolist())))
        output[str(group)] = values
    return output


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    scaler,
    target_transform="standard",
    include_groups=False,
    return_predictions=False,
):
    model.eval()
    predictions, targets, prefixes, domains, cell_ids = [], [], [], [], []
    started = time.perf_counter()
    for batch in loader:
        curves = batch["cycle_curve_data"].to(device)
        mask = batch["curve_attn_mask"].to(device)
        pred = model(curves, mask)
        if target_transform == "log":
            unscaled = torch.exp(pred.clamp(max=12.0)).cpu().numpy()
        else:
            scaled = pred.cpu().numpy()
            unscaled = scaler.inverse_transform(scaled)
        predictions.append(unscaled)
        targets.append(batch["raw_eol"].numpy())
        if include_groups or return_predictions:
            prefixes.append(batch["prefix_length"].numpy())
            domains.extend(batch["domain"])
            cell_ids.extend(batch["cell_id"])
    sync(device)
    predictions = np.concatenate(predictions).reshape(-1)
    targets = np.concatenate(targets).reshape(-1)
    metrics = prepare.compute_metrics(predictions, targets)
    signed_relative_error = (predictions - targets) / np.maximum(targets, 1.0)
    prediction_std = float(np.std(predictions))
    target_std = float(np.std(targets))
    correlation = None
    if prediction_std > 0 and target_std > 0:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
    metrics["prediction_diagnostics"] = {
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": prediction_std,
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "target_mean": float(np.mean(targets)),
        "target_std": target_std,
        "prediction_to_target_std_ratio": prediction_std / max(target_std, 1e-12),
        "mean_signed_relative_error": float(np.mean(signed_relative_error)),
        "relative_error_std": float(np.std(signed_relative_error)),
        "prediction_target_correlation": correlation,
        "extreme_prediction_count": int(
            ((predictions <= 0) | (predictions > 10000)).sum()
        ),
    }
    if include_groups or return_predictions:
        prefixes = np.concatenate(prefixes).reshape(-1)
    if include_groups:
        prefix_groups = np.select(
            [
                prefixes <= 10,
                prefixes <= 25,
                prefixes <= 50,
                prefixes <= 75,
            ],
            ["01-10", "11-25", "26-50", "51-75"],
            default="76-100",
        )
        eol_groups = np.select(
            [targets < 500, targets <= 1000],
            ["low_<500", "middle_500-1000"],
            default="high_>1000",
        )
        metrics["by_prefix"] = _group_metrics(
            predictions, targets, prefix_groups, cell_ids
        )
        metrics["by_domain"] = _group_metrics(
            predictions, targets, domains, cell_ids
        )
        metrics["by_eol"] = _group_metrics(
            predictions, targets, eol_groups, cell_ids
        )
    elapsed = time.perf_counter() - started
    if not return_predictions:
        return metrics, elapsed
    absolute_relative_error = np.abs(signed_relative_error)
    artifact = {
        "prediction": predictions.astype(np.float32),
        "target": targets.astype(np.float32),
        "cell_id": np.asarray(cell_ids, dtype=np.str_),
        "domain": np.asarray(domains, dtype=np.str_),
        "prefix_length": prefixes.astype(np.int16),
        "signed_relative_error": signed_relative_error.astype(np.float32),
        "absolute_relative_error": absolute_relative_error.astype(np.float32),
        "acc15_margin": (0.15 - absolute_relative_error).astype(np.float32),
    }
    return metrics, elapsed, artifact


@torch.no_grad()
def padding_invariance_check(model, loader, device):
    """Measure model sensitivity to non-zero values in masked cycle slots."""
    model.eval()
    batch = next(iter(loader))
    curves = batch["cycle_curve_data"].to(device)
    masks = batch["curve_attn_mask"].to(device)
    padding = (~masks.bool())[:, :, None, None].expand_as(curves)
    cached_padding_max = float(curves.masked_select(padding).abs().max().cpu())
    reference = model(curves, masks)
    perturbed = torch.where(padding, torch.full_like(curves, 0.731), curves)
    changed = model(perturbed, masks)
    difference = (changed - reference).abs()
    return {
        "status": "INVARIANT" if float(difference.max()) <= 1e-6 else "SENSITIVE",
        "cached_padding_absolute_max": cached_padding_max,
        "mean_absolute_standardized_output_change": float(difference.mean().cpu()),
        "max_absolute_standardized_output_change": float(difference.max().cpu()),
        "interpretation": (
            "SENSITIVE means the model depends on upstream zero filling. It does "
            "not mean the locked cached input leaks future cycles."
        ),
    }


@torch.no_grad()
def evaluate_physics_health(
    model,
    loader,
    device,
    config,
    bounds_by_domain=None,
    alpha_max_by_domain=None,
):
    """Evaluate the auxiliary degradation state or what-if physics health on observed validation cycles."""
    if bounds_by_domain is None:
        bounds_by_domain = alpha_max_by_domain
    if config.get("physics_loss_weight", 0.0) <= 0:
        return {"status": "NOT_APPLICABLE"}
    physics_type = config.get("physics_type", "monotonicity")
    if physics_type in {"degradation_law", "degradation_law2", "timewarp_aug"}:
        return {"status": "NOT_APPLICABLE", "reason": "law targets exist for train cells only (val trajectories are never read)"}
    if physics_type in WHATIF_PHYSICS_TYPES:
        if bounds_by_domain is None and physics_type not in ("whatif_trend_p4", "whatif_trend_p4b"):
            return {"status": "NOT_APPLICABLE", "reason": "missing_bounds_by_domain"}
        if bounds_by_domain is None and physics_type in ("whatif_trend_p4", "whatif_trend_p4b"):
            bounds_by_domain = {"__global__": 1.0}
        violation_stats = evaluate_whatif_violation_rate(
            model,
            loader,
            device,
            bounds_by_domain,
            physics_type=physics_type,
            mu_over_s=config.get("_mu_over_s"),
        )
        return {
            "status": "AVAILABLE",
            "physics_type": physics_type,
            "physics_violation_rate": violation_stats["pooled_violation_rate"],
            "by_multiplier": violation_stats.get("by_multiplier"),
            "by_domain": violation_stats.get("by_domain"),
            "total_evaluations": violation_stats.get("total_evaluations"),
            "total_violations": violation_stats.get("total_violations"),
            "mean_relative_drift": violation_stats.get("mean_relative_drift"),
            "mean_relative_drift_std_units": violation_stats.get("mean_relative_drift_std_units"),
            "drift_space": violation_stats.get("drift_space"),
        }
    if not hasattr(model, "aux_degradation_head"):
        return {"status": "NOT_APPLICABLE"}
    model.eval()
    total_weight = 0
    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "monotonicity": 0.0,
        "monotonic_violation_rate": 0.0,
    }
    predicted_states = []
    target_states = []
    for batch in loader:
        curves = batch["cycle_curve_data"].to(device)
        masks = batch["curve_attn_mask"].to(device)
        _, auxiliary = model(curves, masks, return_aux=True)
        components = physics_loss_fn(
            auxiliary,
            masks,
            curves=curves,
            physics_type=config.get("physics_type", "capacity_health"),
            monotonic_tolerance=config.get("physics_monotonic_tolerance", 0.0),
            monotonic_component_weight=config.get(
                "physics_monotonic_component_weight", 0.25
            ),
            return_components=True,
        )
        batch_weight = int(masks.sum().item())
        total_weight += batch_weight
        for key in totals:
            totals[key] += float(components[key].cpu()) * batch_weight
        capacity = curves[:, :, 2, :].max(dim=-1).values
        target = (1.0 - capacity / capacity[:, :1].clamp_min(0.01)).clamp(-0.5, 1.5)
        observed = masks.bool()
        predicted_states.append(auxiliary[observed].cpu())
        target_states.append(target[observed].cpu())
    predicted = torch.cat(predicted_states).numpy()
    target = torch.cat(target_states).numpy()
    correlation = None
    if np.std(predicted) > 0 and np.std(target) > 0:
        correlation = float(np.corrcoef(predicted, target)[0, 1])
    return {
        "status": "AVAILABLE",
        **{key: value / max(total_weight, 1) for key, value in totals.items()},
        "degradation_state_target_correlation": correlation,
        "predicted_state_mean": float(np.mean(predicted)),
        "predicted_state_std": float(np.std(predicted)),
        "target_state_mean": float(np.mean(target)),
        "target_state_std": float(np.std(target)),
        "observed_cycle_count": total_weight,
    }


def write_outputs(
    experiment_id,
    proposal_path,
    source_path,
    model,
    metrics,
    diagnostics,
    result,
    validation_predictions,
):
    final_dir = ROOT / "experiments" / experiment_id
    temp_dir = ROOT / "experiments" / f".{experiment_id}.tmp"
    temp_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_path, temp_dir / "train.py")
    proposal = json.loads(proposal_path.read_text())
    diagnostics["summary"] = summarize_diagnostics(diagnostics)
    record = {"proposal": proposal, "metrics": metrics, "result": result}
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": proposal["model"],
            "training": proposal["training"],
            "metrics": metrics,
            "result": result,
        },
        temp_dir / "best_checkpoint.pt",
    )
    for name, value in (("record.json", record), ("diagnostics.json", diagnostics)):
        with (temp_dir / name).open("w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
    np.savez_compressed(
        temp_dir / "validation_predictions.npz", **validation_predictions
    )
    os.replace(temp_dir, final_dir)

    with (ROOT / "results.tsv").open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                experiment_id,
                proposal["phase"],
                result["type"],
                proposal["model"]["display_name"],
                proposal["reference_experiment"] or "NA",
                result["seed"],
                metrics["mape"],
                metrics["acc15"],
                result["best_epoch"],
                result["completed_epochs"],
                result["num_params"],
                diagnostics["summary"]["overall_gradient_health"],
                RUNNER_VERSION,
                result["steps_per_sec"],
                "VALID",
                "",
                "Awaiting reviewed conclusion.",
                "Review this run before creating another proposal.",
            ]
        )


def run(experiment_id, proposal_path):
    validate(experiment_id, proposal_path)
    proposal = json.loads(proposal_path.read_text())
    config = proposal["training"]
    seed_everything(config["seed"])
    device = choose_device()
    wall_started = time.perf_counter()
    cache_path = None
    if config.get("input_normalization", "nominal") == "soc_window":  # amendment A6
        cache_path = json.loads((Path(__file__).resolve().parent / "protocol_lock.json").read_text())["cache_v2_path"]
    elif config.get("input_normalization", "nominal") == "soc_window_meta":  # amendment A7
        cache_path = json.loads((Path(__file__).resolve().parent / "protocol_lock.json").read_text())["cache_v3_path"]
    train_loader, val_loader, scaler = prepare.load_train_val_data(
        config["batch_size"], config["seed"], cache_path=cache_path
    )
    config["_mu_over_s"] = float(scaler.mean) / float(scaler.scale)  # EOL-space drift diagnostic
    required_device = config.get("required_device")
    if required_device and device.type != required_device:
        raise RuntimeError(
            f"Required device {required_device!r}, but selected {device.type!r}"
        )
    model = build_model(proposal.get("model", {})).to(device)
    if hasattr(model, "set_reference_bank"):
        model.set_reference_bank(train_loader.dataset)
    if config.get("physics_type") in {"degradation_law", "degradation_law2"} and config.get("physics_loss_weight", 0.0) > 0:
        attach_law_head(model, 4 if config["physics_type"] == "degradation_law2" else 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler_name = config.get("scheduler", "none")
    steps_per_epoch = len(train_loader)
    diagnostic_epochs = {
        1,
        config["minimum_epochs"],
        max(1, (config["max_epochs"] + 1) // 2),
        config["max_epochs"],
    }
    if scheduler_name == "cosine":
        total_steps = config.get(
            "scheduler_total_steps", config["max_epochs"] * steps_per_epoch
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.get("min_lr", 1e-6)
        )
    elif scheduler_name == "warmup_cosine":  # amendment A11
        base_lr = float(config["learning_rate"])
        peak_lr = float(config["peak_learning_rate"])
        min_lr = float(config.get("min_lr", 1e-6))
        total_steps = int(config.get("scheduler_total_steps", config["max_epochs"] * steps_per_epoch))
        warmup_steps = max(1, int(round(float(config.get("warmup_epochs", 1)) * steps_per_epoch)))

        def _lr_at(step):
            if step < warmup_steps:
                return base_lr + (peak_lr - base_lr) * step / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _lr_at(step) / base_lr)
        print(f"[scheduler] warmup_cosine base {base_lr:g} -> peak {peak_lr:g} over {warmup_steps} steps, cosine to {min_lr:g} at step {total_steps}", flush=True)
    else:
        scheduler = None
    activations = ActivationCollector(model)
    diagnostics = {"samples": [], "epoch_history": [], "device": str(device)}
    compute_seconds = 0.0
    validation_seconds = 0.0
    steps = 0
    best = None
    best_validation_mape = None
    stale_epochs = 0
    stop_reason = "max_epochs"
    steps_per_epoch = len(train_loader)
    completed_epochs = 0
    last_epoch = 0
    last_epoch_steps = 0
    peak_device_memory_bytes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    target_transform = config.get("target_transform", "standard")
    if target_transform == "log":
        with torch.no_grad():
            if hasattr(model, "head") and isinstance(model.head, nn.Sequential) and isinstance(model.head[-1], nn.Linear):
                model.head[-1].bias.fill_(math.log(max(scaler.mean, 1.0)))
            elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
                model.head.bias.fill_(math.log(max(scaler.mean, 1.0)))

    physics_weight = config.get("physics_loss_weight", 0.0)
    physics_type = config.get("physics_type", "monotonicity")
    bounds_by_domain = None
    alpha_max_by_domain = None
    physics_rng = None
    lambda_p5 = None
    law_targets = None
    law_fn = degradation_law2_loss if physics_type == "degradation_law2" else degradation_law_loss
    tw_rng = torch.Generator().manual_seed(config["seed"] + 20000)
    tw_cal_rng = torch.Generator().manual_seed(config["seed"] + 30000)
    if physics_type == "timewarp_aug":
        aux_cal = lambda model_, pr_, b_: timewarp_aug_loss(model_, b_, scaler, config, tw_cal_rng, device, target_transform)
        aux_train = lambda model_, pr_, b_: timewarp_aug_loss(model_, b_, scaler, config, tw_rng, device, target_transform)
    else:
        aux_cal = aux_train = lambda model_, pr_, b_: law_fn(model_, pr_, list(b_["cell_id"]), law_targets, scaler, target_transform)
    if physics_weight > 0 and physics_type in {"degradation_law", "degradation_law2", "timewarp_aug"}:
        if physics_type != "timewarp_aug":
            law_targets = load_law_targets()
            print(f"[physics] degradation_law: {len(law_targets)} train trajectories, K={LAW_K}", flush=True)
        gen = getattr(train_loader, "generator", None)
        saved_gen_state = gen.get_state() if gen is not None else None
        cal_batches = []
        for b_ in train_loader:
            cal_batches.append(b_)
            if len(cal_batches) == 3:
                break
        if gen is not None and saved_gen_state is not None:
            gen.set_state(saved_gen_state)
        params = [p_ for p_ in model.parameters() if p_.requires_grad]
        ratios, ratios_p5 = [], []
        gnorm = lambda gs: max(math.sqrt(sum(float((g.detach().cpu().double() ** 2).sum()) for g in gs if g is not None)), 1e-12)
        for b_ in cal_batches:
            c_ = b_["cycle_curve_data"].to(device); m_ = b_["curve_attn_mask"].to(device)
            pr_ = model(c_, m_)
            tg_ = torch.log(b_["raw_eol"].to(device).clamp_min(1.0)) if target_transform == "log" else b_["label"].to(device)
            lm_ = compute_main_loss(pr_, tg_, b_["raw_eol"].to(device), scaler, target_transform, config)
            ll_, _ = aux_cal(model, pr_, b_)
            gm_ = torch.autograd.grad(lm_, params, retain_graph=True, allow_unused=True)
            gl_ = torch.autograd.grad(ll_, params, retain_graph=True, allow_unused=True)
            ratios.append(gnorm(gm_) / gnorm(gl_))
            if config.get("physics_horizon_weight", 0.0) > 0:
                pe_ = pr_ * scaler.scale + scaler.mean
                lp_ = torch.mean(torch.abs(b_["prefix_length"].to(device=device, dtype=pr_.dtype) - pe_) / scaler.scale)
                ratios_p5.append(gnorm(gm_) / gnorm(torch.autograd.grad(lp_, params, allow_unused=True)))
        model.zero_grad(set_to_none=True)
        rho = float(config.get("physics_rho", 0.25))
        physics_weight = rho * float(np.median(ratios))
        lambda_p5 = rho * float(np.median(ratios_p5)) if ratios_p5 else None
        print(f"[physics] Calibrated lambda = {physics_weight:.6e} (rho={rho}, batch_ratios={[round(r, 4) for r in ratios]})", flush=True)
        if lambda_p5 is not None:
            print(f"[physics] Calibrated lambda_p5 = {lambda_p5:.6e}", flush=True)
        diagnostics["calibrated_physics_lambda"] = physics_weight
        diagnostics["physics_calibration"] = {"lambda": physics_weight, "lambda_p5": lambda_p5, "rho": rho,
                                              "batch_ratios": ratios, "batch_ratios_p5": ratios_p5,
                                              "law_cells": len(law_targets) if law_targets else 0, "K": LAW_K}
    if physics_weight > 0 and physics_type in WHATIF_PHYSICS_TYPES:
        bounds_by_domain = compute_bounds_per_domain(train_loader.dataset.cells, physics_type)
        k_by_domain = compute_fade_slope_per_domain(train_loader.dataset.cells) if physics_type == "whatif_fade_p0b" else None
        if k_by_domain is not None:
            k_by_domain["__tau__"] = float(config.get("physics_p0b_tau", 0.5))
            config["_p0b_k_by_domain"] = k_by_domain
        if k_by_domain is not None:
            print(f"[physics] P0b k by domain: " + ", ".join(f"{d}={v:.2f}" for d, v in sorted(k_by_domain.items())), flush=True)
        alpha_max_by_domain = bounds_by_domain
        physics_rng = torch.Generator(device="cpu").manual_seed(config["seed"] + 10000)

        gen = getattr(train_loader, "generator", None)
        saved_gen_state = gen.get_state() if gen is not None else None
        cal_batches = []
        for b in train_loader:
            cal_batches.append(b)
            if len(cal_batches) == 3:
                break
        if gen is not None and saved_gen_state is not None:
            gen.set_state(saved_gen_state)

        calibrated_lambda, cal_meta = calibrate_whatif_lambda(
            model,
            cal_batches,
            bounds_by_domain,
            physics_rng,
            config,
            device,
            scaler,
            target_transform,
            physics_type=physics_type,
        )
        physics_weight = calibrated_lambda
        lambda_p5 = cal_meta.get("lambda_p5")
        diagnostics["calibrated_physics_lambda"] = calibrated_lambda
        diagnostics["physics_calibration"] = cal_meta

    rarity_table = None
    if config.get("rarity_alpha", 0.0) > 0:
        rarity_table = build_rarity_weights(train_loader.dataset.cells, alpha=float(config["rarity_alpha"]),
                                            max_w=float(config.get("rarity_max_w", 5.0)), bw=float(config.get("rarity_bw", 0.25)))
        diagnostics["rarity_weighting"] = rarity_table["summary"]
        print(f"[rarity] weights at EOL: {rarity_table['summary']['w_at_eol']}", flush=True)
    lambda_phys2 = None
    phys2_type = config.get("physics2_type")
    phys2_rng = torch.Generator().manual_seed(config["seed"] + 40000)
    if phys2_type:
        if phys2_type in WHATIF_PHYSICS_TYPES:
            config["_phys2_bounds"] = compute_bounds_per_domain(train_loader.dataset.cells, phys2_type)
        gen_ = getattr(train_loader, "generator", None); st_ = gen_.get_state() if gen_ is not None else None
        cal_ = []
        for k_, b_ in enumerate(train_loader):
            if phys2_type == "budget_bound":  # rare constraint: calibrate on batches that contain active samples
                _, af_ = budget_bound_loss(torch.zeros(b_["curve_attn_mask"].size(0)), b_["cycle_curve_data"], b_["curve_attn_mask"], scaler,
                                           gate=float(config.get("budget_gate", 0.05)))
                if af_ <= 0:
                    if k_ > 400:
                        break
                    continue
            cal_.append(b_)
            if len(cal_) == 3:
                break
        if gen_ is not None and st_ is not None:
            gen_.set_state(st_)
        params_ = [p_ for p_ in model.parameters() if p_.requires_grad]
        gn_ = lambda gs: max(math.sqrt(sum(float((g.detach().cpu().double() ** 2).sum()) for g in gs if g is not None)), 1e-12)
        cal_rng = torch.Generator().manual_seed(config["seed"] + 50000)
        rat_ = []
        for b_ in cal_:
            c_ = b_["cycle_curve_data"].to(device); m_ = b_["curve_attn_mask"].to(device)
            pr_ = model(c_, m_)
            tg_ = torch.log(b_["raw_eol"].to(device).clamp_min(1.0)) if target_transform == "log" else b_["label"].to(device)
            lm_ = compute_main_loss(pr_, tg_, b_["raw_eol"].to(device), scaler, target_transform, config)
            if phys2_type == "budget_bound":  # unsigned calibration on the active samples (loss.md §5)
                l2_ = budget_bound_loss(pr_, c_, m_, scaler, target_transform, gate=float(config.get("budget_gate", 0.05)), slack=0.0)[0]
            else:
                l2_ = secondary_physics_loss(phys2_type, model, pr_, b_, c_, m_, scaler, config, cal_rng, device, target_transform,
                                             unsigned=True)  # loss.md §5.2: unsigned calibration for one-sided terms
            gm_ = torch.autograd.grad(lm_, params_, retain_graph=True, allow_unused=True)
            g2_ = torch.autograd.grad(l2_, params_, allow_unused=True)
            if gn_(g2_) > 1e-11:
                rat_.append(gn_(gm_) / gn_(g2_))
        model.zero_grad(set_to_none=True)
        rho2 = float(config.get("physics2_rho", 0.25))
        lambda_phys2 = rho2 * float(np.median(rat_)) if rat_ else 0.0
        print(f"[physics2] {phys2_type}: lambda2 = {lambda_phys2:.6e} (rho2={rho2}, ratios={[round(r, 4) for r in rat_]})", flush=True)
        diagnostics["physics2_calibration"] = {"type": phys2_type, "lambda2": lambda_phys2, "rho2": rho2, "batch_ratios": rat_}
    ema_decay = float(config.get("ema_decay", 0.0))
    ema_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()} if ema_decay > 0 else None
    if ema_state is not None:
        print(f"[ema] weight EMA decay={ema_decay}; validation/checkpoint use EMA weights", flush=True)
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        epoch_steps = 0
        epoch_losses = []
        epoch_main_losses = []
        epoch_physics_losses = []
        epoch_weighted_physics_losses = []
        for batch_index, batch in enumerate(train_loader):
            curves = batch["cycle_curve_data"].to(device)
            masks = batch["curve_attn_mask"].to(device)
            labels = batch["label"].to(device)
            raw_eol = batch["raw_eol"].to(device)
            sampled = (batch_index == 0 and epoch in diagnostic_epochs) or (
                physics_weight > 0 and (steps + 1) % 250 == 0
            )
            before = parameter_snapshot(model) if sampled else None
            activations.enabled = sampled
            activations.current = {}
            if hasattr(model, "capture_attention"):
                model.capture_attention = sampled
            sync(device)
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            physics_components = None
            if physics_weight > 0:
                if physics_type in {"degradation_law", "degradation_law2", "timewarp_aug"}:
                    predictions = model(curves, masks)
                    phys_loss, physics_components = aux_train(model, predictions, batch)
                    weighted_phys = physics_weight * phys_loss
                    if config.get("physics_horizon_weight", 0.0) > 0 and lambda_p5 is not None:
                        weighted_phys = weighted_phys + lambda_p5 * compute_p5_horizon_loss(
                            predictions, batch["prefix_length"], scaler, target_transform)
                elif physics_type in WHATIF_PHYSICS_TYPES:
                    predictions = model(curves, masks)
                    phys_loss, physics_components = whatif_loss(
                        model,
                        curves,
                        masks,
                        batch["domain"],
                        bounds_by_domain,
                        physics_rng,
                        physics_type=physics_type,
                        predictions=predictions,
                        prefix_lengths=batch.get("prefix_length"),
                        anchor_gradient=config.get("physics_anchor_gradient", "stop"),
                        k_by_domain=k_by_domain,
                        mu_over_s=(float(scaler.mean) / float(scaler.scale)) if physics_type == "whatif_fade_p0b" else None,
                        drift_log_mu_over_s=((float(scaler.mean) / float(scaler.scale))
                                             if (physics_type == "prefix_drift" and config.get("physics_drift_space", "standard") == "log") else None),
                    )
                    weighted_phys = physics_weight * phys_loss
                    if (
                        config.get("physics_horizon_weight", 0.0) > 0
                        and lambda_p5 is not None
                        and "prefix_length" in batch
                    ):
                        l_p5 = compute_p5_horizon_loss(
                            predictions, batch["prefix_length"], scaler, target_transform
                        )
                        weighted_phys = weighted_phys + lambda_p5 * l_p5
                elif hasattr(model, "aux_degradation_head"):
                    predictions, aux_deg = model(curves, masks, return_aux=True)
                    physics_components = physics_loss_fn(
                        aux_deg,
                        masks,
                        curves=curves,
                        predictions=predictions,
                        physics_type=physics_type,
                        monotonic_tolerance=config.get(
                            "physics_monotonic_tolerance", 0.0
                        ),
                        monotonic_component_weight=config.get(
                            "physics_monotonic_component_weight", 0.25
                        ),
                        return_components=True,
                    )
                    phys_loss = physics_components["total"]
                    weighted_phys = physics_weight * phys_loss
                    if (
                        config.get("physics_horizon_weight", 0.0) > 0
                        and lambda_p5 is not None
                        and "prefix_length" in batch
                    ):
                        l_p5 = compute_p5_horizon_loss(
                            predictions, batch["prefix_length"], scaler, target_transform
                        )
                        weighted_phys = weighted_phys + lambda_p5 * l_p5
                else:
                    predictions = model(curves, masks)
                    phys_loss = None
                    weighted_phys = 0.0
            else:
                predictions = model(curves, masks)
                phys_loss = None
                weighted_phys = 0.0

            if lambda_phys2:
                if phys2_type in WHATIF_PHYSICS_TYPES or phys2_type == "stat_tail_balance":
                    l2, meta2 = secondary_physics_loss(phys2_type, model, predictions, batch, curves, masks, scaler, config, phys2_rng,
                                                       device, target_transform, return_meta=True)
                else:
                    l2, meta2 = secondary_physics_loss(phys2_type, model, predictions, batch, curves, masks, scaler, config, phys2_rng,
                                                       device, target_transform), {}
                weighted_phys = weighted_phys + lambda_phys2 * l2
                if (steps + 1) % 250 == 0:
                    diagnostics.setdefault("physics2_steps", []).append({
                        "step": steps + 1, "type": phys2_type, "raw": float(l2.detach().cpu()),
                        "weighted": float((lambda_phys2 * l2).detach().cpu()),
                        "active_fraction": float(meta2.get("active_fraction", float("nan")))})
            target = torch.log(raw_eol.clamp_min(1.0)) if target_transform == "log" else labels
            main_loss_name = config.get("main_loss", "mse")
            if rarity_table is not None:
                main_loss = rarity_weighted_relative_huber(predictions, raw_eol, scaler, target_transform, config, rarity_table)
            elif config.get("group_dro_eta", 0.0) > 0:
                main_loss = group_dro_relative_huber(predictions, raw_eol, scaler, batch["domain"], config)
            else:
                main_loss = compute_main_loss(
                    predictions, target, raw_eol, scaler, target_transform, config
                )
            consistency_weight = config.get("prefix_consistency_weight", 0.0)
            loss_interaction = None
            physics_interaction = None
            if consistency_weight > 0:
                consistency_loss, consistency_meta = prefix_consistency_loss(
                    predictions, batch["cell_id"]
                )
                weighted_consistency = consistency_weight * consistency_loss
                loss = main_loss + weighted_consistency + weighted_phys
                if sampled:
                    loss_interaction = loss_gradient_interaction(
                        model, main_loss, weighted_consistency
                    )
            else:
                loss = main_loss + weighted_phys
            if sampled and phys_loss is not None:
                physics_interaction = loss_gradient_interaction(
                    model, main_loss, weighted_phys
                )
            if config.get("physics_pcgrad") and phys_loss is not None and torch.is_tensor(weighted_phys):
                # PCGrad (user note R4-E046): if <g_phys, g_main> < 0, remove the conflicting component of g_phys
                params_pc = [p_ for p_ in model.parameters() if p_.requires_grad]
                other = loss - weighted_phys
                g_m = torch.autograd.grad(other, params_pc, retain_graph=True, allow_unused=True)
                g_p = torch.autograd.grad(weighted_phys, params_pc, allow_unused=True)
                g_m = [torch.zeros_like(p_) if g is None else g for g, p_ in zip(g_m, params_pc)]
                g_p = [torch.zeros_like(p_) if g is None else g for g, p_ in zip(g_p, params_pc)]
                dot = sum((a * b).sum() for a, b in zip(g_m, g_p))
                nm2 = sum((a * a).sum() for a in g_m).clamp_min(1e-20)
                coef = torch.clamp(dot / nm2, max=0.0)
                pcgrad_projected = bool(float(dot) < 0)
                for p_, a, b in zip(params_pc, g_m, g_p):
                    p_.grad = a + b - coef * a
                pcgrad_count = globals().get("_PCGRAD_COUNT", [0, 0])
                pcgrad_count[0] += int(pcgrad_projected); pcgrad_count[1] += 1
                globals()["_PCGRAD_COUNT"] = pcgrad_count
            else:
                loss.backward()
            gradients = gradient_snapshot(model) if sampled else None
            optimizer.step()
            if ema_state is not None:  # weight EMA (user-approved 2026-11-03)
                with torch.no_grad():
                    for k_, v_ in model.state_dict().items():
                        if v_.dtype.is_floating_point:
                            ema_state[k_].mul_(ema_decay).add_(v_.detach(), alpha=1.0 - ema_decay)
                        else:
                            ema_state[k_].copy_(v_)
            if scheduler is not None:
                scheduler.step()
            updates = update_snapshot(model, before) if sampled else None
            sync(device)
            compute_seconds += time.perf_counter() - step_started
            steps += 1
            epoch_steps += 1
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_main_losses.append(float(main_loss.detach().cpu()))
            if phys_loss is not None:
                epoch_physics_losses.append(float(phys_loss.detach().cpu()))
                epoch_weighted_physics_losses.append(
                    float(weighted_phys.detach().cpu())
                )
            if device.type == "mps":
                peak_device_memory_bytes = max(
                    peak_device_memory_bytes,
                    int(torch.mps.current_allocated_memory()),
                )
            if not math.isfinite(float(loss.detach().cpu())):
                stop_reason = "nonfinite_loss"
                break
            if sampled:
                sample = {
                        "step": steps,
                        "loss": float(loss.detach().cpu()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "main_loss": {
                            "name": main_loss_name,
                            "raw": float(main_loss.detach().cpu()),
                        },
                        "mask_observed_fraction": float(masks.mean().detach().cpu()),
                        "empty_masks": int((masks.sum(dim=1) == 0).sum().detach().cpu()),
                        "prediction_batch": {
                            "mean": float(predictions.detach().mean().cpu()),
                            "std": float(
                                predictions.detach().std(unbiased=False).cpu()
                            ),
                            "min": float(predictions.detach().min().cpu()),
                            "max": float(predictions.detach().max().cpu()),
                            "target_mean": float(target.detach().mean().cpu()),
                            "target_std": float(
                                target.detach().std(unbiased=False).cpu()
                            ),
                        },
                        "gradients": gradients,
                        "activations": activations.current,
                        "update_ratios": updates,
                    }
                if phys_loss is not None:
                    raw_recon = (
                        physics_components["reconstruction"].detach().cpu()
                        if "reconstruction" in physics_components
                        else torch.tensor(0.0)
                    )
                    raw_mono = (
                        physics_components["monotonicity"].detach().cpu()
                        if "monotonicity" in physics_components
                        else phys_loss.detach().cpu()
                    )
                    mono_viol = (
                        physics_components["monotonic_violation_rate"].detach().cpu()
                        if torch.is_tensor(physics_components.get("monotonic_violation_rate"))
                        else physics_components.get("monotonic_violation_rate", 0.0)
                    )
                    sample["physics_loss"] = {
                        "type": physics_type,
                        "raw_total": float(phys_loss.detach().cpu()),
                        "raw_reconstruction": float(raw_recon),
                        "raw_monotonicity": float(raw_mono),
                        "monotonic_violation_rate": float(mono_viol),
                        "active_fraction": float(physics_components.get("active_fraction", 0.0)),
                        "weight": physics_weight,
                        "weighted": float(weighted_phys.detach().cpu()),
                        "weighted_to_main_loss_ratio": float(
                            weighted_phys.detach().abs().cpu()
                            / main_loss.detach().abs().clamp_min(1e-12).cpu()
                        ),
                        **physics_interaction,
                        "weighted_to_main_gradient_ratio": (
                            physics_interaction[
                                "weighted_auxiliary_gradient_norm"
                            ]
                            / max(
                                physics_interaction["main_gradient_norm"], 1e-12
                            )
                        ),
                    }
                    if steps % 250 == 0:
                        grad_cos_str = (
                            f"{physics_interaction['gradient_cosine']:.4f}"
                            if physics_interaction.get("gradient_cosine") is not None
                            else "NA"
                        )
                        act_frac = physics_components.get("active_fraction", 0.0)
                        print(
                            f"[step={steps}] L_main={main_loss.item():.4f} L_phys={phys_loss.item():.4f} "
                            f"active_frac={act_frac:.3f} grad_cosine={grad_cos_str}",
                            flush=True,
                        )
                if consistency_weight > 0:
                    sample["loss_terms"] = {
                        "main_loss_name": main_loss_name,
                        "raw_main_loss": float(main_loss.detach().cpu()),
                        "raw_prefix_consistency": float(
                            consistency_loss.detach().cpu()
                        ),
                        "weighted_prefix_consistency": float(
                            weighted_consistency.detach().cpu()
                        ),
                        **consistency_meta,
                        **loss_interaction,
                    }
                if hasattr(model, "attention_diagnostics"):
                    sample["attention"] = model.attention_diagnostics()
                if hasattr(model, "gate_diagnostics"):
                    sample["gates"] = model.gate_diagnostics()
                if hasattr(model, "lifecycle_diagnostics"):
                    sample["lifecycle"] = model.lifecycle_diagnostics()
                diagnostics["samples"].append(sample)
        last_epoch = epoch
        last_epoch_steps = epoch_steps
        epoch_complete = epoch_steps == steps_per_epoch
        if epoch_complete:
            completed_epochs = epoch
        epoch_progress = completed_epochs + (
            0.0 if epoch_complete else epoch_steps / steps_per_epoch
        )
        if ema_state is not None:
            train_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema_state)
        metrics, val_seconds = evaluate(
            model, val_loader, device, scaler, target_transform=target_transform
        )
        eval_state = copy.deepcopy(model.state_dict()) if ema_state is not None else None
        if ema_state is not None:
            model.load_state_dict(train_state)
        validation_seconds += val_seconds
        diagnostics["epoch_history"].append(
            {
                "epoch": epoch,
                "complete": epoch_complete,
                "steps": epoch_steps,
                "train_loss_mean": float(np.mean(epoch_losses)),
                "train_loss_std": float(np.std(epoch_losses)),
                "train_loss_min": float(np.min(epoch_losses)),
                "train_loss_max": float(np.max(epoch_losses)),
                "loss_balance": {
                    "main_loss_mean": float(np.mean(epoch_main_losses)),
                    "raw_physics_loss_mean": (
                        float(np.mean(epoch_physics_losses))
                        if epoch_physics_losses
                        else None
                    ),
                    "weighted_physics_loss_mean": (
                        float(np.mean(epoch_weighted_physics_losses))
                        if epoch_weighted_physics_losses
                        else None
                    ),
                    "weighted_physics_to_main_ratio": (
                        float(np.mean(epoch_weighted_physics_losses))
                        / max(float(np.mean(epoch_main_losses)), 1e-12)
                        if epoch_weighted_physics_losses
                        else None
                    ),
                },
                "validation": metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "cumulative_training_compute_seconds": round(compute_seconds, 3),
            }
        )
        print(
            f"epoch={epoch} epoch_steps={epoch_steps}/{steps_per_epoch} "
            f"epoch_progress={epoch_progress:.3f} steps={steps} "
            f"train_compute={compute_seconds:.2f}s "
            f"val_mape={metrics['mape']:.6f} val_acc15={metrics['acc15']:.6f}",
            flush=True,
        )
        checkpoint_tolerance = config.get("checkpoint_mape_tolerance", 0.0)
        improved_minimum = (
            best_validation_mape is None
            or metrics["mape"] < best_validation_mape
        )
        if improved_minimum:
            best_validation_mape = metrics["mape"]
            stale_epochs = 0
        elif epoch >= config["minimum_epochs"]:
            stale_epochs += 1

        candidate_is_better = checkpoint_candidate_is_better(
            metrics,
            best["metrics"] if best is not None else None,
            best_validation_mape,
            checkpoint_tolerance,
        )
        if candidate_is_better:
            best = {
                "epoch": epoch,
                "epoch_steps": epoch_steps,
                "epoch_complete": epoch_complete,
                "metrics": metrics,
                "weights": eval_state if eval_state is not None else copy.deepcopy(model.state_dict()),
            }
        if stop_reason == "nonfinite_loss":
            break
        if epoch >= config["minimum_epochs"] and stale_epochs >= config["patience"]:
            stop_reason = "early_stopping"
            break
        if (
            epoch_complete
            and epoch >= config["minimum_epochs"]
            and compute_seconds >= proposal["estimated_budget"]["training_seconds"]
        ):
            stop_reason = "training_compute_safety_limit"
            break

    activations.close()
    if best is None:
        raise RuntimeError("No valid validation checkpoint")
    model.load_state_dict(best["weights"])
    final_metrics, val_seconds, validation_predictions = evaluate(
        model,
        val_loader,
        device,
        scaler,
        target_transform=target_transform,
        include_groups=True,
        return_predictions=True,
    )
    validation_seconds += val_seconds
    activations.enabled = False
    diagnostics["padding_invariance"] = padding_invariance_check(
        model, val_loader, device
    )
    if bounds_by_domain is None and physics_type in WHATIF_PHYSICS_TYPES:
        bounds_by_domain = compute_bounds_per_domain(train_loader.dataset.cells, physics_type)
        alpha_max_by_domain = bounds_by_domain

    if config.get("physics_pcgrad"):
        pc = globals().get("_PCGRAD_COUNT", [0, 0])
        diagnostics["pcgrad"] = {"projected_steps": pc[0], "total_steps": pc[1], "projected_fraction": pc[0] / max(pc[1], 1)}
        print(f"[physics] PCGrad projected {pc[0]}/{pc[1]} steps", flush=True)
    diagnostics["physics_validation"] = evaluate_physics_health(
        model, val_loader, device, config, bounds_by_domain=bounds_by_domain
    )
    if phys2_type in WHATIF_PHYSICS_TYPES and phys2_type != "prefix_drift":
        diagnostics["physics2_validation"] = evaluate_whatif_violation_rate(
            model, val_loader, device, config.get("_phys2_bounds"), physics_type=phys2_type)
        print(f"[physics2] held-out {phys2_type} violation rate = {diagnostics['physics2_validation'].get('pooled_violation_rate')}", flush=True)
    if (
        diagnostics["physics_validation"].get("status") == "AVAILABLE"
        and "physics_violation_rate" in diagnostics["physics_validation"]
    ):
        final_metrics["physics_violation_rate"] = diagnostics["physics_validation"]["physics_violation_rate"]
    diagnostics["checkpoint_selection"] = {
        "rule": "max_acc15_within_mape_tolerance_then_lower_mape_then_earlier_epoch",
        "mape_tolerance": config.get("checkpoint_mape_tolerance", 0.0),
        "minimum_validation_mape": best_validation_mape,
        "selected_epoch": best["epoch"],
        "selected_mape": final_metrics["mape"],
        "selected_acc15": final_metrics["acc15"],
    }
    diagnostics["nonfinite_detected"] = any(
        any(block["nonfinite_gradients"] for block in sample["gradients"].values())
        or any(block["nonfinite"] for block in sample["activations"].values())
        for sample in diagnostics["samples"]
    )
    status = (
        "invalid"
        if stop_reason == "nonfinite_loss" or diagnostics["nonfinite_detected"]
        else "pending_analysis"
    )
    state = json.loads((ROOT / "state.json").read_text())
    result = {
        "benchmark_round": state["benchmark_round"],
        "protocol_fingerprint": json.loads((ROOT / "protocol_lock.json").read_text())["protocol_fingerprint"],
        "type": proposal["type"],
        "seed": config["seed"],
        "training_compute_seconds": round(compute_seconds, 3),
        "validation_seconds": round(validation_seconds, 3),
        "wall_seconds": round(time.perf_counter() - wall_started, 3),
        "num_steps": steps,
        "runner_version": RUNNER_VERSION,
        "steps_per_sec": round(steps / max(compute_seconds, 1e-12), 6),
        "num_params": sum(parameter.numel() for parameter in model.parameters()),
        "model_bytes_fp32": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else peak_device_memory_bytes or None
        ),
        "best_epoch": best["epoch"],
        "minimum_validation_mape": best_validation_mape,
        "checkpoint_mape_tolerance": config.get(
            "checkpoint_mape_tolerance", 0.0
        ),
        "best_epoch_complete": best["epoch_complete"],
        "best_epoch_steps": best["epoch_steps"],
        "steps_per_epoch": steps_per_epoch,
        "completed_epochs": completed_epochs,
        "last_epoch": last_epoch,
        "last_epoch_steps": last_epoch_steps,
        "epoch_progress": round(
            completed_epochs
            + (0.0 if last_epoch_steps == steps_per_epoch else last_epoch_steps / steps_per_epoch),
            6,
        ),
        "stop_reason": stop_reason,
        "status": status,
        "description": proposal["modification"],
        "device": str(device),
    }
    write_outputs(
        experiment_id,
        proposal_path,
        Path(__file__),
        model,
        final_metrics,
        diagnostics,
        result,
        validation_predictions,
    )
    print(json.dumps({"metrics": final_metrics, "result": result}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=False)
    parser.add_argument("--proposal", required=False, type=Path)
    parser.add_argument(
        "--eval-violation",
        type=str,
        default=None,
        help="Evaluate what-if physics violation rate for a saved experiment ID",
    )
    parser.add_argument(
        "--physics-type",
        type=str,
        default=None,
        help="Physics type to evaluate violation rate for (defaults to config or whatif_fade_p0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for evaluation (e.g. cpu, mps)",
    )
    args = parser.parse_args()

    if args.eval_violation:
        dev = torch.device(args.device) if args.device else None
        evaluate_checkpoint_violation(args.eval_violation, device=dev, physics_type=args.physics_type)
        return

    if not args.experiment_id or not args.proposal:
        parser.error(
            "Both --experiment-id and --proposal are required unless --eval-violation is used."
        )
    log_path = ROOT / f".{args.experiment_id}.train.log"
    live_log_path = ROOT / "live.log"
    succeeded = False
    with log_path.open("w") as experiment_log, live_log_path.open("a") as live_log:
        stdout = Tee(sys.stdout, experiment_log, live_log)
        stderr = Tee(sys.stderr, experiment_log, live_log)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(
                f"\n===== {args.experiment_id} START {time.strftime('%Y-%m-%d %H:%M:%S')} =====",
                flush=True,
            )
            try:
                run(args.experiment_id, args.proposal)
                succeeded = True
                print(f"===== {args.experiment_id} COMPLETE =====", flush=True)
            except Exception:
                traceback.print_exc()
                print(f"===== {args.experiment_id} FAILED =====", flush=True)
                raise
    if succeeded and log_path.exists():
        final_log_path = ROOT / "experiments" / args.experiment_id / "train.log"
        os.replace(log_path, final_log_path)


if __name__ == "__main__":
    main()
