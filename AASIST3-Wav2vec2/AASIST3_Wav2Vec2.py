"""
AASIST3_Wav2Vec2.py
====================
Replaces the SincConv (RawFrontend) + ResNet encoder of AASIST3_Raw with a
pre-trained wav2vec 2.0 feature extractor, keeping the full KAN-based graph
attention backbone (GraphFormation -> MultiBranchArchitecture -> OutputHead)
intact.

Architecture:
    Raw Waveform  (B, 1, T)  or  (B, T)
         |
    Wav2Vec2Frontend
         - facebook/wav2vec2-base  ->  hidden_size = 768,  95 M params
         - facebook/wav2vec2-large -> hidden_size = 1024, 317 M params  [upgrade]
         - CNN feature extractor always frozen (stable low-level filters)
         - Transformer layers: frozen by default, top-N unfrozen for fine-tuning
         - Weighted-sum over all hidden layers (learnable softmax weights)
         - LayerNorm + Dropout on aggregated representation
         |
    (B, T_ds, hidden_size)   T_ds ~ T / 320  for 16 kHz audio
         |
    Wav2Vec2ProjectionHead
         - Conv1D  hidden_size -> 256  (kernel 3)
         - BatchNorm1d + GELU
         - Conv1D  256 -> 256  (kernel 3, dilation 2)
         - Residual shortcut (1x1 conv)
         |
    (B, 256, T_ds)
         |
    GraphFormation   (temporal + spatial KAN-GAL, 50 nodes each)
         |
    MultiBranchArchitecture (4 branches, KAN_HS_GAL)
         |
    AASIST3OutputHead (KAN -> 2 classes)

RTX 5060 Ti 16 GB recommended config
--------------------------------------
  Model       : wav2vec2-base  (95 M params,  ~4 GB VRAM at bs=24)
  Batch size  : 24  (no grad accumulation needed)
  AMP dtype   : BF16 (Blackwell-native, more stable than FP16)
  compile     : --use_compile  (torch.compile, ~20-30% speedup)
  Graph nodes : 50 temporal + 50 spatial  (vs. 25 in base config)
  Training    : single-stage, epochs 1-30. Top-6 transformer layers
                trainable throughout (--unfreeze_top_n), backbone LR 1e-4,
                frontend LR 5e-6. (A prior two-stage freeze/unfreeze design
                was removed — it rebuilt the optimizer partway through
                without actually unlocking any new parameters, which only
                discarded Adam's momentum state and destabilized training.)

  Upgrade to wav2vec2-large (317 M params): reduce batch_size to 12.
"""

import math
import os
import json
import gc
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Global precision settings (RTX 5060 Ti / Blackwell)
#   TF32: ~3x faster float32 matmuls on tensor cores, negligible precision loss
#   cuDNN benchmark: picks fastest conv algorithm for fixed input sizes
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision('high')   # enables TF32 tensor cores
torch.backends.cudnn.benchmark = True         # auto-tune conv kernels

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd

# ---------------------------------------------------------------------------
# HuggingFace wav2vec2
# ---------------------------------------------------------------------------
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    raise ImportError(
        "transformers is required. Install with:\n"
        "  pip install transformers"
    )

# ---------------------------------------------------------------------------
# Re-use the KAN / graph layers from AASIST3.py
# ---------------------------------------------------------------------------
import sys
_aasist_dir = os.path.dirname(os.path.abspath(__file__))
if _aasist_dir not in sys.path:
    sys.path.insert(0, _aasist_dir)

from AASIST3 import (
    KANLayer,
    KAN_GAL,
    KAN_GraphPool,
    KAN_HS_GAL,
    GraphFormation,
    BranchModule,
    MultiBranchArchitecture,
    AASIST3OutputHead,
    AudioProcessor,
    RawASV5Dataset,
    MetricsCalculation,
    FocalLoss,
    TrainAASIST3,
    count_parameters,
    print_training_summary,
    print_model_summary,
)


# ===========================================================================
# Wav2Vec2 Frontend
# ===========================================================================

class Wav2Vec2Frontend(nn.Module):
    """
    Drop-in frontend that replaces RawFrontend + AASIST3Encoder.

    Outputs (B, encoder_dim, T_ds) — same shape convention as AASIST3Encoder —
    so GraphFormation can be used without any modification.

    Parameters
    ----------
    model_name : str
        HuggingFace model id.  Options:
            'facebook/wav2vec2-base'          →  768-dim,  95 M params
            'facebook/wav2vec2-large'         → 1024-dim, 317 M params
            'facebook/wav2vec2-large-xlsr-53' → 1024-dim, 317 M params
    freeze_feature_extractor : bool
        Always freeze the CNN feature extractor (recommended — these low-level
        filters are stable and freezing saves ~2 M params).
    freeze_transformer_layers : bool
        Freeze ALL transformer layers on init.  Use unfreeze_top_n_transformer
        to selectively re-enable the top N layers for fine-tuning.
    unfreeze_top_n_transformer : int
        Number of transformer layers (counted from the top) to keep trainable
        even when freeze_transformer_layers=True.  0 means fully frozen.
    dropout : float
        Dropout applied to the weighted-sum output before the projection head.
    output_dim : int
        Channel dimension fed into GraphFormation.  Default 256 matches the
        AASIST3Encoder output that GraphFormation expects.
    """

    # Supported model configs: (hidden_size, num_hidden_layers)
    _MODEL_CONFIGS = {
        "facebook/wav2vec2-base":          (768,  12),
        "facebook/wav2vec2-large":         (1024, 24),
        "facebook/wav2vec2-large-xlsr-53": (1024, 24),
        "facebook/wav2vec2-xls-r-300m":    (1024, 24),
    }

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: bool = True,
        unfreeze_top_n_transformer: int = 4,
        dropout: float = 0.1,
        output_dim: int = 256,
    ):
        super().__init__()

        self.model_name = model_name
        self.output_dim = output_dim

        # ── Load pre-trained wav2vec2 ──────────────────────────────────────
        print(f"[Wav2Vec2Frontend] Loading '{model_name}' ...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=True,   # we want all layers
        )

        # Infer hidden_size and num_layers from the loaded config
        cfg = self.wav2vec2.config
        self.hidden_size: int = cfg.hidden_size          # e.g. 768
        self.num_hidden_layers: int = cfg.num_hidden_layers  # e.g. 12

        # ── Freeze CNN feature extractor (always recommended) ─────────────
        if freeze_feature_extractor:
            for p in self.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
            # Also freeze the feature projection (linear after CNN)
            for p in self.wav2vec2.feature_projection.parameters():
                p.requires_grad = False
            print(f"[Wav2Vec2Frontend] CNN feature extractor frozen.")

        # ── Freeze / selectively unfreeze transformer layers ─────────────
        if freeze_transformer_layers:
            for p in self.wav2vec2.encoder.parameters():
                p.requires_grad = False

            # Unfreeze the top N transformer layers for fine-tuning
            if unfreeze_top_n_transformer > 0:
                total = self.num_hidden_layers
                unfreeze_from = total - unfreeze_top_n_transformer
                layers = self.wav2vec2.encoder.layers
                for i, layer in enumerate(layers):
                    if i >= unfreeze_from:
                        for p in layer.parameters():
                            p.requires_grad = True
                print(
                    f"[Wav2Vec2Frontend] Transformer frozen except top "
                    f"{unfreeze_top_n_transformer} layers "
                    f"(layers {unfreeze_from}–{total-1})."
                )
            else:
                print("[Wav2Vec2Frontend] Transformer fully frozen.")

        # ── Weighted sum over all hidden layers (incl. layer 0 = embedding) ─
        # +1 because output_hidden_states includes the embedding output
        self.num_layers = self.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers))

        # ── Post-aggregation normalisation ────────────────────────────────
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        # ── Projection: hidden_size → output_dim ─────────────────────────
        self.projection = Wav2Vec2ProjectionHead(
            in_dim=self.hidden_size,
            out_dim=output_dim,
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(
            f"[Wav2Vec2Frontend] Trainable params: "
            f"{trainable:,} / {total:,} total"
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, T)  or  (B, T)   — raw waveform, 16 kHz, peak-normalised

        Returns
        -------
        (B, output_dim, T_ds)   — T_ds ≈ T / 320
        """
        # Normalise shape to (B, T)
        if x.dim() == 3:
            x = x.squeeze(1)   # (B, 1, T) → (B, T)

        # wav2vec2 forward — returns (last_hidden_state, ..., hidden_states)
        outputs = self.wav2vec2(
            input_values=x,
            output_hidden_states=True,
        )

        # hidden_states: tuple of (B, T_ds, hidden_size),  len = num_layers
        hidden_states = outputs.hidden_states  # tuple[Tensor]

        # Stack → (num_layers, B, T_ds, hidden_size)
        stacked = torch.stack(hidden_states, dim=0)

        # Learnable weighted sum  → (B, T_ds, hidden_size)
        weights = F.softmax(self.layer_weights, dim=0)          # (num_layers,)
        # broadcast: (num_layers,1,1,1) * (num_layers,B,T,H) → sum over dim0
        aggregated = (stacked * weights[:, None, None, None]).sum(dim=0)

        aggregated = self.layer_norm(aggregated)
        aggregated = self.dropout(aggregated)

        # (B, T_ds, hidden_size) → (B, hidden_size, T_ds)
        aggregated = aggregated.transpose(1, 2)

        # Project to output_dim:  (B, output_dim, T_ds)
        out = self.projection(aggregated)
        return out

    # ------------------------------------------------------------------
    def unfreeze_top_n_layers(self, n: int):
        """
        Utility: unfreeze the top-N transformer layers at any point during
        training (e.g. after the graph head has warmed up).
        """
        total = self.num_hidden_layers
        unfreeze_from = max(0, total - n)
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = (i >= unfreeze_from)
        print(f"[Wav2Vec2Frontend] Unfroze top {n} transformer layers.")

    def freeze_all_transformer(self):
        for p in self.wav2vec2.encoder.parameters():
            p.requires_grad = False
        print("[Wav2Vec2Frontend] All transformer layers frozen.")


class Wav2Vec2ProjectionHead(nn.Module):
    """
    Two-layer dilated Conv1D block that maps wav2vec2's hidden_size to the
    target output_dim expected by GraphFormation, with a residual shortcut.

    Input / output shape: (B, C, T_ds)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(out_dim)

        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3,
                               padding=2, dilation=2)
        self.bn2   = nn.BatchNorm1d(out_dim)

        # Residual shortcut (1×1 conv to match channels)
        self.shortcut = nn.Conv1d(in_dim, out_dim, kernel_size=1)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim, T_ds)
        residual = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


# ===========================================================================
# Full AASIST3 + Wav2Vec2 model
# ===========================================================================

class AASIST3_Wav2Vec2(nn.Module):
    """
    AASIST3 with a wav2vec 2.0 frontend.

    The graph backbone (GraphFormation + MultiBranchArchitecture +
    AASIST3OutputHead) is identical to AASIST3_Raw — only the frontend is
    swapped.

    Parameters
    ----------
    wav2vec2_model_name : str
        HuggingFace model id for the wav2vec2 encoder.
    freeze_feature_extractor : bool
        Freeze CNN feature extractor of wav2vec2.
    freeze_transformer_layers : bool
        Freeze transformer layers of wav2vec2 on init.
    unfreeze_top_n_transformer : int
        How many top transformer layers to keep trainable.
    frontend_dropout : float
        Dropout in the wav2vec2 weighted-sum output.
    encoder_out_dim : int
        Projection output dim; must match what GraphFormation expects.
    num_temporal_nodes : int
        Number of temporal graph nodes after initial pooling.
    num_spatial_nodes : int
        Number of spatial graph nodes after initial pooling.
    temporal_dim : int
        Feature dimension for temporal graph nodes.
    spatial_dim : int
        Feature dimension for spatial graph nodes.
    stack_dim : int
        Dimension of the stack (memory) node.
    num_branches : int
        Number of cascaded BranchModules.
    pool_ratio : float
        Pooling ratio per branch.
    temperature : float
        Temperature scaling in graph attention softmax.
    num_classes : int
        Output classes (2 for binary spoof/bonafide).
    """

    def __init__(
        self,
        wav2vec2_model_name: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: bool = True,
        unfreeze_top_n_transformer: int = 4,
        frontend_dropout: float = 0.1,
        encoder_out_dim: int = 256,
        num_temporal_nodes: int = 25,
        num_spatial_nodes: int = 25,
        temporal_dim: int = 64,
        spatial_dim: int = 64,
        stack_dim: int = 128,
        num_branches: int = 4,
        pool_ratio: float = 0.5,
        temperature: float = 1.0,
        num_classes: int = 2,
    ):
        super().__init__()

        # ── Frontend ──────────────────────────────────────────────────────
        self.frontend = Wav2Vec2Frontend(
            model_name=wav2vec2_model_name,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
            unfreeze_top_n_transformer=unfreeze_top_n_transformer,
            dropout=frontend_dropout,
            output_dim=encoder_out_dim,
        )

        # ── Graph backbone ────────────────────────────────────────────────
        self.graph_formation = GraphFormation(
            encoder_dim=encoder_out_dim,
            num_temporal_nodes=num_temporal_nodes,
            num_spatial_nodes=num_spatial_nodes,
            temporal_dim=temporal_dim,
            spatial_dim=spatial_dim,
            pool_ratio=pool_ratio,
            temperature=temperature,
        )

        self.backbone = MultiBranchArchitecture(
            temporal_dim=temporal_dim,
            spatial_dim=spatial_dim,
            # Pass pooled node counts so all parallel branches are correctly sized
            # (same fix as AASIST3_Raw — graph_formation already pooled the nodes).
            num_temporal_nodes=self.graph_formation.pooled_temporal_nodes,
            num_spatial_nodes=self.graph_formation.pooled_spatial_nodes,
            stack_dim=stack_dim,
            num_branches=num_branches,
            pool_ratio=pool_ratio,
            temperature=temperature,
            dropout_p1=0.2,
            dropout_p2=0.5,
        )

        self.output_head = AASIST3OutputHead(
            hidden_dim=self.backbone.hidden_dim,
            num_classes=num_classes,
            use_intermediate=True,
            intermediate_dim=128,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, T)  — raw waveform, 16 kHz, peak-normalised

        Returns
        -------
        logits : (B, num_classes)
        """
        # (B, 1, T) → (B, encoder_out_dim, T_ds)
        features = self.frontend(x)

        # Graph formation
        h_t, h_s = self.graph_formation(features)

        # Multi-branch backbone
        hidden = self.backbone(h_t, h_s)

        # Classification head
        logits = self.output_head(hidden)
        return logits

    # ------------------------------------------------------------------
    # Convenience wrappers for two-stage training
    # ------------------------------------------------------------------
    def freeze_frontend(self):
        """Freeze all wav2vec2 parameters (graph backbone stays trainable)."""
        for p in self.frontend.parameters():
            p.requires_grad = False
        # Keep layer_weights trainable
        self.frontend.layer_weights.requires_grad = True
        print("[AASIST3_Wav2Vec2] Frontend frozen (layer_weights kept).")

    def unfreeze_top_n_transformer(self, n: int):
        """Progressively unfreeze top-N wav2vec2 transformer layers."""
        self.frontend.unfreeze_top_n_layers(n)

    def get_param_groups(
        self,
        lr_frontend: float = 1e-5,
        lr_backbone: float = 1e-4,
        weight_decay_backbone: float = 0.0,
    ) -> List[dict]:
        """
        Return two param groups so you can assign different learning rates:
            • frontend (wav2vec2)  →  lr_frontend  (typically 10× smaller)
            • graph backbone       →  lr_backbone

        weight_decay_backbone is applied to the graph backbone only — the
        wav2vec2 frontend always uses weight_decay=0, to avoid decaying
        pretrained weights.

        Example
        -------
            optimizer = Adam(
                model.get_param_groups(lr_frontend=1e-5, lr_backbone=1e-4,
                                        weight_decay_backbone=1e-4)
            )
        """
        frontend_ids = set(id(p) for p in self.frontend.parameters())
        frontend_params = [p for p in self.parameters()
                           if id(p) in frontend_ids and p.requires_grad]
        backbone_params  = [p for p in self.parameters()
                           if id(p) not in frontend_ids and p.requires_grad]
        return [
            {"params": frontend_params, "lr": lr_frontend, "weight_decay": 0.0,
             "name": "frontend"},
            {"params": backbone_params,  "lr": lr_backbone,
             "weight_decay": weight_decay_backbone, "name": "backbone"},
        ]


# ===========================================================================
# VRAM estimator
# ===========================================================================

def estimate_vram(model_name: str, batch_size: int, max_len: int = 64600) -> dict:
    """
    Rough VRAM estimate for planning batch size on RTX 5060 Ti 16 GB.
    Numbers are approximate (BF16, forward + backward + Adam states).
    """
    # wav2vec2 hidden states dominate; T_ds = max_len / 320
    T_ds = max_len // 320
    hidden = 768 if "base" in model_name else 1024
    num_layers = 13 if "base" in model_name else 25

    # Bytes: 2 per BF16 element
    # All hidden states kept for weighted sum: num_layers * B * T_ds * hidden
    hidden_states_gb = (num_layers * batch_size * T_ds * hidden * 2) / 1e9
    # Model weights (BF16): ~190 MB base, ~634 MB large
    weights_gb = 0.19 if "base" in model_name else 0.63
    # Adam optimizer states (2x FP32 moments) for trainable params
    # Assuming ~50M trainable for base with top-6 unfrozen
    optim_gb = 0.4 if "base" in model_name else 1.2
    # Graph activations (rough)
    graph_gb = batch_size * 0.05

    total = hidden_states_gb + weights_gb + optim_gb + graph_gb
    return {
        "hidden_states_gb": round(hidden_states_gb, 2),
        "weights_gb":       round(weights_gb, 2),
        "optimizer_gb":     round(optim_gb, 2),
        "graph_gb":         round(graph_gb, 2),
        "total_est_gb":     round(total, 2),
    }


# ===========================================================================
# Training entry-point
# ===========================================================================

def run_wav2vec2_training():
    """
    End-to-end training script for AASIST3_Wav2Vec2 on ASVspoof5.

    Optimised defaults for RTX 5060 Ti 16 GB / CUDA 12.0 / BF16.

    Single-stage training, epochs 1-30: the wav2vec2 CNN feature extractor
    stays frozen, the top --unfreeze_top_n transformer layers are trainable
    from epoch 1 at frontend LR (10x smaller than backbone LR), and the graph
    backbone trains throughout at backbone LR. One optimizer for the whole
    run. (A prior two-stage freeze/unfreeze design rebuilt the optimizer
    partway through expecting to unlock new parameters — it didn't, since
    the same top-N layers were already trainable from construction, so the
    "stage 2" transition only discarded Adam's momentum/variance state for
    params already training, which destabilized training right at that
    boundary. Removed in favor of this single-stage setup.)
    """
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(
        description="AASIST3 + Wav2Vec2 Training on ASVspoof5"
    )

    dataset_root    = r"M:\Datasets\ASVspoof5"
    default_results = r"M:\Results\ASVspoof5\AASIST3Wav2Vec2"

    # ── Dataset paths ──────────────────────────────────────────────────
    parser.add_argument("--train_audio_dir",  type=str,
                        default=os.path.join(dataset_root, "flac_T"))
    parser.add_argument("--train_protocol",   type=str,
                        default=os.path.join(dataset_root, "ASVspoof5.train.tsv"))
    parser.add_argument("--dev_audio_dir",    type=str,
                        default=os.path.join(dataset_root, "flac_D"))
    parser.add_argument("--dev_protocol",     type=str,
                        default=os.path.join(dataset_root, "ASVspoof5.dev.track_1.tsv"))

    # ── Output ─────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint_dir",  type=str, default=default_results)
    parser.add_argument("--experiment_name", type=str, default="aasist3_wav2vec2")

    # ── Training hyper-params (RTX 5060 Ti 16 GB defaults) ─────────────
    parser.add_argument("--epochs",              type=int,   default=30)
    parser.add_argument("--start_epoch",         type=int,   default=0)
    parser.add_argument("--batch_size",          type=int,   default=24,
                        help="RTX 5060 Ti 16 GB: 24 for wav2vec2-base, 12 for large")
    parser.add_argument("--lr_backbone",         type=float, default=1e-4,
                        help="LR for graph backbone (stage 1 and 2)")
    parser.add_argument("--lr_frontend",         type=float, default=5e-6,
                        help="LR for wav2vec2 params (stage 2 fine-tuning)")
    parser.add_argument("--max_len",             type=int,   default=64600,
                        help="Max audio samples (4 s @ 16 kHz = 64600)")
    parser.add_argument("--num_workers",         type=int,   default=4,
                        help="DataLoader workers (4-8 on Windows is fine)")
    parser.add_argument("--patience",            type=int,   default=5,
                        help="Early-stopping patience on val_min_dcf. Lowered from "
                             "15: the frozen-frontend backbone starts overfitting "
                             "(val_loss/val_eer worsening) within a couple epochs, "
                             "so a long patience just trains further into the overfit.")
    parser.add_argument("--subset",              type=int,   default=None)
    parser.add_argument("--augment",             action="store_true", default=False,
                        help="Enable on-the-fly training augmentation (random gain, "
                             "low-pass filtering, mu-law quantization, additive "
                             "Gaussian noise, pitch shift) — no extra dataset/ffmpeg "
                             "required. Train split only; dev stays clean. See "
                             "AudioProcessor.augment_waveform() in AASIST3.py.")
    parser.add_argument("--accumulation_steps",  type=int,   default=1,
                        help="Grad accumulation (1 = off; 16 GB VRAM is sufficient)")
    parser.add_argument("--max_grad_norm",       type=float, default=0.5,
                        help="Gradient clip norm. 0.5 is safer with KAN layers.")
    # Paper (Section 3.11) found plain Adam with no weight decay optimal for the
    # original AASIST3_Raw config. This run's history shows the graph backbone
    # overfitting the frozen wav2vec2 features almost immediately (train_acc
    # >99% by epoch 2 while val_loss climbs), so a small weight decay is applied
    # to the backbone param group only — the wav2vec2 frontend always keeps 0.0.
    parser.add_argument("--weight_decay",        type=float, default=1e-4,
                        help="L2 regularization applied to the graph-backbone "
                             "param group only (frontend/wav2vec2 always uses "
                             "weight_decay=0).")
    parser.add_argument("--resume",              type=str,   default=None)

    # ── Loss options ────────────────────────────────────────────────────
    # Paper (Section 3.10): "regular cross-entropy was indeed efficacious.
    # Focal loss, weighted CE, and label smoothing all proved ineffective."
    parser.add_argument("--use_focal",           action="store_true", default=False,
                        help="Use FocalLoss (paper: proved ineffective, disabled by default).")
    parser.add_argument("--focal_gamma",         type=float, default=2.0)
    parser.add_argument("--label_smoothing",     type=float, default=0.0,
                        help="Label smoothing. Paper: 0.0 (plain CE is best).")

    # ── wav2vec2 options ───────────────────────────────────────────────
    parser.add_argument("--wav2vec2_model",      type=str,
                        default="facebook/wav2vec2-base",
                        help="HuggingFace model id. "
                             "Use 'facebook/wav2vec2-large' for upgrade (bs=12).")
    parser.add_argument("--freeze_transformer",  action="store_true",
                        default=True,
                        help="Freeze wav2vec2 transformer layers on init")
    parser.add_argument("--unfreeze_top_n",      type=int, default=6,
                        help="Top-N transformer layers trainable from epoch 1 "
                             "(6 of 12 for base, 8 of 24 for large). Single-stage "
                             "training — set to 0 to keep the whole transformer frozen.")

    # ── Graph backbone options (larger for 16 GB) ──────────────────────
    parser.add_argument("--num_temporal_nodes",  type=int, default=25,
                        help="Temporal graph nodes. 25=original AASIST, 50=larger but 4x slower")
    parser.add_argument("--num_spatial_nodes",   type=int, default=25,
                        help="Spatial graph nodes")
    parser.add_argument("--temporal_dim",        type=int, default=64)
    parser.add_argument("--spatial_dim",         type=int, default=64)
    parser.add_argument("--stack_dim",           type=int, default=128)
    parser.add_argument("--num_branches",        type=int, default=4)

    # ── Precision / compilation (RTX 5060 Ti Blackwell) ───────────────
    parser.add_argument("--amp_dtype",           type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="AMP dtype. bf16 is native on Blackwell and more stable.")
    parser.add_argument("--disable_amp",         action="store_true", default=False,
                        help="Disable AMP entirely (sets amp_dtype=fp32)")
    parser.add_argument("--use_compile",         action="store_true", default=False,
                        help="Apply torch.compile() for ~20-30%% speedup on Blackwell. "
                             "Adds ~2 min warm-up on first batch.")
    parser.add_argument("--compile_mode",        type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")

    # ── LR schedule ────────────────────────────────────────────────────
    # Paper (Section 3.12): scheduler proved ineffective — default is 'none'.
    parser.add_argument("--scheduler",           type=str, default="none",
                        choices=["none", "cosine", "plateau"],
                        help="LR scheduler. Paper: none is optimal.")
    parser.add_argument("--warmup_epochs",       type=int, default=2,
                        help="Linear warmup epochs (only used when --scheduler cosine).")

    parser.add_argument("--sw_every_n_epochs",   type=int, default=5,
                        help="Run expensive sliding-window validation every N epochs. "
                             "Fast validate() is used for other epochs. (0=always SW)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── GPU info + VRAM estimate ────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        bf16_ok  = torch.cuda.is_bf16_supported()
        print(f"GPU  : {gpu_name}  ({vram_gb:.1f} GB)")
        print(f"BF16 : {'supported' if bf16_ok else 'NOT supported'}")
        vram_est = estimate_vram(args.wav2vec2_model, args.batch_size, args.max_len)
        print(f"VRAM estimate for bs={args.batch_size}: ~{vram_est['total_est_gb']:.1f} GB")
        if vram_est['total_est_gb'] > vram_gb * 0.9:
            print(f"[WARNING] Estimated VRAM ({vram_est['total_est_gb']:.1f} GB) "
                  f"is close to GPU limit ({vram_gb:.1f} GB). "
                  f"Consider reducing --batch_size.")
    else:
        print("Device: CPU (training will be very slow)")

    # ── Resolve AMP dtype ──────────────────────────────────────────────
    if args.disable_amp:
        use_amp = False
        amp_dtype_torch = torch.float32
    else:
        use_amp = True
        if args.amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype_torch = torch.bfloat16
        elif args.amp_dtype == "fp16":
            amp_dtype_torch = torch.float16
        else:
            amp_dtype_torch = torch.float32
            use_amp = False
    print(f"AMP  : {'ON' if use_amp else 'OFF'}  dtype={amp_dtype_torch}")

    # ── Datasets ───────────────────────────────────────────────────────
    print("Loading datasets...")
    train_dataset = RawASV5Dataset(
        args.train_audio_dir, args.train_protocol,
        max_len=args.max_len, is_train=True, strict_labels=True,
        augment=args.augment
    )
    print(f"Training augmentation: {'ON' if args.augment else 'OFF'}")
    dev_dataset = RawASV5Dataset(
        args.dev_audio_dir, args.dev_protocol,
        max_len=args.max_len, is_train=False, strict_labels=False
    )

    if args.subset:
        train_dataset = torch.utils.data.Subset(
            train_dataset, torch.randperm(len(train_dataset))[:args.subset]
        )
        dev_dataset = torch.utils.data.Subset(
            dev_dataset, torch.randperm(len(dev_dataset))[:args.subset]
        )
        print(f"Subset: {args.subset} samples each.")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0)
    )

    # ── Model ──────────────────────────────────────────────────────────
    print("Initialising AASIST3_Wav2Vec2 model...")
    model = AASIST3_Wav2Vec2(
        wav2vec2_model_name=args.wav2vec2_model,
        freeze_feature_extractor=True,
        freeze_transformer_layers=args.freeze_transformer,
        unfreeze_top_n_transformer=args.unfreeze_top_n,
        encoder_out_dim=256,
        num_temporal_nodes=args.num_temporal_nodes,
        num_spatial_nodes=args.num_spatial_nodes,
        temporal_dim=args.temporal_dim,
        spatial_dim=args.spatial_dim,
        stack_dim=args.stack_dim,
        num_branches=args.num_branches,
    ).to(device)

    # ── torch.compile (Blackwell ~20-30% speedup) ─────────────────────
    # torch.compile's default backend (inductor) requires Triton.
    # Triton is not available on Windows — detect via importlib (linter-safe).
    if args.use_compile and hasattr(torch, "compile"):
        import importlib.util
        _triton_ok = importlib.util.find_spec("triton") is not None

        if _triton_ok:
            print(f"Applying torch.compile(mode='{args.compile_mode}') ...")
            print("  NOTE: First batch will take ~2 min to compile. This is normal.")
            model = torch.compile(model, mode=args.compile_mode)
        else:
            print("[WARNING] --use_compile skipped: Triton is not installed "
                  "(not supported on Windows natively).")
            print("  Training will proceed without torch.compile.")

    # ── Optimizer ─────────────────────────────────────────────────────
    # Paper (Section 3.11): plain Adam is optimal — no weight_decay.
    # Two param groups: frontend (frozen by default) and backbone.
    raw_model = model if not hasattr(model, '_orig_mod') else model._orig_mod
    optimizer = Adam(
        raw_model.get_param_groups(
            lr_frontend=args.lr_frontend,
            lr_backbone=args.lr_backbone,
            weight_decay_backbone=args.weight_decay,
        ),
    )
    print(f"Optimizer: Adam (backbone weight_decay={args.weight_decay}, frontend weight_decay=0.0)")

    # ── LR Scheduler ───────────────────────────────────────────────────
    def make_scheduler(opt, stage_epochs, warmup=args.warmup_epochs):
        if args.scheduler == "cosine":
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warmup_sched = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                                    total_iters=warmup)
            cosine_sched = CosineAnnealingLR(opt, T_max=max(1, stage_epochs - warmup),
                                             eta_min=1e-7)
            return SequentialLR(opt, schedulers=[warmup_sched, cosine_sched],
                                milestones=[warmup])
        elif args.scheduler == "plateau":
            return ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
        else:
            return None

    scheduler = make_scheduler(optimizer, args.epochs)

    # ── Loss ───────────────────────────────────────────────────────────
    # Paper (Section 3.10): "regular cross-entropy was indeed efficacious.
    # Focal loss, class weighting, and label smoothing all proved ineffective."
    if args.use_focal:
        # Kept as an opt-in for experimentation only.
        criterion = FocalLoss(gamma=args.focal_gamma)
        print(f"Loss: FocalLoss(gamma={args.focal_gamma}) [non-default, paper: ineffective]")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        smoothing_tag = f"label_smoothing={args.label_smoothing}" if args.label_smoothing > 0 else "no smoothing"
        print(f"Loss: CrossEntropyLoss ({smoothing_tag})")

    # ── Print summary ──────────────────────────────────────────────────
    raw_model = model if not hasattr(model, '_orig_mod') else model._orig_mod
    params = count_parameters(raw_model)
    print(f"Total parameters  : {params['total']:,}")
    print(f"Trainable params  : {params['trainable']:,}")
    try:
        print_model_summary(model, device, (1, 1, args.max_len))
    except Exception:
        pass
    torch.cuda.empty_cache()
    sw_every = args.sw_every_n_epochs

    trainer = TrainAASIST3(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        accumulation_steps=args.accumulation_steps,
        use_amp=use_amp,
        max_grad_norm=args.max_grad_norm,
        max_len=args.max_len,
        sample_rate=16000,
    )

    if amp_dtype_torch == torch.bfloat16:
        trainer.scaler = torch.amp.GradScaler('cuda', enabled=False)
        print("GradScaler: disabled (BF16 does not require loss scaling).")

    resume_epoch = 0
    if args.resume:
        resume_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resuming from epoch {resume_epoch + 1}")

    start_epoch = args.start_epoch if args.start_epoch > 0 else resume_epoch

    # Single-stage training: the top-N transformer layers (--unfreeze_top_n)
    # are already trainable from model construction above (see
    # Wav2Vec2Frontend.__init__), so there is nothing left to "unfreeze"
    # partway through. A prior two-stage design rebuilt the optimizer at a
    # later epoch expecting to unlock new parameters — it didn't (same
    # unfreeze_top_n value both times), so all that stage transition ever did
    # was discard Adam's momentum/variance state for params already training,
    # which is exactly what caused the instability observed at that boundary
    # (gradient norms climbing sharply right at the switch). One optimizer,
    # one fit() call, for the whole run avoids that reset entirely.
    print(f"\n=== TRAINING: epochs {start_epoch+1}-{args.epochs} "
          f"(top-{args.unfreeze_top_n} transformer layers trainable throughout, "
          f"frontend LR={args.lr_frontend}, backbone LR={args.lr_backbone}) ===")
    try:
        trainer.fit(
            train_loader, dev_loader,
            num_epochs=args.epochs,
            early_stopping_patience=args.patience,
            start_epoch=start_epoch,
        )
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print("FATAL ERROR:\n", error_msg)
        log_path = os.path.join(os.getcwd(), "debug_error_wav2vec2.txt")
        with open(log_path, "w") as f:
            f.write(error_msg)
        raise


def quick_sanity_test():
    """
    Instantiate the model with RTX 5060 Ti 16 GB optimal settings and run
    a forward pass. Also profiles VRAM usage and throughput.
    """
    print("=" * 65)
    print("  AASIST3_Wav2Vec2 -- Sanity Test  (RTX 5060 Ti 16 GB config)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU  : {gpu_name}")
        print(f"VRAM : {vram_gb:.1f} GB")
        print(f"BF16 : {'supported' if torch.cuda.is_bf16_supported() else 'NOT supported'}")
        print(f"Torch: {torch.__version__}")

    MODEL_NAME         = "facebook/wav2vec2-base"
    BATCH_SIZE         = 24   
    NUM_TEMPORAL_NODES = 50
    NUM_SPATIAL_NODES  = 50
    T                  = 64600  # 4 s @ 16 kHz

    vram_est = estimate_vram(MODEL_NAME, BATCH_SIZE, T)
    print(f"\nEstimated VRAM for bs={BATCH_SIZE}: ~{vram_est['total_est_gb']:.1f} GB")

    print("\nLoading model...")
    model = AASIST3_Wav2Vec2(
        wav2vec2_model_name=MODEL_NAME,
        freeze_feature_extractor=True,
        freeze_transformer_layers=True,
        unfreeze_top_n_transformer=6,
        encoder_out_dim=256,
        num_temporal_nodes=NUM_TEMPORAL_NODES,
        num_spatial_nodes=NUM_SPATIAL_NODES,
        temporal_dim=64,
        spatial_dim=64,
        stack_dim=128,
        num_branches=4,
    ).to(device)

    model.eval()

    params = count_parameters(model)
    print(f"Total parameters    : {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")

    x = torch.randn(2, 1, T).to(device) 
    print(f"\nForward pass (bs=2)...")
    print(f"  Input shape : {tuple(x.shape)}")
    with torch.no_grad():
        logits = model(x)
    print(f"  Output shape: {tuple(logits.shape)}")
    print(f"  Logits      : {logits.tolist()}")

    if torch.cuda.is_bf16_supported():
        print("\nBF16 AMP forward pass...")
        x_bf = x.to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits_bf = model(x_bf)
        print(f"  BF16 output shape: {tuple(logits_bf.shape)}")
        print(f"  BF16 logits      : {logits_bf.tolist()}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved  = torch.cuda.memory_reserved(0)  / 1024**3
        print(f"\nVRAM allocated: {allocated:.2f} GB")
        print(f"VRAM reserved : {reserved:.2f} GB")

    print("\n[OK] Sanity test passed.")
    print("\nRecommended training command (RTX 5060 Ti 16 GB):")
    print("  python AASIST3_Wav2Vec2.py \\")
    print("    --batch_size 24 --accumulation_steps 1 \\")
    print("    --amp_dtype bf16 --use_compile \\")
    print("    --num_temporal_nodes 50 --num_spatial_nodes 50 \\")
    print("    --unfreeze_top_n 6 \\")
    print("    --lr_backbone 1e-4 --lr_frontend 5e-6 \\")
    print("    --scheduler cosine --epochs 30")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        quick_sanity_test()
    else:
        run_wav2vec2_training()
