from __future__ import annotations

import argparse
import gc
import logging
import math
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from util_funcs import (
    add_knn_edges,
    annealing_weights,
    compute_full_correlation_matrix,
    knn_impute_aknn,
    knn_impute_aknnbiased,
    load_dataset,
    select_topk_features_per_feature,
    soft_clamp,
)

LOGGER_NAME = "ours"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-JEPA pipeline with auxiliary binary loss to boost AUC.")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--group_size", type=int, default=10)
    parser.add_argument("--mr", type=float, default=0.6)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--jepa_layers", type=int, default=2)
    parser.add_argument("--jepa_heads", type=int, default=4)
    parser.add_argument("--jepa_patches", type=int, default=16)
    parser.add_argument("--jepa_momentum", type=float, default=0.995)
    parser.add_argument("--jepa_dropout", type=float, default=0.1)
    parser.add_argument("--jepa_backbone_layers", type=int, default=3)
    parser.add_argument("--jepa_context_patches", type=int, default=1)
    parser.add_argument("--jepa_target_patches", type=int, default=4)
    parser.add_argument("--jepa_patch_dropout", type=float, default=0.1)
    parser.add_argument("--jepa_ssl_weight", type=float, default=0.05)
    parser.add_argument("--alpha_sel", type=float, default=0.5)
    parser.add_argument("--clamp_beta_max", type=float, default=50.0)
    parser.add_argument("--update_warmup", type=int, default=15)
    parser.add_argument("--impute_alpha", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--mae_tolerance", type=float, default=2e-4, help="Minimum MAE improvement considered significant.")
    parser.add_argument("--auc_tolerance", type=float, default=2e-4, help="Minimum AUC improvement considered significant.")
    parser.add_argument("--gate_bias_scale", type=float, default=4.0)
    parser.add_argument("--aux_bce_weight", type=float, default=0.5, help="Weight for auxiliary binary classification loss.")
    parser.add_argument("--log_dir", type=str, default="logs/ours")
    parser.add_argument("--feature_dir", type=str, default="features/ours")
    parser.add_argument("--feature_ratio", type=float, default=0.4)
    parser.add_argument("--feature_constant", type=float, default=10.0)
    return parser.parse_args()


def setup_signal_handlers() -> None:
    def _clean_up(signum, _frame):
        logging.getLogger(LOGGER_NAME).info("Signal %s received. Releasing resources.", signum)
        torch.cuda.empty_cache()
        gc.collect()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP):
        signal.signal(sig, _clean_up)


def build_logger(args: argparse.Namespace) -> logging.Logger:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"gjauc_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.dataset}_mr{args.mr}_gs{args.group_size}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


@dataclass
class GroupBundle:
    gid: int
    idx_cpu: torch.Tensor
    idx_gpu: torch.Tensor
    support_idx: torch.Tensor
    model: nn.Module
    optimizer: torch.optim.Optimizer
    best_loss: float = float("inf")


class GatedLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        L_sel: int,
        L_knn: int,
        L_int: int,
        sel_mask_mean: torch.Tensor,
        gate_scale: float,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        if L_sel > 0:
            alpha = 0.2
            soft_mask = ((sel_mask_mean * (1 - alpha)) + (0.5 * alpha)).clamp(1e-4, 1 - 1e-4)
            logit = torch.log(soft_mask / (1 - soft_mask)) * gate_scale
            self.linear.bias.data[:L_sel] = logit
        if L_knn + L_int > 0:
            self.linear.bias.data[L_sel : L_sel + L_knn + L_int] = 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)) * x


class PatchTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        num_layers = max(1, num_layers)
        num_heads = max(1, num_heads)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 2,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() == 2:
            patches = patches.unsqueeze(0)
        encoded = self.encoder(patches)
        return encoded.squeeze(0)


class GraphJepaBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        num_layers = max(2, num_layers)
        self.convs = nn.ModuleList()
        for idx in range(num_layers):
            in_dim = -1 if idx == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = x
        for idx, conv in enumerate(self.convs):
            out = conv(out, edge_index)
            if idx != len(self.convs) - 1:
                out = F.relu(out)
                out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class GraphJepaImputerBatchAUC(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        L_sel: int,
        L_knn: int,
        L_int: int,
        sel_mask_mean: torch.Tensor,
        dropout: float,
        gate_scale: float,
        num_patches: int,
        transformer_layers: int,
        num_heads: int,
        momentum: float,
        transformer_dropout: float,
        backbone_layers: int,
        num_context: int,
        num_target: int,
        patch_dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = max(1, num_patches)
        self.num_context = max(1, num_context)
        self.num_target = max(1, num_target)
        self.patch_dropout = float(max(0.0, min(1.0, patch_dropout)))
        self.momentum = float(momentum)

        self.gate = GatedLayer(in_dim, L_sel, L_knn, L_int, sel_mask_mean, gate_scale)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.encoder_q = GraphJepaBackbone(hidden_dim, backbone_layers, dropout)
        self.encoder_k = GraphJepaBackbone(hidden_dim, backbone_layers, dropout)
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.context_patch_encoder = PatchTransformer(hidden_dim, num_heads, transformer_layers, transformer_dropout)
        self.target_patch_encoder = PatchTransformer(hidden_dim, num_heads, transformer_layers, transformer_dropout)
        self.target_patch_encoder.load_state_dict(self.context_patch_encoder.state_dict())
        for param in self.target_patch_encoder.parameters():
            param.requires_grad = False

        self.patch_positional = nn.Embedding(self.num_patches, hidden_dim)
        self.target_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.target_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.ssl_loss: torch.Tensor | None = None

    def _assign_patches(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        if num_nodes == 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        repeats = math.ceil(num_nodes / self.num_patches)
        base = torch.arange(self.num_patches, device=device).repeat_interleave(repeats)
        patch_ids = base[:num_nodes].clone()
        perm = torch.randperm(num_nodes, device=device)
        return patch_ids[perm]

    def _aggregate_patches(self, node_repr: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        hidden_dim = node_repr.size(1)
        patch_repr = node_repr.new_zeros(self.num_patches, hidden_dim)
        patch_repr.index_add_(0, patch_ids, node_repr)
        counts = torch.bincount(patch_ids, minlength=self.num_patches).clamp_min(1).unsqueeze(1).to(node_repr.dtype)
        return patch_repr / counts

    def _select_patch_indices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        total = self.num_patches
        perm = torch.randperm(total, device=device)
        num_context = min(self.num_context, total)
        context_idx = perm[:num_context]
        remaining = perm[num_context:]
        if remaining.numel() == 0:
            remaining = perm
        num_target = min(self.num_target, remaining.numel())
        target_idx = remaining[:num_target]
        if context_idx.numel() == 0:
            context_idx = perm[:1]
        if target_idx.numel() == 0:
            target_idx = perm[:1]
        if self.patch_dropout > 0 and context_idx.numel() > 1:
            keep_mask = torch.rand(context_idx.size(0), device=device) > self.patch_dropout
            if not keep_mask.any():
                keep_mask[0] = True
            context_idx = context_idx[keep_mask]
        return context_idx, target_idx

    def forward(self, inp: torch.Tensor, edge_index: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
        device = inp.device
        x = self.gate(inp)
        x = self.input_proj(x)
        context_nodes = self.encoder_q(x, edge_index)
        with torch.no_grad():
            target_nodes = self.encoder_k(x, edge_index)

        patch_ids = self._assign_patches(context_nodes.size(0), context_nodes.device)
        context_patch = self._aggregate_patches(context_nodes, patch_ids)
        target_patch = self._aggregate_patches(target_nodes, patch_ids)
        context_idx, target_idx = self._select_patch_indices(device)

        patch_pos = self.patch_positional.weight
        context_tokens = context_patch[context_idx] + patch_pos[context_idx]
        target_tokens = target_patch[target_idx] + patch_pos[target_idx]

        context_tokens = context_tokens.unsqueeze(0)
        target_tokens = target_tokens.unsqueeze(0)
        context_encoded = self.context_patch_encoder(context_tokens).squeeze(0)
        with torch.no_grad():
            target_encoded = self.target_patch_encoder(target_tokens).squeeze(0)
            target_encoded = torch.cat([torch.cosh(target_encoded), torch.sinh(target_encoded)], dim=-1)
            target_teacher = self.target_projection(target_encoded)

        context_summary = context_encoded.mean(dim=0, keepdim=True)
        target_prediction_embeddings = context_summary + patch_pos[target_idx]
        target_pred = self.target_predictor(target_prediction_embeddings)
        if target_pred.size(0) == target_teacher.size(0):
            self.ssl_loss = F.smooth_l1_loss(target_pred, target_teacher.detach())
        else:
            self.ssl_loss = torch.zeros(1, device=device)

        patch_context_full = context_nodes.new_zeros(self.num_patches, self.hidden_dim)
        patch_context_full[context_idx] = context_encoded
        patch_target_full = context_nodes.new_zeros(self.num_patches, self.hidden_dim)
        patch_target_full[target_idx] = target_pred

        context_per_node = patch_context_full[patch_ids]
        target_per_node = patch_target_full[patch_ids]
        combined = torch.cat([context_nodes, context_per_node, target_per_node], dim=1)
        out_reg = soft_clamp(self.output_head(combined), beta)
        out_cls = self.classification_head(combined)
        return out_reg, out_cls

    @torch.no_grad()
    def momentum_update(self) -> None:
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum)
            param_k.data.add_((1.0 - self.momentum) * param_q.data)
        for param_q, param_k in zip(self.context_patch_encoder.parameters(), self.target_patch_encoder.parameters()):
            param_k.data.mul_(self.momentum)
            param_k.data.add_((1.0 - self.momentum) * param_q.data)


def chunk_features(num_features: int, group_size: int) -> List[List[int]]:
    group_size = max(1, group_size)
    return [list(range(i, min(i + group_size, num_features))) for i in range(0, num_features, group_size)]


def build_support_sets(groups: Sequence[Sequence[int]], sel_topk: Sequence[torch.Tensor]) -> List[List[int]]:
    subsets: List[List[int]] = []
    for group in groups:
        support = set()
        for feat_idx in group:
            support.update(sel_topk[feat_idx].tolist())
        support.difference_update(group)
        subsets.append(sorted(support))
    return subsets


def create_group_bundles(
    groups: Sequence[Sequence[int]],
    subsets: Sequence[Sequence[int]],
    mask_mean: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> List[GroupBundle]:
    bundles: List[GroupBundle] = []
    for gid, (group, support) in enumerate(zip(groups, subsets)):
        idx_cpu = torch.tensor(group, dtype=torch.long)
        idx_gpu = idx_cpu.to(device)
        support_idx = torch.tensor(support, dtype=torch.long, device=device)
        L_sel = support_idx.numel()
        L_knn = len(group)
        L_int = len(group)
        in_dim = L_sel + 2 * L_knn
        sel_mask_mean = mask_mean[support_idx] if L_sel > 0 else torch.empty(0, device=device)
        model = GraphJepaImputerBatchAUC(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=L_knn,
            L_sel=L_sel,
            L_knn=L_knn,
            L_int=L_int,
            sel_mask_mean=sel_mask_mean,
            dropout=args.dropout,
            gate_scale=args.gate_bias_scale,
            num_patches=args.jepa_patches,
            transformer_layers=args.jepa_layers,
            num_heads=args.jepa_heads,
            momentum=args.jepa_momentum,
            transformer_dropout=args.jepa_dropout,
            backbone_layers=args.jepa_backbone_layers,
            num_context=args.jepa_context_patches,
            num_target=args.jepa_target_patches,
            patch_dropout=args.jepa_patch_dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bundles.append(GroupBundle(gid=gid, idx_cpu=idx_cpu, idx_gpu=idx_gpu, support_idx=support_idx, model=model, optimizer=optimizer))
    return bundles


def create_random_masks(x: torch.Tensor, mr: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_tot = torch.rand_like(x)
    obs_rate = 1.0 - mr
    threshold_val = obs_rate + (mr / 2.0)
    train_mask = mask_tot < obs_rate
    val_mask = (mask_tot >= obs_rate) & (mask_tot < threshold_val)
    test_mask = mask_tot >= threshold_val
    return train_mask, val_mask, test_mask


def mean_neighbor_feature(train_x: torch.Tensor, feature_idx: int, sel_topk: Sequence[torch.Tensor]) -> torch.Tensor:
    neighbors = sel_topk[feature_idx]
    if neighbors.numel() == 0:
        return train_x[:, feature_idx].unsqueeze(1)
    return train_x[:, neighbors].mean(dim=1, keepdim=True)


def build_model_input(
    train_x: torch.Tensor,
    knn_epoch: torch.Tensor,
    bundle: GroupBundle,
    sel_topk: Sequence[torch.Tensor],
    alpha_sel: float,
) -> torch.Tensor:
    device = train_x.device
    segments: List[torch.Tensor] = []
    if bundle.support_idx.numel() > 0:
        x_sel = train_x[:, bundle.support_idx]
        enriched = [mean_neighbor_feature(train_x, int(idx), sel_topk) for idx in bundle.support_idx.tolist()]
        if enriched:
            x_sel = x_sel + alpha_sel * torch.cat(enriched, dim=1)
        segments.append(x_sel)
    x_knn = knn_epoch[:, bundle.idx_cpu].to(device)
    segments.append(x_knn)
    local = [mean_neighbor_feature(train_x, int(idx), sel_topk) for idx in bundle.idx_gpu.tolist()]
    segments.append(torch.cat(local, dim=1))
    return torch.cat(segments, dim=1)


def update_missing_entries(
    train_x: torch.Tensor,
    pred_eval: torch.Tensor,
    bundle: GroupBundle,
    train_mask: torch.Tensor,
    alpha: float,
) -> None:
    miss_mask = ~train_mask[:, bundle.idx_cpu]
    if miss_mask.sum().item() == 0:
        return
    rows, cols = torch.where(miss_mask)
    rows_gpu = rows.to(train_x.device)
    cols_gpu = cols.to(train_x.device)
    feat_idx = bundle.idx_gpu[cols_gpu]
    blended = alpha * pred_eval.detach()[rows_gpu, cols_gpu] + (1 - alpha) * train_x[rows_gpu, feat_idx]
    train_x[rows_gpu, feat_idx] = torch.clamp(blended, 0.0, 1.0)


def _binary_auc(pred: torch.Tensor, target: torch.Tensor) -> float:
    flat_target = target.view(-1)
    mask = (flat_target == 0) | (flat_target == 1)
    if not mask.any():
        return float("nan")
    flat_pred = pred.view(-1)[mask]
    flat_target = flat_target[mask]
    pos = flat_target.sum().item()
    neg = flat_target.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(flat_pred)
    sorted_target = flat_target[order]
    ranks = torch.arange(1, flat_target.numel() + 1, dtype=flat_pred.dtype, device=flat_pred.device)
    rank_sum_pos = ranks[sorted_target == 1].sum()
    auc = (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc.item())


def _recall_non_zero(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    non_zero_mask = mask & (target == 1)
    if not non_zero_mask.any():
        return float("nan")
    return float(pred[non_zero_mask].mean().item())


def evaluate_predictions(
    predictions: torch.Tensor,
    cls_logits: torch.Tensor,
    original_x: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> dict[str, float]:
    pred_cpu = predictions.detach().cpu().clamp(0.0, 1.0)
    cls_prob = torch.sigmoid(cls_logits.detach()).cpu().clamp(0.0, 1.0)
    val_pred = pred_cpu[val_mask]
    val_true = original_x[val_mask]
    test_pred = pred_cpu[test_mask]
    test_true = original_x[test_mask]
    val_prob = cls_prob[val_mask]
    test_prob = cls_prob[test_mask]
    val_mae = F.l1_loss(val_pred, val_true)
    val_rmse = torch.sqrt(F.mse_loss(val_pred, val_true))
    test_mae = F.l1_loss(test_pred, test_true)
    test_rmse = torch.sqrt(F.mse_loss(test_pred, test_true))
    val_auc = _binary_auc(val_prob, val_true)
    test_auc = _binary_auc(test_prob, test_true)
    val_recall_non_zero = _recall_non_zero(pred_cpu, original_x, val_mask)
    test_recall_non_zero = _recall_non_zero(pred_cpu, original_x, test_mask)
    return {
        "val_mae": float(val_mae.item()),
        "val_rmse": float(val_rmse.item()),
        "val_auc": val_auc,
        "val_recall_non_zero": val_recall_non_zero,
        "test_mae": float(test_mae.item()),
        "test_rmse": float(test_rmse.item()),
        "test_auc": test_auc,
        "test_recall_non_zero": test_recall_non_zero,
    }


def select_tracked_cells(
    original_x: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, max_cells: int = 10
) -> list[tuple[int, int]]:
    tracked: list[tuple[int, int]] = []

    def _add_from(mask: torch.Tensor, limit: int) -> None:
        if limit <= 0:
            return
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            return
        step = max(1, coords.size(0) // max(1, limit))
        for coord in coords[::step][:limit]:
            cell = (int(coord[0]), int(coord[1]))
            if cell not in tracked:
                tracked.append(cell)

    half = max_cells // 2
    _add_from(val_mask & (original_x == 1), half)
    _add_from(test_mask & (original_x == 1), max_cells - len(tracked))
    if len(tracked) < max_cells:
        _add_from(val_mask, max_cells - len(tracked))
    if len(tracked) < max_cells:
        _add_from(test_mask, max_cells - len(tracked))
    return tracked[:max_cells]


def format_tracked_cells(
    pred_cpu: torch.Tensor, cls_prob_cpu: torch.Tensor, original_x: torch.Tensor, cells: list[tuple[int, int]]
) -> list[str]:
    formatted: list[str] = []
    for row, col in cells:
        if row >= pred_cpu.size(0) or col >= pred_cpu.size(1):
            continue
        formatted.append(
            f"({row},{col}) gt={original_x[row, col]:.0f} reg={pred_cpu[row, col]:.3f} prob={cls_prob_cpu[row, col]:.3f}"
        )
    return formatted


def save_artifacts(
    args: argparse.Namespace,
    train_x_snapshot: torch.Tensor,
    data,
    logger: logging.Logger,
    suffix: str = "",
) -> None:
    feat_dir = Path(args.feature_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feat_dir / f"{args.dataset}_{args.feature_ratio}_C{args.feature_constant}{suffix}.pkl"
    with feature_path.open("wb") as f:
        pickle.dump(train_x_snapshot.numpy(), f)

    rows, cols = data.edge_index.detach().cpu()
    adj = sp.coo_matrix(
        (np.ones(len(rows)), (rows.numpy(), cols.numpy())),
        shape=(train_x_snapshot.shape[0], train_x_snapshot.shape[0]),
    )
    adj_path = feat_dir / f"{args.dataset}_sp_adj{suffix}.pkl"
    with adj_path.open("wb") as f:
        pickle.dump(adj, f)

    labels_path = feat_dir / f"{args.dataset}_labels{suffix}.pkl"
    with labels_path.open("wb") as f:
        pickle.dump(data.y.detach().cpu().numpy(), f)
    logger.info("Saved features to %s", feature_path)


def run(args: argparse.Namespace, logger: logging.Logger) -> None:
    logger.info("torch.cuda.is_available() = %s", torch.cuda.is_available())
    device = torch.device(args.device)
    dataset = load_dataset(args.dataset)
    data = dataset.to("cpu") if args.dataset == "Arxiv" else dataset[0]
    data = data.clone()
    original_x = data.x.float().cpu()
    train_mask, val_mask, test_mask = create_random_masks(original_x, args.mr)
    tracked_cells = select_tracked_cells(original_x, val_mask, test_mask, max_cells=10)
    if tracked_cells:
        logger.info("Tracking %d cells (row, col): %s", len(tracked_cells), tracked_cells)
    else:
        logger.info("No cells selected for tracking (masks might be empty).")
    train_x_cpu = original_x.clone()
    train_x_cpu[~train_mask] = float("nan")
    train_x_cpu = knn_impute_aknn(train_x_cpu)
    initial_gt = train_x_cpu.clone()
    data.edge_index = add_knn_edges(train_x_cpu, data.edge_index, args.k)
    data = data.to(device)

    corr = compute_full_correlation_matrix(train_x_cpu.numpy(), device="cpu")
    sel_topk = [
        torch.tensor(feats, dtype=torch.long, device=device)
        for feats in select_topk_features_per_feature(corr, k=args.K)
    ]
    groups = chunk_features(original_x.shape[1], args.group_size)
    subsets = build_support_sets(groups, sel_topk)
    mask_mean = train_mask.float().mean(dim=0).to(device)
    bundles = create_group_bundles(groups, subsets, mask_mean, args, device)

    train_x = train_x_cpu.to(device)
    gcn_pred_all = torch.empty_like(train_x)
    cls_pred_all = torch.empty_like(train_x)

    best_mae_metrics: dict[str, float] | None = None
    best_auc_metrics: dict[str, float] | None = None
    best_mae_snapshot: torch.Tensor | None = None
    best_auc_snapshot: torch.Tensor | None = None
    mae_snapshot_saved = False
    auc_snapshot_saved = False
    epochs_no_improve = 0

    knn_epoch: torch.Tensor | None = None
    KNN_REFRESH_INTERVAL = 5

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if knn_epoch is None or (epoch - 1) % KNN_REFRESH_INTERVAL == 0:
            train_x_cpu_cur = train_x.detach().cpu()
            knn_epoch = knn_impute_aknnbiased(train_x_cpu_cur, train_mask)
        gcn_pred_all.zero_()
        cls_pred_all.zero_()
        beta = 1.0 + (args.clamp_beta_max - 1.0) * (epoch - 1) / max(1, args.epochs - 1)
        if epoch == 1:
            w_mid_prev, w_true_prev = 1.0, 0.0
        else:
            w_mid_prev, w_true_prev = annealing_weights(epoch - 1, args.epochs)

        for bundle in bundles:
            inp = build_model_input(train_x, knn_epoch, bundle, sel_topk, args.alpha_sel)
            mask_group_cpu = train_mask[:, bundle.idx_cpu]
            if mask_group_cpu.sum().item() == 0:
                bundle.model.eval()
                with torch.no_grad():
                    pred_eval, pred_eval_logits = bundle.model(inp, data.edge_index, beta)
                gcn_pred_all[:, bundle.idx_gpu] = pred_eval
                cls_pred_all[:, bundle.idx_gpu] = pred_eval_logits
                continue

            bundle.model.train()
            bundle.optimizer.zero_grad()
            pred_reg, pred_logits = bundle.model(inp, data.edge_index, beta)
            mask_group = mask_group_cpu.to(device)
            target = (
                w_mid_prev * initial_gt[:, bundle.idx_cpu] + w_true_prev * original_x[:, bundle.idx_cpu]
            ).to(device)
            loss = F.huber_loss(pred_reg[mask_group], target[mask_group], delta=0.5)

            bin_mask = mask_group & ((target == 0) | (target == 1))
            if bin_mask.any() and args.aux_bce_weight > 0:
                # clamp logits before sigmoid to avoid extreme gradients
                logits = pred_logits[bin_mask]
                target_bin = target[bin_mask]
                bce_loss = F.binary_cross_entropy_with_logits(logits, target_bin)
                loss = loss + args.aux_bce_weight * bce_loss

            extra_ssl = getattr(bundle.model, "ssl_loss", None)
            if extra_ssl is not None and args.jepa_ssl_weight > 0:
                loss = loss + args.jepa_ssl_weight * extra_ssl
            loss.backward()
            bundle.optimizer.step()
            if hasattr(bundle.model, "momentum_update"):
                bundle.model.momentum_update()

            bundle.model.eval()
            with torch.no_grad():
                pred_eval, pred_eval_logits = bundle.model(inp, data.edge_index, beta)
            gcn_pred_all[:, bundle.idx_gpu] = pred_eval
            cls_pred_all[:, bundle.idx_gpu] = pred_eval_logits

            if loss.item() < bundle.best_loss and epoch > args.update_warmup:
                bundle.best_loss = loss.item()
                update_missing_entries(train_x, pred_eval, bundle, train_mask, args.impute_alpha)

        metrics = evaluate_predictions(gcn_pred_all, cls_pred_all, original_x, val_mask, test_mask)
        # Monitor how far the iteratively updated train_x has moved away
        # from the initial KNN-imputed features. This helps to quantify
        # the actual impact of the EM-style updates.
        with torch.no_grad():
            mean_update = (train_x.detach().cpu() - initial_gt).abs().mean().item()
        logger.info(
            "Epoch %03d | Val MAE %.4f RMSE %.4f AUC %.4f RecallNZ %.4f | "
            "Test MAE %.4f RMSE %.4f AUC %.4f RecallNZ %.4f | mean|train_x-initial_gt| %.6f | Time %.2fs",
            epoch,
            metrics["val_mae"],
            metrics["val_rmse"],
            metrics["val_auc"],
            metrics["val_recall_non_zero"],
            metrics["test_mae"],
            metrics["test_rmse"],
            metrics["test_auc"],
            metrics["test_recall_non_zero"],
            mean_update,
            time.time() - epoch_start,
        )

        if tracked_cells:
            pred_cpu = gcn_pred_all.detach().cpu().clamp(0.0, 1.0)
            cls_prob_cpu = torch.sigmoid(cls_pred_all.detach()).cpu().clamp(0.0, 1.0)
            tracked_str = ", ".join(format_tracked_cells(pred_cpu, cls_prob_cpu, original_x, tracked_cells))
            logger.info("Tracked cells (row,col -> gt/reg/prob): %s", tracked_str)

        mae_improved = False
        auc_improved = False

        if best_mae_metrics is None:
            mae_improved = True
        else:
            mae_delta = metrics["val_mae"] - best_mae_metrics["val_mae"]
            if mae_delta < -args.mae_tolerance:
                mae_improved = True
            elif 0.0 <= mae_delta <= args.mae_tolerance:
                auc_delta = metrics["val_auc"] - best_mae_metrics["val_auc"]
                if auc_delta > args.auc_tolerance:
                    mae_improved = True

        if mae_improved:
            best_mae_metrics = metrics.copy()
            best_mae_snapshot = train_x.detach().cpu()

        if best_auc_metrics is None:
            auc_improved = True
        else:
            auc_delta = metrics["val_auc"] - best_auc_metrics["val_auc"]
            if auc_delta > args.auc_tolerance:
                auc_improved = True
            elif 0.0 <= -auc_delta <= args.auc_tolerance:
                mae_delta_for_auc = best_auc_metrics["val_mae"] - metrics["val_mae"]
                if mae_delta_for_auc > args.mae_tolerance:
                    auc_improved = True

        if auc_improved:
            best_auc_metrics = metrics.copy()
            best_auc_snapshot = train_x.detach().cpu()

        if mae_improved or auc_improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                if best_mae_snapshot is not None and not mae_snapshot_saved:
                    save_artifacts(args, best_mae_snapshot, data, logger)
                    mae_snapshot_saved = True
                if best_auc_snapshot is not None and not auc_snapshot_saved:
                    save_artifacts(args, best_auc_snapshot, data, logger, suffix="_bestAUC")
                    auc_snapshot_saved = True
                logger.info("Early stopping triggered (patience=%d).", args.patience)
                break

    if best_mae_snapshot is not None and not mae_snapshot_saved:
        save_artifacts(args, best_mae_snapshot, data, logger)
        mae_snapshot_saved = True
    if best_auc_snapshot is not None and not auc_snapshot_saved:
        save_artifacts(args, best_auc_snapshot, data, logger, suffix="_bestAUC")
        auc_snapshot_saved = True

    if best_mae_metrics:
        logger.info(
            "Best (MAE) Val MAE %.4f RMSE %.4f AUC %.4f | Test MAE %.4f RMSE %.4f AUC %.4f",
            best_mae_metrics["val_mae"],
            best_mae_metrics["val_rmse"],
            best_mae_metrics["val_auc"],
            best_mae_metrics["test_mae"],
            best_mae_metrics["test_rmse"],
            best_mae_metrics["test_auc"],
        )
    if best_auc_metrics:
        logger.info(
            "Best (AUC) Val MAE %.4f RMSE %.4f AUC %.4f | Test MAE %.4f RMSE %.4f AUC %.4f",
            best_auc_metrics["val_mae"],
            best_auc_metrics["val_rmse"],
            best_auc_metrics["val_auc"],
            best_auc_metrics["test_mae"],
            best_auc_metrics["test_rmse"],
            best_auc_metrics["test_auc"],
        )
    torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    args = parse_args()
    setup_signal_handlers()
    logger = build_logger(args)
    logger.info("Starting JEPA+AUC pipeline with args: %s", args)
    run(args, logger)


if __name__ == "__main__":
    main()
