import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc

import pickle
import os
import scipy.sparse as sp
import subprocess

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, AttributedGraphDataset, Reddit, HeterophilousGraphDataset
from torch_geometric.datasets.elliptic import EllipticBitcoinDataset
from torch_geometric.datasets.gdelt_lite import GDELTLite
from torch_geometric.utils import to_dense_adj
import numpy as np
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import math
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, coalesce
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors

from torch_geometric.data import Data
import hnswlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import pickle
import scipy.sparse as sp


def plot_prediction_histogram(scores: torch.Tensor,
                              labels: torch.Tensor,
                              epoch: int,
                              dataset: str,
                              save_dir: str = "./hist_plots/",
                              lower_pct: float = 2.0,
                              upper_pct: float = 98.0):
    """
    Plot histograms of scores for true labels 0 and 1, using a dual-axis
    layout (left y-axis for true=0, right y-axis for true=1).

    Args:
      - scores: torch.Tensor, shape=(N,), prediction scores, assumed clamped to [0, 1].
      - labels: torch.Tensor, shape=(N,), ground-truth 0/1 labels (will be rounded to float).
      - epoch: int, current epoch, used for title and filename.
      - dataset: str, dataset name, used to distinguish files.
      - save_dir: str, directory to save figures (created if needed).
      - lower_pct: float in [0, 100], lower percentile to clip scores (default 2.0).
      - upper_pct: float in [0, 100], upper percentile to clip scores (default 98.0).
    """
    # --- 1. Convert to numpy arrays and separate two classes ---
    scores_np = scores.detach().cpu().numpy().ravel()
    labels_np = labels.detach().cpu().numpy().ravel()

    zero_scores = scores_np[labels_np == 0]
    one_scores  = scores_np[labels_np == 1]

    if zero_scores.size == 0 and one_scores.size == 0:
        print(f"[Epoch {epoch}] No samples to plot.")
        return

    # --- 2. Remove outliers: keep scores within [lower_pct%, upper_pct%] ---
    all_scores = scores_np
    low_val  = np.percentile(all_scores, lower_pct)
    high_val = np.percentile(all_scores, upper_pct)
    if high_val <= low_val:
        low_val  = max(0.0, low_val - 0.01)
        high_val = min(1.0, high_val + 0.01)

    zero_inlier = zero_scores[(zero_scores >= low_val) & (zero_scores <= high_val)]
    one_inlier  = one_scores[(one_scores  >= low_val) & (one_scores  <= high_val)]

    # --- 3. Create output directory and filename ---
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"pred_hist_{dataset}_epoch_{epoch:03d}.png")

    # --- 4. Set up figure with dual y-axes ---
    fig, ax_left = plt.subplots(figsize=(6, 4))
    ax_right = ax_left.twinx()  # independent y-axis on the right

    # Use the same bins so x-axis aligns
    bins = np.linspace(low_val, high_val, 51)  # 50 bins

    # --- 5. Plot histogram for true=0 on the left axis ---
    if zero_inlier.size > 0:
        ax_left.hist(
            zero_inlier,
            bins=bins,
            color="C0",
            alpha=0.6,
            label="true=0",
            edgecolor="none"
        )

    # --- 6. Plot histogram for true=1 on the right axis ---
    if one_inlier.size > 0:
        ax_right.hist(
            one_inlier,
            bins=bins,
            color="C1",
            alpha=0.6,
            label="true=1",
            edgecolor="none"
        )

    # --- 7. Set axis labels and colors ---
    ax_left.set_xlabel("Predicted score (inlier range)")
    ax_left.set_ylabel("Frequency (true=0)", color="C0")
    ax_left.tick_params(axis="y", labelcolor="C0")

    ax_right.set_ylabel("Frequency (true=1)", color="C1")
    ax_right.tick_params(axis="y", labelcolor="C1")

    # --- 8. Show number of removed outliers in title ---
    num_below = np.sum(all_scores < low_val)
    num_above = np.sum(all_scores > high_val)
    title = (f"{dataset}: Prediction Distribution (Epoch {epoch})\n"
             f"Ignored <{lower_pct}pct, >{upper_pct}pct → "
             f"({num_below} + {num_above} samples)")
    ax_left.set_title(title)

    # --- 9. Add legends for both y-axes by merging handles/labels ---
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")

    # --- 10. Adjust layout and save ---
    fig.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)

    print(f"[Epoch {epoch}] Prediction histogram saved to {fname} "
          f"(inlier range: [{low_val:.4f}, {high_val:.4f}])")


def plot_auc_illustration(scores: torch.Tensor,
                          labels: torch.Tensor,
                          epoch: int,
                          dataset: str,
                          save_dir: str = "./auc_plots/"):
    """
    Plot the ROC curve given model scores and ground-truth labels, and fill
    the area between the ROC curve and y=1 to visualize 1 - AUC (depending
    on coloring). The output filename contains both dataset and epoch and is
    stored under save_dir.

    Args:
      - scores: torch.Tensor, shape=(N,), prediction scores (higher → more likely label=1), no clamp required.
      - labels: torch.Tensor, shape=(N,), 0/1 ground-truth labels (rounded for safety).
      - epoch: int, current epoch, used for filename and plot title.
      - dataset: str, dataset name, used to distinguish outputs.
      - save_dir: str, directory to save the plot, default "./auc_plots/" (created if needed).
    """
    # Convert to numpy and ensure 1D float arrays
    scores_np = scores.detach().cpu().numpy().ravel()
    labels_np = labels.detach().cpu().numpy().ravel().astype(int)

    # 1. Compute ROC curve: fpr (False Positive Rate), tpr (True Positive Rate)
    fpr, tpr, _ = roc_curve(labels_np, scores_np)
    # Compute AUC value
    roc_auc = auc(fpr, tpr)

    # 2. Create output directory and filename
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"auc_{dataset}_epoch_{epoch:03d}.png")

    # 3. Plot ROC curve and shaded region
    plt.figure(figsize=(6, 5))
    # 3.1 Draw a horizontal line y = 1 (top boundary)
    plt.plot([0, 1], [1, 1], color="gray", linestyle="--", linewidth=1)

    # 3.2 Plot ROC curve
    plt.plot(fpr, tpr, color="blue", linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")

    # 3.3 (Optional) mark a particular breakpoint on ROC curve,
    #     e.g. the point closest to (0.5, 0.5). Currently commented out.
    # idx = np.argmin(np.abs(fpr - 0.5) + np.abs(tpr - 0.5))
    # x_break, y_break = fpr[idx], tpr[idx]
    # plt.axvline(x=x_break, ymin=0, ymax=y_break, color="gray", linestyle="--", linewidth=1)
    # plt.axhline(y=y_break, xmin=0, xmax=x_break, color="gray", linestyle="--", linewidth=1)
    # plt.scatter([x_break], [y_break], color="black", s=20)

    # 3.4 Fill the area between ROC curve and y=1
    plt.fill_between(fpr, tpr, 1.0, where=(tpr <= 1.0), color="red", alpha=0.3, hatch="///")

    # 4. Set axes limits, labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset}: ROC Curve (Epoch {epoch})")
    plt.legend(loc="lower right")

    # 5. Hide top/right spines for a cleaner look
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 6. Save plot and close figure
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f"[Epoch {epoch}] AUC illustration saved to {fname} (AUC = {roc_auc:.4f})")

def run_classification(filled_x,
                       edge_index,
                       labels,
                       eva_script_path: str,
                       dataset: str,
                       method: str,
                       train_fts_ratio: float,
                       c: float):
    """
    - filled_x: torch.Tensor or np.ndarray, [N, D]
    - edge_index: torch.LongTensor or np.ndarray, [2, E]
    - labels: torch.Tensor or np.ndarray, [N,]
    - eva_script_path: full path to evaluation script, e.g. "/home/.../eva_classfication_AX.py"
    - dataset, method, train_fts_ratio, c: correspond directly to argparse arguments above
    """
    # 1. Convert to numpy
    if isinstance(filled_x, torch.Tensor):
        filled_x = filled_x.cpu().numpy()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    N, D = filled_x.shape

    # 2. Prepare directories
    feat_dir = os.path.join('features', method)
    os.makedirs(feat_dir, exist_ok=True)
    data_dir = os.path.join('data', dataset)
    os.makedirs(data_dir, exist_ok=True)

    # 3. Save filled features
    f_feat = os.path.join(
        feat_dir,
        f'gene_fts_train_ratio_{dataset}_{train_fts_ratio}_G1.0_R1.0_C{c}.pkl'
    )
    with open(f_feat, 'wb') as f:
        pickle.dump(filled_x, f)

    # 4. Save test indices (here all nodes)
    idx = np.arange(N, dtype=np.int64)
    f_idx = os.path.join(feat_dir, f'{dataset}_{train_fts_ratio}_test_fts_idx.pkl')
    with open(f_idx, 'wb') as f:
        pickle.dump(idx, f)

    # 5. Save adjacency
    rows, cols = edge_index
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    f_adj = os.path.join(feat_dir, f'{dataset}_sp_adj.pkl')
    with open(f_adj, 'wb') as f:
        pickle.dump(adj, f)

    # 6. Save labels
    f_lbl = os.path.join(data_dir, f'{dataset}_labels.pkl')
    with open(f_lbl, 'wb') as f:
        pickle.dump(labels, f)

    # 7. Call eva_classfication_AX.py
    cmd = [
        'python', eva_script_path,
        '--method_name',     method,
        '--dataset',         dataset,
        '--train_fts_ratio', str(train_fts_ratio),
        '--c',               str(c)
    ]
    subprocess.run(cmd, check=True)



    
def load_arxiv(root):
    """
    Load the Arxiv dataset (OGB format) and return a torch_geometric.data.Data object.
    """
    features = torch.from_numpy(np.loadtxt(f'{root}/arxiv/raw/node-feat.csv.gz', delimiter=',', dtype=np.float32))
    labels = torch.from_numpy(np.loadtxt(f'{root}/arxiv/raw/node-label.csv.gz', delimiter=',', dtype=np.int64))
    edge_index = torch.from_numpy(np.loadtxt(f'{root}/arxiv/raw/edge.csv.gz', delimiter=',', dtype=np.int64))
    edge_index = to_undirected(edge_index.t())
    return Data(x=features, y=labels, edge_index=edge_index)

def load_dataset(dataset_name="Cornell"):
    if dataset_name in ["Texas", "Wisconsin", "Cornell"]:
        dataset = WebKB(root='./data', name=dataset_name)
    elif dataset_name in ["Reddit"]:
        dataset = Reddit(root='./data')
    elif dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root='./data', name=dataset_name)
    elif dataset_name in ["Computers", "Photo"]:
        dataset = Amazon(root='./data', name=dataset_name)
    elif dataset_name in ["CS", "Physics"]:
        dataset = Coauthor(root='./data', name=dataset_name)
    elif dataset_name in ["BlogCatalog", "PPI", "Flickr", "Facebook", "TWeibo"]:
        dataset = AttributedGraphDataset(root='./data', name=dataset_name)
    elif dataset_name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
        dataset = HeterophilousGraphDataset(root='./data/hetero', name=dataset_name)
    elif dataset_name == "Arxiv":  # newly added Arxiv dataset support
        dataset = load_arxiv(root='./data')
    # add EllipticBitcoinDataset dataset
    elif dataset_name == "EllipticBitcoin":
        dataset = EllipticBitcoinDataset(root='./data')
    elif dataset_name == "GDELTLite":
        dataset = GDELTLite(root='./data')
    else:
        raise ValueError("Unsupported dataset")
    print(f"Loaded dataset: {dataset_name}")
    return dataset

# Remaining helper functions (knn_impute_sklearn, knn_impute_vectorized, add_knn_edges, ...)
# -------------  Keep original implementation unchanged -------------

def knn_impute_sklearn(train_x):
    train_x_np = train_x.cpu().detach().numpy()
    imputer = KNNImputer(n_neighbors=5)
    x_filled_np = imputer.fit_transform(train_x_np)
    return torch.tensor(x_filled_np, dtype=train_x.dtype, device=train_x.device)

def knn_impute_aknn(train_x: torch.Tensor,
                    n_neighbors: int = 5,
                    ef_construction: int = 100,
                    ef_search: int = 20) -> torch.Tensor:

    # 1) Convert to numpy
    X = train_x.cpu().detach().numpy().astype(np.float32)    # (n, d)
    mask_obs = ~np.isnan(X)                                  # True means observed values
    col_mean = np.nanmean(X, axis=0)
    X_init = np.where(mask_obs, X, col_mean)                 # (n, d)

    n, d = X_init.shape
    p = hnswlib.Index(space='l2', dim=d)
    p.init_index(max_elements=n, ef_construction=ef_construction, M=16)
    p.add_items(X_init)
    p.set_ef(ef_search)
    labels, _ = p.knn_query(X_init, k=n_neighbors + 1)       
    neigh_idx = labels[:, 1:]                               # (n, k)
    neigh_feats = X_init[neigh_idx]                         # (n, k, d)
    mean_neigh = neigh_feats.mean(axis=1)                   # (n, d)
    X_filled = X_init.copy()
    X_filled[~mask_obs] = mean_neigh[~mask_obs]

    return torch.from_numpy(X_filled).to(train_x.device).type_as(train_x)

def knn_impute_vectorized(stage_filled, mask, n_neighbors=5):
    stage_filled = stage_filled.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()
    num_nodes, _ = stage_filled.shape
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(stage_filled)
    _, indices = nbrs.kneighbors(stage_filled)
    imputed = stage_filled.copy()
    for i in range(num_nodes):
        missing = ~mask[i]
        if missing.any():
            neigh_vals = stage_filled[indices[i]]
            imputed[i, missing] = neigh_vals[:, missing].mean(axis=0)
    return imputed

def knn_impute_aknnbiased(stage_filled: torch.Tensor,
                          mask: torch.Tensor,
                          n_neighbors: int = 5,
                          ef_construction: int = 100,
                          ef_search: int = 20) -> torch.Tensor:
    """
    Use hnswlib to perform approximate KNN search, greatly accelerating the
    NearestNeighbors.kneighbors step, while keeping the rest of the logic
    identical to the original vectorized imputation.
    """
    # 1) Convert to numpy
    X = stage_filled.cpu().detach().numpy().astype(np.float32)  # (n, d)
    M = mask.cpu().detach().numpy().astype(bool)               # (n, d)
    n, d = X.shape

    # 2) Build hnswlib index
    p = hnswlib.Index(space='l2', dim=d)
    p.init_index(max_elements=n, ef_construction=ef_construction, M=16)
    p.add_items(X)
    p.set_ef(ef_search)  # trade-off: larger → more accurate but slower

    # 3) aKNN query
    labels, distances = p.knn_query(X, k=n_neighbors)  # labels.shape == (n, k)

    # 4) Vectorized mean imputation (same as before)
    neigh_feats = X[labels]             # (n, k, d)
    mean_neigh = neigh_feats.mean(axis=1)  # (n, d)
    X_imputed = X.copy()
    X_imputed[~M] = mean_neigh[~M]

    # 5) Convert back to torch
    return torch.from_numpy(X_imputed).to(stage_filled.device).type_as(stage_filled)

def add_knn_edges(x, edge_index, k, compute_device=None):
    output_device = edge_index.device
    if compute_device is not None:
        compute_device = torch.device(compute_device)
        x_work = x.to(compute_device)
        edge_index_work = edge_index.to(compute_device)
    else:
        x_work = x
        edge_index_work = edge_index

    knn_edge_index = knn_graph(x_work, k=k, loop=False)
    knn_edge_index = to_undirected(knn_edge_index)
    new_edge_index = torch.cat([edge_index_work, knn_edge_index], dim=1)
    new_edge_index, _ = coalesce(new_edge_index, None, x_work.size(0), x_work.size(0))
    if new_edge_index.device != output_device:
        new_edge_index = new_edge_index.to(output_device)
    return new_edge_index

def compute_full_correlation_matrix(features, variance_threshold=1e-8, device="cuda:1"):
    num_features = features.shape[1]
    var = np.var(features, axis=0)
    idx_keep = np.where(var > variance_threshold)[0]
    filt = features[:, idx_keep]
    T = torch.tensor(filt, dtype=torch.float32).to(device)
    mu = T.mean(0, keepdim=True)
    C = (T - mu).T @ (T - mu) / (T.shape[0] - 1)
    s = torch.sqrt(torch.diag(C)).unsqueeze(0)
    corr = torch.clamp(C / (s.T @ s), -1, 1).cpu().numpy()
    full = np.zeros((num_features, num_features), dtype=corr.dtype)
    full[np.ix_(idx_keep, idx_keep)] = corr

    return full

def select_topk_features_per_feature(corr_matrix, k=50):
    topk = []
    for i in range(corr_matrix.shape[0]):
        order = np.argsort(-np.abs(corr_matrix[i]))
        order = order[order != i][:k]
        topk.append(order.tolist())
    return topk

def filter_edges(x, edge_index, k_all):
    num_nodes = x.size(0)
    x_np = x.cpu().numpy()
    e_np = edge_index.cpu().numpy()
    nbrs = {i: [] for i in range(num_nodes)}
    for u, v in e_np.T:
        nbrs[u].append(v)
        nbrs[v].append(u)
    top = {}
    for i in range(num_nodes):
        vs = nbrs[i]
        if len(vs) <= k_all:
            top[i] = set(vs)
        else:
            d = np.linalg.norm(x_np[vs] - x_np[i], axis=1)
            sel = np.argsort(d)[:k_all]
            top[i] = set(np.array(vs)[sel])
    new_edges = {(min(u,v), max(u,v)) for u,v in e_np.T if (v in top[u]) or (u in top[v])}
    new_edge_index = torch.tensor(sorted(new_edges), dtype=torch.long).t().contiguous()
    return new_edge_index

def annealing_weights(epoch, total, eps=0.001):
    if epoch >= total-2:
        return 0., 1.
    beta = math.log(1/eps)/total
    w_mid = math.exp(-beta*epoch)
    return w_mid, 1-w_mid

def compute_corr(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the metric defined in Eq. (19):
      corr = (1/d) * sum_j [ 1 - sum_i (pred_ij - true_ij)^2 / sum_i (true_ij - true_mean_j)^2 ]
    pred, true: shape [N, d]
    Returns a scalar; larger is better.
    """
    # Per-feature ground-truth mean (shape [d])
    true_mean = true.mean(dim=0)
    # Denominator: sum_i (x_ij - mean_j)^2  shape [d]
    ss_tot = ((true - true_mean) ** 2).sum(dim=0)
    # Numerator: sum_i (pred_ij - x_ij)^2      shape [d]
    ss_res = ((pred - true) ** 2).sum(dim=0)
    # Per-dimension corr_j = 1 - ss_res/ss_tot
    corr_per_dim = 1 - ss_res / (ss_tot + eps)
    # Average across dimensions and return
    return corr_per_dim.mean()

def soft_clamp(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    # scaled softplus:  softplus(z;β) = (1/β) * log(1 + exp(β z))
    sp_pos = F.softplus(beta * (x - 1.0)) / beta
    sp_neg = F.softplus(-beta * x)        / beta
    return x - sp_pos + sp_neg


def plot_error_cdf(predictions: torch.Tensor,
                   targets: torch.Tensor,
                   mask: torch.BoolTensor,
                   epoch: int,
                   save_dir: str = "./cdf_plots/dimupdate_aknn_batch/"):

    # Extract absolute errors and flatten
    errors = (predictions[mask] - targets[mask]).abs().view(-1).cpu().numpy()
    if errors.size == 0:
        print(f"[Epoch {epoch}] No errors")
        return

    # Sort errors and compute empirical CDF
    sorted_err = np.sort(errors)
    cdf = np.linspace(0, 1, len(sorted_err), endpoint=True)

    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"gcngm_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.dataset}_error_cdf_epoch_{epoch}.png")

    # Plot CDF
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_err, cdf)
    plt.xlabel("Absolute Error")
    plt.ylabel("Empirical CDF")
    plt.title(f"Test Error CDF (Epoch {epoch})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[Epoch {epoch}] CDF Saved to {fname}")

def plot_error_bar(predictions: torch.Tensor,
                   targets: torch.Tensor,
                   mask: torch.BoolTensor,
                   epoch: int,
                   save_dir: str = "./error_bar_plots/dimupdate_aknn_batch/",
                   max_bars: int = None):

    # Extract absolute errors and flatten
    errors = (predictions[mask] - targets[mask]).abs().view(-1).cpu().numpy()
    if errors.size == 0:
        print(f"[Epoch {epoch}] No errors")
        return

    # Sort errors
    sorted_err = np.sort(errors)
    if max_bars is not None and len(sorted_err) > max_bars:
        # Optionally keep only the largest max_bars errors (or the smallest, depending on use)
        sorted_err = sorted_err[-max_bars:]

    # Generate x-axis indices
    x = np.arange(len(sorted_err))

    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"gcngm_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.dataset}_error_cdf_epoch_{epoch}.png")

    # Plot bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(x, sorted_err)
    plt.xlabel("Sample index (sorted)")
    plt.ylabel("Absolute Error")
    plt.title(f"Sorted Test Errors (Epoch {epoch})")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[Epoch {epoch}] Bar chart saved to: {fname}")

def log_tail_quantile_errors(errors: np.ndarray, epoch: int, logger=None):
    """
    Log a compact one-line summary of tail quantile errors (Test Set),
    with indentation so it fits neatly into log output.
    """
    quantiles = [50, 75, 90, 92, 94, 95, 96, 97, 98, 99, 100]
    q_vals = np.percentile(errors, quantiles)

    # Build output string
    parts = [f"{q}p:{val:.5f}" for q, val in zip(quantiles, q_vals)]
    line = "\t[Q-Err] " + " | ".join(parts)

    if logger:
        logger.info(line)
    else:
        print(line)

def find_best_threshold(scores: torch.Tensor,
                        labels: torch.Tensor,
                        mode: str = "acc") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given scores in [0, 1] and binary labels, scan thresholds in [0.0, 1.0]
    (101 points) and find the threshold that maximizes the chosen metric.

    Args:
      - scores: torch.Tensor, shape=(N,), continuous prediction scores, expected clamped to [0, 1].
      - labels: torch.Tensor, shape=(N,), binary 0/1 labels; will be rounded for safety.
      - mode: str, "acc" or "precision":
          * "acc": maximize Accuracy = (preds == labels).mean()
          * "precision": maximize Precision(1) = TP / (TP + FP)

    Returns:
      - best_thresh: torch.Tensor, threshold that maximizes the chosen metric.
      - best_metric: torch.Tensor, the maximum value of the chosen metric.
    """
    labels = labels.detach().clone().round().float()
    scores = scores.detach().clone()
    device = scores.device

    thresholds = torch.linspace(0.0, 1.0, steps=101, device=device)
    best_thresh = torch.tensor(0.0, device=device)
    best_metric = torch.tensor(0.0, device=device)

    for t in thresholds:
        preds = (scores >= t).float()  # 0/1 predictions

        if mode == "acc":
            # Accuracy = (TP + TN) / N
            metric_val = (preds == labels).float().mean()

        elif mode == "precision":
            # Precision(1) = TP / (TP + FP)
            pred_positives = preds.sum()  # TP + FP
            if pred_positives.item() == 0:
                metric_val = torch.tensor(0.0, device=device)
            else:
                true_positive = (preds * labels).sum()
                metric_val = true_positive / pred_positives

        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'acc' or 'precision'.")

        if metric_val > best_metric:
            best_metric = metric_val
            best_thresh = t

    return best_thresh, best_metric

def l2_norm(grads):
    return torch.sqrt(sum((g.detach()**2).sum() for g in grads if g is not None))
