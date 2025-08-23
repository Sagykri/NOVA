import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
import torch.nn.functional as F

def compute_block_distances(X1: torch.Tensor,
                            X2: torch.Tensor,
                            block_size: int,
                            metric: str,
                            same_label: bool) -> torch.Tensor:
    """
    Compute pairwise distances in blocks, returning a 1D tensor of all distances.
    """
    n1 = X1.size(0)
    blocks = []

    if not same_label:
        # Cross-label distances
        for start in range(0, n1, block_size):
            end = min(start + block_size, n1)
            if metric == 'euclidean':
                d = torch.cdist(X1[start:end], X2, p=2)
            else:
                d = 1 - X1[start:end] @ X2.t()
            blocks.append(d.view(-1))
    else:
        # Self-label distances (upper triangular)
        for b1 in range(0, n1, block_size):
            e1 = min(b1 + block_size, n1)
            for b2 in range(b1, n1, block_size):
                e2 = min(b2 + block_size, n1)
                if metric == 'euclidean':
                    d = torch.cdist(X1[b1:e1], X1[b2:e2], p=2)
                else:
                    d = 1 - X1[b1:e1] @ X1[b2:e2].t()
                if b1 == b2:
                    # Exclude diagonal
                    for i in range(d.size(0)):
                        blocks.append(d[i, i+1:].view(-1))
                else:
                    blocks.append(d.view(-1))

    # Concatenate all blocks into one vector
    return torch.cat(blocks)


def compute_stats_from_distances(d_all: torch.Tensor,
                                 full_stats: bool,
                                 percentiles: list) -> dict:
    """
    Compute summary statistics from a tensor of distances.
    If full_stats is False, only median is computed.
    """
    stats = {}
    if full_stats:
        try:
            # try GPU quantile
            qs = torch.tensor([p / 100 for p in percentiles], device=d_all.device)
            p = torch.quantile(d_all, qs)

        except RuntimeError as e:
            # fallback if tensor too large for torch.quantile on GPU
            vals = d_all.cpu().numpy()
            p = np.percentile(vals, percentiles)

        # fill in pXX entries dynamically
        for perc, val in zip(percentiles, p):
            stats[f"p{int(perc)}"] = float(val)

        # if p25 and p75 are available compute IQR whiskers
        if 25 in percentiles and 75 in percentiles:
            iqr = stats["p75"] - stats["p25"]
            stats["lower_whisker"] = stats["p25"] - 1.5 * iqr
            stats["upper_whisker"] = stats["p75"] + 1.5 * iqr
    else:
        # Only median (50th percentile) - faster computation
        try:
            median = torch.median(d_all).item()
        except RuntimeError:
            median = np.median(d_all.cpu().numpy())
        stats['p50'] = median

    return stats


def compute_label_pair_distances_stats(embeddings: np.ndarray,
                                       labels: np.ndarray,
                                       metric: str = 'euclidean',
                                       full_stats: bool = False,
                                       percentile_perc: list = [5, 10, 25, 50, 75, 90, 95],
                                       normalize_emb:bool=False) -> pd.DataFrame:
    """
    Compute distance statistics for every pair of labels.
    If full_stats=False, only median (p50) is computed for faster runtime.

    Returns:
        DataFrame with columns: label1, label2, block_size, total_pairs,
        dist_time_s, stats_time_s, p50 (and other percentiles if requested).
    """
    # Validate metric
    if metric not in ('euclidean'):
        print(f"Metric '{metric}' not recognized; using default 'euclidean'")
        metric = 'euclidean'

    # Select compute device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb = torch.from_numpy(embeddings).float().to(device)

    if normalize_emb:
        # Normalize embeddings if requested
        emb = F.normalize(emb, dim=-1)

    unique_labels = np.unique(labels)
    results = []

    # Iterate over pairs of labels
    for i, label1 in enumerate(unique_labels):
        idx1 = np.where(labels == label1)[0]
        X1 = emb[idx1]
        for label2 in unique_labels[i:]:
            idx2 = np.where(labels == label2)[0]
            X2 = emb[idx2]

            n1, n2 = X1.size(0), X2.size(0)
            total_pairs = int(n1 * n2 if label1 != label2 else n1 * (n1 - 1) // 2)

            # Determine block size
            same_label = (label1 == label2)
            block_size = get_block_size(n1, n2, device, X1, same_label)
            print(f"[START] {label1} ({n1}) vs {label2} ({n2}): block_size={block_size}")

            # Compute distances
            t0 = time.perf_counter()
            d_all = compute_block_distances(X1, X2, block_size, metric, same_label)
            dist_time = time.perf_counter() - t0
            print(f"[DIST] done in {dist_time:.3f}s; count={d_all.numel()}")

            # Compute statistics
            t1 = time.perf_counter()
            stats = compute_stats_from_distances(d_all, full_stats, percentile_perc)
            stats_time = time.perf_counter() - t1
            tag = 'FULL' if full_stats else 'MEDIAN'
            print(f"[{tag}] done in {stats_time:.3f}s")

            # Build result row
            row = {
                'label1': label1,
                'label2': label2,
                'block_size': block_size,
                'total_pairs': total_pairs,
                'dist_time_s': dist_time,
                'stats_time_s': stats_time,
                **stats
            }
            results.append(row)

            # Cleanup and free GPU memory
            del d_all
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return pd.DataFrame(results)

def get_block_size(n1: int,
                   n2: int,
                   device: torch.device,
                   X1: torch.Tensor,
                   same_label: bool) -> int:
    """
    Determine an optimal block size based on available GPU memory or fallback to full size on CPU.
    """
    if device.type == "cuda":
        try:
            # Query free GPU memory and compute safe block sizes
            free, _ = torch.cuda.mem_get_info()
            elem_size = X1.element_size()
            # For cross-label distances, limit by free memory
            cross = int(free * 0.5 // (elem_size * n2))
            # For same-label (self-pairs), consider square blocks
            selfp = int((free * 0.5 / elem_size) ** 0.5)
            # Choose block size between 1 and n1
            return max(1, min(n1, selfp if same_label else cross))
        except Exception:
            # On failure, process all at once
            return n1
    else:
        # On CPU, rely on system RAM and avoid chunking overhead
        return n1
  
def summarize_times(df: pd.DataFrame) -> None:
    """
    Print the maximum and total of dist_time_s and stats_time_s from the results DataFrame.
    """
    max_dist = df['dist_time_s'].max()
    total_dist = df['dist_time_s'].sum()
    mean_dist_time = df['dist_time_s'].mean()
    max_stats = df['stats_time_s'].max()
    total_stats = df['stats_time_s'].sum()
    mean_stats_time = df['stats_time_s'].mean()

    print(f"Max dist_time_s:   {max_dist:.3f}s")
    print(f"Total dist_time_s: {total_dist:.3f}s")
    print(f"Mean dist_time_s: {mean_dist_time:.3f}s")
    print(f"Max stats_time_s:  {max_stats:.3f}s")
    print(f"Total stats_time_s:{total_stats:.3f}s")
    print(f"Mean stats_time: {mean_stats_time:.3f}s")

def merge_batches_by_key(dfs, names, key_cols=('label1', 'label2'), value_col='p50'):
    """
    Merge a list of DataFrames on key_cols, renaming value_col in each with its corresponding name.
    
    Returns:
        - merged DataFrame
        - list of renamed value column names
    """
    merged = dfs[0][[*key_cols, value_col]].rename(columns={value_col: f"{value_col}_{names[0]}"})
    for df, name in zip(dfs[1:], names[1:]):
        merged = merged.merge(
            df[[*key_cols, value_col]].rename(columns={value_col: f"{value_col}_{name}"}),
            on=list(key_cols),
            how="inner"
        )
    value_cols = [f"{value_col}_{n}" for n in names]
    return merged, value_cols

def correlate_columns(df, cols, method='spearman'):
    """
    Compute correlation and p-value matrices for specified columns.

    Returns:
        - correlation DataFrame
        - p-value DataFrame
    """
    data = df[cols].to_numpy()

    if method == 'spearman':
        corr_mat, pval_mat = spearmanr(data, axis=0)
    elif method == 'pearson':
        corr_mat = np.corrcoef(data, rowvar=False)
        pval_mat = np.ones_like(corr_mat)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                r, p = pearsonr(data[:, i], data[:, j])
                corr_mat[i, j] = corr_mat[j, i] = r
                pval_mat[i, j] = pval_mat[j, i] = p
                pval_mat[i, i] = pval_mat[j, j] = 0.0
    else:
        raise ValueError(f"Unsupported method: {method}")

    return pd.DataFrame(corr_mat, index=cols, columns=cols), pd.DataFrame(pval_mat, index=cols, columns=cols)

def get_base_label(label: str) -> str:
    ## Get the base label by removing the last part after the last underscore
    ## This assumes labels are formatted as "base_rep" 
    ## e.g. "DAPI_rep1" -> "DAPI"
    return ('_').join(label.split('_')[0:-1])

def get_base_to_reps(labels):
    base_to_reps = {}
    unique_labels = np.unique(labels)
    # Group labels by base name
    for label in unique_labels:
        base = get_base_label(label)
        base_to_reps.setdefault(base, []).append(label)
    return base_to_reps