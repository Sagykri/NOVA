import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage
import networkx as nx
import warnings
  
def plot_umap(embeddings_umap, labels):
    labels = np.asarray(labels)
    unique_labels = sorted(np.unique(labels))
    palette = sns.color_palette('hls', len(unique_labels))
    label_to_color = dict(zip(unique_labels, palette))

    plt.figure()
    ax = sns.scatterplot(
        x=embeddings_umap[:, 0],
        y=embeddings_umap[:, 1],
        hue=labels,
        palette=label_to_color,
        s=10,
        alpha=0.7,
        linewidth=0,
        legend=False  # hide legend
    )

    # Add label annotations at the median UMAP position per class
    for label in unique_labels:
        coords = embeddings_umap[labels == label]
        if len(coords) == 0:
            continue
        x_median, y_median = np.median(coords, axis=0)
        plt.text(x_median, y_median, label,
                 fontsize=9, weight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', lw=0.5, alpha=0.6))

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP Projection of Embeddings')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_custom_boxplot(summary_df, med_col = "p50", q1_col = "p25", q3_col = "p75"):
    # Sort by median
    summary_df = summary_df.sort_values(by=med_col).reset_index(drop=True)
    plt.figure()
    for i, row in summary_df.iterrows():
        # Box
        plt.plot([i, i], [row[q1_col], row[q3_col]], color='black', linewidth=10)

        # Median
        plt.plot([i - 0.2, i + 0.2], [row[med_col], row[med_col]], color='red', linewidth=2)

        # Whiskers
        plt.plot([i, i], [row["lower_whisker"], row[q1_col]], color='black', linewidth=2)
        plt.plot([i, i], [row[q3_col], row["upper_whisker"]], color='black', linewidth=2)

        # Whisker caps
        plt.plot([i - 0.1, i + 0.1], [row["lower_whisker"], row["lower_whisker"]], color='black')
        plt.plot([i - 0.1, i + 0.1], [row["upper_whisker"], row["upper_whisker"]], color='black')

    plt.xticks(ticks=range(len(summary_df)), 
               labels=[f"{l1} vs {l2}" for l1, l2 in zip(summary_df['label1'], summary_df['label2'])], 
               rotation=45, ha='right')
    plt.ylabel("Distance")
    plt.title("Custom Box Plot of Pairwise Distances")
    plt.tight_layout()
    plt.show()

def plot_label_clustermap(df: pd.DataFrame,
                          metric_col: str = "p50",
                          method: str = "average",
                          cmap: str = "viridis",
                          figsize: tuple = (10,10),
                          save_path: str = None):
    """
    Build a symmetric distance matrix from df[['label1','label2',metric_col]],
    cluster it, and draw a seaborn clustermap.
    """
    labels = sorted(set(df['label1']) | set(df['label2']))
    dist_mat = pd.DataFrame(np.nan, index=labels, columns=labels)

    # Fill both off‐diag and diag entries from your df
    for _, row in df.iterrows():
        i, j = row['label1'], row['label2']
        v = row[metric_col]
        dist_mat.loc[i, j] = v
        dist_mat.loc[j, i] = v

    link = linkage(dist_mat.values, method=method)
    cg = sns.clustermap(dist_mat,
                        row_linkage=link, col_linkage=link,
                        cmap=cmap,
                        figsize=figsize)
    cg.fig.suptitle(f"Clustered heatmap ({metric_col})", y=1.02)
    if save_path:
        cg.savefig(save_path, bbox_inches='tight')

def plot_thresholded_network(df: pd.DataFrame,
                             metric_col: str = "p50",
                             threshold: float = None,
                             top_k: int = None,
                             figsize: tuple = (12,12),
                             node_size: int = 400,
                             edge_cmap: str = "coolwarm",
                             cluster_map: dict = None,
                             save_path: str = None):
    """
    Build a graph whose nodes are labels and whose edges connect pairs
    with metric_col <= threshold (or keep only top_k smallest distances).
    Draw with edge widths proportional to closeness.

    Optional cluster_map: { node_label -> cluster_id } for per‑cluster node colors.
    """
    if (threshold is None) == (top_k is None):
        raise ValueError("Specify exactly one of threshold or top_k")

    # 1) Filter edges
    edges = df[['label1','label2',metric_col]].copy()
    if threshold is not None:
        edges = edges[edges[metric_col] <= threshold]
    else:
        edges = edges.nsmallest(top_k, metric_col)

    # 2) Build graph and strip suffix
    all_labels = sorted(set(df['label1']) | set(df['label2']))
    G = nx.Graph()
    G.add_nodes_from(all_labels)
    for _, row in edges.iterrows():
        G.add_edge(row['label1'], row['label2'], weight=row[metric_col])

    # 3) Layout: more repulsion & iterations
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=300)

    # 4) Node colors: by cluster if provided, else uniform
    if cluster_map:
        # map each node to its cluster color
        clusters = np.array([cluster_map.get(n, -1) for n in G.nodes()])
        # assign each unique cluster an integer color index
        _, labels_idx = np.unique(clusters, return_inverse=True)
        node_colors = labels_idx
        cmap_nodes = plt.get_cmap("tab20")
    else:
        node_colors = "lightblue"
        cmap_nodes = None

    # 5) Compute edge widths & colors as before
    weights = np.array([G[u][v]['weight'] for u,v in G.edges()])
    inv = (weights.max() - weights) + 0.1*weights.max()
    widths = inv / inv.max() * 6
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    edge_cmap = plt.get_cmap(edge_cmap)
    edge_colors = edge_cmap(norm(weights))

    # 6) Plot
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_size,
                           node_color=node_colors,
                           cmap=cmap_nodes,
                           ax=ax)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        nx.draw_networkx_edges(G, pos,
                               width=widths,
                               edge_color=edge_colors,
                               alpha=0.7,
                               ax=ax)

    # 7) Draw labels with bounding boxes
    for n, p in pos.items():
        ax.text(p[0], p[1], n,
                fontsize=8,
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))

    # 8) Colorbar for edges
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label(metric_col)

    title = (f"Thresholded network (≤ {threshold:.3g})" if threshold is not None
             else f"Top {top_k} closest pairs")
    ax.set_title(title)
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

def compare_euclidean_cosine(df_euclid: pd.DataFrame,
                     df_cosine: pd.DataFrame,
                     column: str):
    """
    Given two DataFrames with the same index and a distance column,
    prints max and mean absolute difference between Euclid²/2 and cosine,
    and shows a scatter plot comparing them.
    """
    # Prepare comparison DataFrame
    euclid_sq_over2 = df_euclid[column].pow(2).div(2)
    cosine_dist = df_cosine[column]
    cmp = pd.DataFrame({
        'euclid_sq_over2': euclid_sq_over2,
        'cosine_dist':     cosine_dist
    })
    # Compute differences
    diff = (cmp['euclid_sq_over2'] - cmp['cosine_dist']).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    # Print summary
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    
    # Scatter plot
    plt.figure()
    plt.scatter(cmp['euclid_sq_over2'], cmp['cosine_dist'], s=5)
    plt.xlabel('Euclid² / 2')
    plt.ylabel('Cosine distance')
    plt.title('Comparison of Euclid²/2 vs. Cosine distance')
    # Diagonal reference line
    min_val = min(cmp.min())
    max_val = max(cmp.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    plt.show()