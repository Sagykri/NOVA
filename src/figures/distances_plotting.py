import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage
import networkx as nx
import warnings
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Added by Nancy
from itertools import combinations
from matplotlib.cm import get_cmap


plt.rcParams.update({
        'figure.dpi': 100,
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
    })

def plot_custom_boxplot(summary_df,
                      med_col="p50",
                      q1_col="p25",
                      q3_col="p75",
                      lw=1.5,
                      width=0.6):
    # sort by median
    summary_df = summary_df.sort_values(by=med_col).reset_index(drop=True)

    # build stats list
    stats = []
    for row in summary_df.itertuples(index=False):
        stats.append({
            "label":  f"{row.label1} vs {row.label2}",
            "med":    getattr(row, med_col),
            "q1":     getattr(row, q1_col),
            "q3":     getattr(row, q3_col),
            "whislo": row.lower_whisker,
            "whishi": row.upper_whisker,
            "fliers": []
        })

    fig, ax = plt.subplots(figsize=(max(8, len(stats)*0.3), 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.bxp(
        stats,
        patch_artist=True,         
        widths=width,
        showmeans=False,
        showfliers=False,
        whiskerprops=dict(color="black", linewidth=lw),
        capprops    =dict(color="black", linewidth=lw),
        boxprops    =dict(facecolor="lightgray",
                          edgecolor="black",
                          linewidth=lw,
                          alpha=0.7),
        medianprops =dict(color="red", linewidth=lw),
    )

    # tidy up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticklabels([s["label"] for s in stats],
                       rotation=90, fontsize=8)
    for tick, row in zip(ax.get_xticklabels(), summary_df.itertuples(index=False)):
        if row.label1 == row.label2:
            tick.set_fontweight("bold")
    ax.set_ylabel("Distance")
    ax.set_title("Pairwise Distance Distribution")
    plt.tight_layout()
    plt.show()

def plot_label_clustermap(df: pd.DataFrame,
                          metric_col: str = "p50",
                          method: str = "average",
                          cmap: str = "viridis",
                          figsize: tuple = (10,10),
                          highlight_thresh: float = None,
                          save_path: str = None):
    """
    Build a symmetric distance matrix from df[['label1','label2',metric_col]],
    cluster it, and draw a seaborn clustermap, and (if highlight_thresh is set)
    draw a red box around any cell with value ≤ highlight_thresh.
    """
    labels = sorted(set(df['label1']) | set(df['label2']))
    dist_mat = pd.DataFrame(np.nan, index=labels, columns=labels)
    
    # Fill both off‐diag and diag entries from your df
    for _, row in df.iterrows():
        i, j = row['label1'], row['label2']
        v = row[metric_col]
        # Nancy's comment: note, this uses the last value and DOES NOT aggregate values from same i,j
        dist_mat.loc[i, j] = v
        dist_mat.loc[j, i] = v

    #dist_mat = dist_mat.fillna(0) # Added by Nancy - to support diff number of reps in each cell line
    
    link = linkage(dist_mat.values, method=method)
    cg = sns.clustermap(dist_mat,
                        row_linkage=link, col_linkage=link,
                        cmap=cmap,
                        figsize=figsize)
    cg.fig.suptitle(f"Clustered heatmap ({metric_col})", y=1.02)
    # Highlight low‑distance cells
    if highlight_thresh is not None:
        # Map original matrix indices to clustered heatmap positions
        order = cg.dendrogram_row.reordered_ind
        inv_map = {orig: new for new, orig in enumerate(order)}
        ax = cg.ax_heatmap
        data = dist_mat.values
        n = data.shape[0]
        for i in range(n):
            for j in range(n):
                if data[i, j] <= highlight_thresh:
                    yi, xi = inv_map[i], inv_map[j]
                    ax.add_patch(Rectangle((xi, yi), 1, 1,
                                           fill=False,
                                           edgecolor='red',
                                           linewidth=1))
    if save_path:
        cg.savefig(save_path, bbox_inches='tight')

def plot_network1(df: pd.DataFrame,
                             metric_col: str = "p50",
                             threshold: float = None,
                             top_k: int = None,
                             figsize: tuple = (12,12),
                             node_size: int = 400,
                             edge_cmap: str = "coolwarm",
                             save_path: str = None,
                            method = 'spring'):
    """
    Build a graph whose nodes are labels and whose edges connect pairs
    with metric_col <= threshold (or keep only top_k smallest distances).
    Draw with edge widths proportional to closeness.

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

    # 3) Layout
    if method == 'spring':
        print('spring')
        pos = nx.spring_layout(G, seed=42, k=0.8, iterations=300)
    else:
        print('kamada')
        pos = nx.kamada_kawai_layout(G, weight='length', scale=1.0, center=(0,0))

    # 4) Node colors
    node_colors = "lightblue"
    cmap_nodes = None

    # 5) Compute edge widths and colors as before
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

def plot_cluster_proximity_network(df_stats,
                                   metric='p50',
                                   intra_metric='p50',
                                   threshold=None,
                                   top_k=None,
                                   figsize=(10, 10),
                                   min_node_size=200,
                                   max_node_size=1000,
                                   cmap_name='viridis',
                                   edge_width=2,
                                   layout_method='kamada_kawai'):
    """
    Visualise cluster proximity for publication:

    - Node size proportional to the intra‑cluster median value (intra_metric).
    - Edge colour and transparency proportional to the inter‑cluster median (metric),
      with closer clusters rendered in darker, more opaque hues.
    - Node positions computed to respect median distances via specified layout.
    - Optionally retain only edges below a threshold or the top_k closest pairs.
    """
    intra_df = df_stats[df_stats.label1 == df_stats.label2].set_index('label1')
    inter_df = df_stats[df_stats.label1 != df_stats.label2].copy()

    if threshold is not None:
        inter_df = inter_df[inter_df[metric] <= threshold]
    elif top_k is not None:
        inter_df = inter_df.nsmallest(top_k, metric)

    G = nx.Graph()
    G.add_nodes_from(intra_df.index)
    for _, row in inter_df.iterrows():
        a, b = row['label1'], row['label2']
        d = row[metric]
        G.add_edge(a, b, length=d, median=d)

    intra_vals = np.array([intra_df.loc[n, intra_metric] for n in G.nodes()])
    norm_intra = (intra_vals - intra_vals.min()) / (intra_vals.ptp() + 1e-6)
    node_sizes = norm_intra * (max_node_size - min_node_size) + min_node_size

    medians = np.array([G[u][v]['median'] for u, v in G.edges()])
    norm_med = (medians - medians.min()) / (medians.ptp() + 1e-6)
    cmap = plt.get_cmap(cmap_name)
    edge_colors = cmap(norm_med)
    edge_alphas = 1.0 - norm_med

    if layout_method == 'spring':
        pos = nx.spring_layout(G, weight='length', seed=42, k=0.8, iterations=300)
    else:
        pos = nx.kamada_kawai_layout(G, weight='length', scale=1.0, center=(0,0))

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color='lightsteelblue',
                           edgecolors='black',
                           linewidths=0.5,
                           ax=ax)
    for (u, v), col, alpha in zip(G.edges(), edge_colors, edge_alphas):
        ax.plot([pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color=col, alpha=alpha, linewidth=edge_width)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family='serif', ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=medians.min(),
                                                  vmax=medians.max()))
    sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01)
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.03, fraction=0.05)
    cbar.set_label(metric, family='serif')

    # Legend for node sizes (intra-cluster medians)
    sample_vals = np.percentile(intra_vals, [0, 50, 100])
    sample_sizes = ((sample_vals - intra_vals.min()) / (intra_vals.ptp() + 1e-6)
                   * (max_node_size - min_node_size) + min_node_size)
    handles = [
        Line2D([0], [0],
               marker='o',
               color='lightsteelblue',
               markeredgecolor='black',
               markersize=np.sqrt(sz),
               linewidth=0)
        for sz in sample_sizes
    ]
    labels = [f"{val:.2f}" for val in sample_vals]
    legend = ax.legend(
        handles, labels,
        title="Intra‑cluster median",
        frameon=True,
        framealpha=0.9,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5)
    )
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)

    if threshold is not None:
        title = f"Clusters with {metric} ≤ {threshold:.3g}"
    elif top_k is not None:
        title = f"Top {top_k} Closest Cluster Pairs"
    else:
        title = "Cluster Proximity Network"
    ax.set_title(title, family='serif')
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    ax.set_xlim(min(x_vals) - 0.15, max(x_vals) + 0.15)
    ax.set_ylim(min(y_vals) - 0.15, max(y_vals) + 0.15)

    ax.axis('off')
    plt.tight_layout()
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

def plot_dist_histogram(df, metric='p50', bins=30):
    """
    Plot overlaid histograms of intra‑ and inter‑cluster medians with a light grid.
    """
    intra = df[df.label1 == df.label2][metric]
    inter = df[df.label1 != df.label2][metric]

    plt.figure(figsize=(6, 4))
    plt.hist(inter, bins=bins, alpha=0.5, label='inter‑cluster')
    plt.hist(intra, bins=bins, alpha=0.5, label='intra‑cluster')
    plt.grid(alpha=0.3, linestyle='--')
    plt.title('Median Distance Distribution')
    plt.xlabel(metric)
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_p50_across_batches(df, metric_prefix='p50_'):
    """
    For each unique (label1, label2) pair in df, plots a bar chart of
    all the columns starting with `metric_prefix`.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'label1', 'label2' and columns like 'p50_batch1', 'p50_batch2', ...
    metric_prefix : str
        Prefix of the metric columns to plot (default 'p50_').
    """
    # 1) find all batch‐columns
    p50_cols = [c for c in df.columns if c.startswith(metric_prefix)]
    # extract the batch identifiers (e.g. 'batch1', 'batch2', ...)
    batches = [c[len(metric_prefix):] for c in p50_cols]

    # 2) loop over each pair
    pairs = df[['label1','label2']].drop_duplicates()
    for l1, l2 in pairs.values:
        row = df[(df.label1 == l1) & (df.label2 == l2)]
        if row.empty:
            continue
        values = row.iloc[0][p50_cols].values.astype(float)

        # 3) plot
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(batches, values, color='steelblue', edgecolor='black')
        plt.title(f"Median Distances: {l1} vs {l2}", family='serif')
        plt.xlabel("Batch", family='serif')
        plt.ylabel("Median Distance", family='serif')
        plt.xticks(rotation=45, ha='right')
        ymin, ymax = values.min(), values.max()
        ax.set_ylim(ymin*0.95, ymax*1.05)
        plt.tight_layout()
        plt.show()

def plot_boxplot_all_pairs(df, metric_prefix='p50_', figsize=(12,6)):
    """
    Create a box plot showing the distribution of p50 values 
    across batches for each (label1, label2) pair, sorted by median ascending.
    Bold-face the labels where label1 == label2.
    """
    # Identify batch columns and melt
    p50_cols = [c for c in df.columns if c.startswith(metric_prefix)]
    df_long = df.melt(
        id_vars=['label1', 'label2'],
        value_vars=p50_cols,
        var_name='batch',
        value_name='p50'
    )
    df_long['pair'] = df_long['label1'] + ' vs ' + df_long['label2']
    
    # Compute median per pair and sort
    medians = df_long.groupby('pair')['p50'].median()
    order = medians.sort_values().index.tolist()
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df_long,
        x='pair',
        y='p50',
        order=order,
        palette='pastel',
        ax=ax
    )
    ax.set_xticklabels([
        lbl if lbl.split(' vs ')[0] != lbl.split(' vs ')[1]
        else lbl.replace(lbl, lbl, 1)  # placeholder
        for lbl in order
    ], rotation=45, ha='right')
    
    # Bold intra‐cluster labels
    for label in ax.get_xticklabels():
        text = label.get_text()
        l1, l2 = text.split(' vs ')
        if l1 == l2:
            label.set_fontweight('bold')
    
    ax.set_xlabel('Label Pair')
    ax.set_ylabel('Median Distance (p50)')
    ax.set_title('Distribution of Median Distances Across Batches')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(corr_df, method="spearman", figsize=(6, 5), cmap="viridis"):
    """
    Plots a heatmap of the correlation matrix with annotations.

    Args:
        corr_df (pd.DataFrame): The correlation matrix.
        method (str): Correlation method name to use in the colorbar label and title.
        figsize (tuple): Size of the figure.
        cmap (str): Matplotlib colormap to use.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_df,
        annot=True, fmt=".2f", cmap=cmap,
        cbar_kws=dict(label=method)
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    title = f"Pairwise {method} correlations of p50"
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pval_heatmap(pval_matrix, labels, title="Correlation p-values", figsize=(8,6), cmap='Reds'):
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(pval_matrix, dtype=bool))
    sns.heatmap(pval_matrix, xticklabels=labels, yticklabels=labels, 
                mask=mask, cmap=cmap, annot=True, fmt=".2g", 
                cbar_kws={"label": "p-value"}, square=True)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_replicate_bars(df, metric='p50', figsize=(14,6), pad_frac=0.05):
    def base(label): 
        return "_".join(label.split("_")[:-1])

    df2 = df.copy()
    df2['group']   = df2['label1'].map(base)
    df2['rep1']    = df2['label1'].str.split('_').str[-1]
    df2['rep2']    = df2['label2'].str.split('_').str[-1]
    df2 = df2[df2['group'] == df2['label2'].map(base)]
    df2['rep_pair'] = df2.apply(lambda r: f"{r.rep1}-{r.rep2}", axis=1)

    # Define palette and hue order
    palette = {
        'rep1-rep1': "#FFD1BA",   # light peach
        'rep2-rep2': "#FFD1BA",   # same peach
        'rep1-rep2': "#D35400"    # rich burnt orange
    }
    hue_order = ['rep1-rep1', 'rep2-rep2', 'rep1-rep2']

    vmin, vmax = df2[metric].min(), df2[metric].max()
    pad = (vmax - vmin) * pad_frac

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df2,
        x='group', y=metric,
        hue='rep_pair',
        hue_order=hue_order,
        palette=palette,
        edgecolor='black',
        linewidth=0.5,
        dodge=True
    )
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel("Group")
    ax.set_ylabel(f"Median Distance ({metric})")
    ax.set_title("Intra- vs Inter-Replicate Distances", weight='bold')
    ax.legend(title="Replicate Pair", loc='upper right')
    plt.tight_layout()
    plt.show()
    
# Added by Nancy - to support more than 2 reps automatically
def plot_replicate_bars_extended(df, metric='p50', figsize=(5, 4), pad_frac=0.05):
    def base(label): 
        return "_".join(label.split("_")[:-1])

    df2 = df.copy()
    df2['group'] = df2['label1'].map(base)
    df2 = df2[df2['group'] == df2['label2'].map(base)]

    df2['rep1'] = df2['label1'].str.split('_').str[-1]
    df2['rep2'] = df2['label2'].str.split('_').str[-1]

    # Assign self or replicate pair labels
    df2['rep_pair'] = df2.apply(
        lambda r: f'self_{r["rep1"]}' if r['label1'] == r['label2'] else '-'.join(sorted([r['rep1'], r['rep2']])),
        axis=1
    )

    # Generate palette
    rep_pairs = sorted(df2['rep_pair'].unique())
    cmap = get_cmap("Set2")
    palette = {pair: cmap(i % cmap.N) for i, pair in enumerate(rep_pairs)}
    for pair in rep_pairs:
        if pair.startswith('self_'):
            palette[pair] = 'black'

    vmin, vmax = df2[metric].min(), df2[metric].max()
    pad = (vmax - vmin) * pad_frac

    groups = sorted(df2['group'].unique())
    n_rows = len(groups)

    fig, axes = plt.subplots(
        nrows=n_rows,
        figsize=(figsize[0] * 1.8, figsize[1] * n_rows),
        sharey=True
    )

    if n_rows == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        sub_df = df2[df2['group'] == group]
        sns.barplot(
            data=sub_df,
            x='rep_pair',
            hue='rep_pair',
            y=metric,
            palette=palette,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
            legend=False
        )
        ax.set_title(group, loc='left', fontsize=11, weight='bold')
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.grid(axis='y', linestyle='--', linewidth=0.3)
        ax.set_xlabel("")  # hide x-axis label per subplot
        ax.set_ylabel("")  # optional: hide y-axis label
        plt.setp(ax.get_xticklabels(), rotation=90)


    axes[-1].set_xlabel("Replicate Pair")
    axes[n_rows // 2].set_ylabel(f"Median Distance ({metric})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Replicate Pair", loc='upper right', bbox_to_anchor=(1.02, 1))
    fig.suptitle("Intra- vs Inter-Replicate Distances", fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.show()






def plot_replicate_boxes(df, figsize=(14,6), pad_frac=0.05):
    def base(label): 
        return "_".join(label.split("_")[:-1])

    df2 = df.copy()
    df2['group'] = df2['label1'].map(base)
    df2['rep1']  = df2['label1'].str.split('_').str[-1]
    df2['rep2']  = df2['label2'].str.split('_').str[-1]
    df2 = df2[df2['group'] == df2['label2'].map(base)]
    df2['rep_pair'] = df2.apply(lambda r: f"{r.rep1}-{r.rep2}", axis=1)

    # Colors
    palette = {
        'rep1-rep1': "#FFD1BA",
        'rep2-rep2': "#FFD1BA",
        'rep1-rep2': "#D35400"
    }
    hue_order = ['rep1-rep1', 'rep2-rep2', 'rep1-rep2']
    rep_offset = {'rep1-rep1': -0.25, 'rep2-rep2': 0.0, 'rep1-rep2': 0.25}

    groups = sorted(df2['group'].unique())
    x_positions = {g: i for i, g in enumerate(groups)}

    vmin = df2['lower_whisker'].min()
    vmax = df2['upper_whisker'].max()
    pad = (vmax - vmin) * pad_frac

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df2.iterrows():
        group = row['group']
        x_base = x_positions[group]
        offset = rep_offset.get(row['rep_pair'], 0)
        x = x_base + offset
        color = palette.get(row['rep_pair'], 'gray')

        # Draw box (p25 to p75)
        box = patches.Rectangle(
            (x - 0.1, row['p25']),
            0.2,
            row['p75'] - row['p25'],
            facecolor=color,
            edgecolor='black',
            linewidth=0.8
        )
        ax.add_patch(box)

        # Draw median
        ax.plot([x - 0.1, x + 0.1], [row['p50'], row['p50']], color='black', linewidth=1.2)

        # Draw whiskers
        ax.plot([x, x], [row['lower_whisker'], row['p25']], color='black', linewidth=0.8)
        ax.plot([x, x], [row['p75'], row['upper_whisker']], color='black', linewidth=0.8)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_ylabel("Distance")
    ax.set_title("Intra- vs Inter-Replicate Distance Box Summary", weight='bold')

    # Create legend
    for label, color in palette.items():
        ax.plot([], [], color=color, label=label, linewidth=8)
    ax.legend(title="Replicate Pair")

    plt.tight_layout()
    plt.show()