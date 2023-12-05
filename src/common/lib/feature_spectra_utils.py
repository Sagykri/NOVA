import os 

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc
from matplotlib.colors import ListedColormap

import scipy
from collections import defaultdict
import networkx as nx

def load_multiple_vqindhists(batches, embeddings_folder, datasets = ['trainset','valset','testset']):
    vqindhist, labels, paths = [] , [], []
    for batch in batches:
        for dataset_type in datasets:
            cur_vqindhist, cur_labels, cur_paths = np.load(os.path.join(embeddings_folder, f"batch{batch}_16bit_no_downsample/vqindhist1_{dataset_type}.npy")),\
                    np.load(os.path.join(embeddings_folder, f"batch{batch}_16bit_no_downsample/vqindhist1_labels_{dataset_type}.npy")),\
                    np.load(os.path.join(embeddings_folder, f"batch{batch}_16bit_no_downsample/vqindhist1_paths_{dataset_type}.npy"))
            cur_vqindhist = cur_vqindhist.reshape(cur_vqindhist.shape[0], -1)
            vqindhist.append(cur_vqindhist)
            labels.append(cur_labels)
            paths.append(cur_paths)   
    return vqindhist, labels, paths

def create_vqindhists_df(vqindhist, labels, paths, arange_labels=True):
    vqindhist = np.concatenate(vqindhist)
    labels = np.concatenate(labels)
    paths = np.concatenate(paths)
    
    hist_df = pd.DataFrame(vqindhist)
    hist_df['label'] = labels
    hist_df['label'] = hist_df['label'].str.replace("_16bit_no_downsample", "")
    hist_df['label'] = hist_df['label'].str.replace("_spd_format", "")
    hist_df['label'] = hist_df['label'].str.replace(os.sep, "_")

    def rearrange_string(s):
        parts = s.split('_')
        return f"{parts[4]}_{parts[1]}_{parts[2]}_{parts[0]}_{parts[3]}"

    if arange_labels:
        hist_df['label'] = hist_df['label'].apply(rearrange_string)
    hist_df_with_path = hist_df.copy()
    hist_df_with_path['path'] = paths
    
    return hist_df, hist_df_with_path

def cut_dendrogram_get_clusters(clustermap, corr, cutoff = 14.2): ## Cut the dendrogram to get indices clusters
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    den = scipy.cluster.hierarchy.dendrogram(clustermap.dendrogram_col.linkage,
                                             labels = corr.index,
                                             color_threshold=cutoff,
                                             no_labels=True,
                                             ax=axs[0])
    axs[0].axhline(cutoff, c='black', linestyle="-")
    #return den
    
    def get_cluster_classes(den):
        cluster_classes = defaultdict(list)
        seen = []
        cur_cluster = 1
        last_color = den['leaves_color_list'][0]
        for label, color in zip(den['ivl'], den['leaves_color_list']):
            if color != last_color:
                cur_cluster += 1
                last_color = color
            cluster_classes[cur_cluster].append(label)     
        return cluster_classes

    clusters = get_cluster_classes(den)
    cluster = []
    corr_with_clusters = corr.copy()

    for i in corr_with_clusters.index:
        included=False
        for j in clusters.keys():
            if i in clusters[j]:
                cluster.append(j)
                included=True
        if not included:
            cluster.append(None)

    corr_with_clusters["cluster"] = cluster

    # visualize the cluster counts
    sns.countplot(data=corr_with_clusters.sort_values(by='cluster'), x='cluster', palette='coolwarm', ax=axs[1])

    # Add labels and title
    axs[1].set_xlabel('Cluster')
    axs[1].set_ylabel('Indices Count')
    axs[1].set_title('Indices Counts per Cluster')

    plt.tight_layout()
    # Show
    plt.show()
    corr_with_clusters['cluster'] = corr_with_clusters['cluster'].astype(str)
    corr_with_clusters['cluster'] = 'C' + corr_with_clusters['cluster']
    return corr_with_clusters

def add_condition_to_label(label, condition):
    l = label.split("_")
    l.insert(2, condition)
    return '_'.join(l)
    
def create_codebook_heatmap(hist_df, save_path=None, to_save=False, filename=None):
    ## Average the histograms per each label and save in a new dataframe (mean_spectra_per_marker)
    mean_spectra_per_marker = hist_df.groupby('label').mean()
    
    ## Correlate the indices histograms    
    corr = mean_spectra_per_marker.corr()
    
    # remove columns or rows that are all nan (which can happen when the data is constant before the correlation)
    corr.dropna(axis=0, how='all', inplace=True)
    corr.dropna(axis=1, how='all', inplace=True)
    
    ## Plot correlation heatmap
    kws = dict(cbar_kws=dict(ticks=[-1,0,1]))
    clustermap = sns.clustermap(corr, center=0, cmap='bwr', vmin=-1, vmax=1, figsize=(12,7), xticklabels=False, **kws)
    clustermap.ax_row_dendrogram.set_visible(False)
    clustermap.ax_cbar.set_position([clustermap.ax_col_dendrogram.get_position().x1+0.01, # x location 
                                     clustermap.ax_col_dendrogram.get_position().y0+0.01, # y location
                                     0.01,                                                # width
                                     clustermap.ax_col_dendrogram.get_position().height-0.05]) #height
    clustermap.ax_cbar.set_title('Pearson r',fontsize=6)
    clustermap.cax.tick_params(axis='y', labelsize=6, length=0, pad=0.1) 
    #plt.show()
    if to_save:
        clustermap.figure.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)
    
    return clustermap, corr

def plot_histograms(axs, cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, div, 
                    color_by_cond, colors, max_per_condition, cluster_counts, plot_delta, plot_cluster_lines=True, linewidth=1, show_yscale=True, scale_max=True):
    # plot the histograms
    for i, label in enumerate(cur_groups[::-1]):
        if plot_delta:
            label1 = add_condition_to_label(label, condition=first_cond)
            label2 = add_condition_to_label(label, condition=second_cond)
            d1 = total_spectra_per_marker_ordered.loc[label1, :]
            d2 = total_spectra_per_marker_ordered.loc[label2, :]
            if div:
                d1 = d1.apply(lambda x: 1 if x < 1 else x)
                d2 = d2.apply(lambda x: 1 if x < 1 else x)    
                d = np.log2(d1 / d2)
            else:
                d = d1 - d2
        else:
            d = total_spectra_per_marker_ordered.loc[label, :]
        if color_by_cond:
            if first_cond in label: 
                cur_color = colors[first_cond]
            else:
                cur_color = colors[second_cond]
            
            axs[i].fill_between(range(len(d)), d, color=cur_color, label=label, linewidth=linewidth)
            if scale_max:# Set same y label limit for pairs to be compared
                max_limit = max_per_condition.loc[max_per_condition['label']==label, 'max'].values[0]
                axs[i].set_ylim(0, max_limit+(0.25*max_limit))
        else:
            axs[i].fill_between(range(len(d)), d, color=colors[i], label=label, linewidth=linewidth)
        
        if div:
            axs[i].set_ylim(-4,6)
        else:
            axs[i].margins(y=0.25)
        axs[i].set_xticklabels([])
        axs[i].set_xticks([])
        if not show_yscale:
            axs[i].set_yticklabels([])
            axs[i].set_yticks([])
        axs[i].tick_params(axis='y', labelsize=4, length=0, pad=0.1)
        #splitted_label = label.split(sep)
        label_for_plot = label.replace('_', ' ')#'' # comment out - old label editing, when label had also batch, rep, cell line..
        # if len(cur_cell_lines)>1:
        #     label_for_plot+= f'{splitted_label[-4]}_'        
        # if len(cur_conditions)>1:
        #     label_for_plot+= f'{splitted_label[-3]}_'  
        # if len(cur_markers)>1:
        #     label_for_plot+= f'{splitted_label[-5]}_'
        # if len(cur_batches)>1:
        #     label_for_plot+= f'{splitted_label[-2]}_'
        # if len(cur_reps)>1:
        #     label_for_plot+= f'{splitted_label[-1]}'
        # if label_for_plot.endswith("_"):
        #     label_for_plot = label_for_plot[:-1]
        axs[i].text(1.02, 0.5, label_for_plot, transform=axs[i].transAxes,
                    rotation=0, va='center', ha='left')
        if plot_cluster_lines:
            # add cluster lines to histograms
            prev_count = 0
            for j, cluster in enumerate(cluster_counts.cluster):
                cur_count = cluster_counts.iloc[j]['count']
                cluster_end = cur_count + prev_count

                if cur_count < 15:
                    prev_count = cluster_end
                    continue
                axs[i].axvline(x=cluster_end, color='black',linestyle="--", linewidth=0.4)
                prev_count = cluster_end

        axs[i].spines['bottom'].set_color('lightgray')
        axs[i].spines['top'].set_color('lightgray')
        axs[i].spines['right'].set_color('lightgray')
        axs[i].spines['left'].set_color('lightgray')
        axs[i].margins(x=0)
    return axs

def plot_heatmap_with_clusters_and_histograms(corr_with_clusters, hist_df, labels, save_path=None, to_save=False, color_by_cond=False,
                                              plot_delta=False, div=False, sep_histograms = False,
                                              sep = "_", colormap_name = "viridis", plot_hists=True,
                                             filename="codeword_idx_heatmap_and_histograms.tiff",
                                             colors = {"Untreated": "#52C5D5", 'stress': "#F7810F"},
                                             first_cond='stress',second_cond='Untreated',
                                             title=None, figsize=(9,3)):
    # create the heatmap and dendrogram
    kws = dict(cbar_kws=dict(ticks=[-1,0,1]))
    clustermap = sns.clustermap(corr_with_clusters.drop(columns=['cluster']), center=0, cmap='bwr', vmin=-1, vmax=1, figsize=(9,5), xticklabels=False, yticklabels=False, col_colors=corr_with_clusters.cluster, **kws)
    clustermap.ax_row_dendrogram.set_visible(False)

    # get the indices order from the dendrogram 
    hierarchical_order = clustermap.dendrogram_col.reordered_ind
    
    # prepare labels and filter histograms of wanted labels
    real_labels = []
    for label in labels:
        if label not in np.unique(hist_df.label):
            real_labels += [real_label for real_label in np.unique(hist_df.label) if label in real_label]
        else:
            real_labels.append(label)
    hist_df_cur = hist_df[hist_df.label.isin(real_labels)]
    cur_groups = real_labels
    splitted_labels = hist_df_cur.label.str.split(sep)
    cur_batches = np.unique(splitted_labels.str[-2])
    cur_markers = np.unique(splitted_labels.str[-5])
    cur_cell_lines = np.unique(splitted_labels.str[-4])
    cur_conditions = np.unique(splitted_labels.str[-3])
    cur_reps =  np.unique(splitted_labels.str[-1])
    # Mean the histograms by labels and re-order by the indices order
    total_spectra_per_marker_ordered = hist_df_cur.groupby('label').mean()[hierarchical_order] #TODO: change to mean?
    
    if color_by_cond:
        # Set same y label limit for pairs to be compared
        tmp1 = pd.DataFrame(total_spectra_per_marker_ordered.max(axis=1)).reset_index()
        tmp1.columns = ['label', 'max']
        #tmp1['label_s'] = tmp1['label'].str.split("_").apply(lambda x: '_'.join(x[:2] + x[3:]))
        tmp1['label_s'] = tmp1['label'].str.split("_").str[0]
        tmp2 = tmp1.groupby('label_s').max()
        tmp2.columns = ['label', 'max']
        max_per_condition = tmp1[['label', 'label_s']].merge(tmp2['max'], right_index=True, left_on='label_s')
    else:
        max_per_condition = None
    if plot_delta:
        # Pairs to be compared
        #list_of_pairs = list(set(total_spectra_per_marker_ordered.reset_index()['label'].str.split("_").str[0]))#.apply(lambda x: '_'.join(x[:2] + x[3:]))))
        #cur_groups = list_of_pairs
        cur_groups = labels
    # calc clusters locations
    cluster_counts = pd.DataFrame(corr_with_clusters.cluster.value_counts()).reset_index()
    cluster_counts.cluster = cluster_counts.cluster.str.replace('C','').astype('int')
    cluster_counts.sort_values(by='cluster', inplace=True)
    # cluster_positions = clustermap.ax_col_dendrogram.get_position()
    # num_samples = len(clustermap.dendrogram_col.data)
    if plot_hists:
        # make room for the histograms in the plot
        hist_height = 0.05
        clustermap.fig.subplots_adjust(top=hist_height*len(cur_groups)+1, bottom=hist_height*len(cur_groups))
        
        # add axes for the histograms
        axs=[]
        for i, label in enumerate(cur_groups):
            axs.append(clustermap.fig.add_axes([clustermap.ax_heatmap.get_position().x0, 0+i*hist_height, clustermap.ax_heatmap.get_position().width, hist_height]))

        if not color_by_cond:
            # create colors
            colors = sns.color_palette(colormap_name, n_colors=len(cur_groups))
        
        ### PLOT THE HISTOGRAMS ###
        axs = plot_histograms(axs, cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, div, 
                        color_by_cond, colors, max_per_condition, cluster_counts, plot_delta)
    
    # fix the cbar appearance 
    clustermap.ax_cbar.set_position([clustermap.ax_col_dendrogram.get_position().x1+0.01, # x location 
                                     clustermap.ax_col_dendrogram.get_position().y0+0.01, # y location
                                     0.01,                                                # width
                                     clustermap.ax_col_dendrogram.get_position().height-0.05]) #height
    clustermap.ax_cbar.set_title('Pearson r',fontsize=6)
    clustermap.cax.tick_params(axis='y', labelsize=6, length=0, pad=0.1)
   
    # add cluster lines to the heatmap
    prev_count = 0
    for j, cluster in enumerate(cluster_counts.cluster):
        cur_count = cluster_counts.iloc[j]['count']

        cluster_end = cur_count + prev_count
        
        if cur_count < 15:
            prev_count = cluster_end
            continue
        clustermap.ax_heatmap.axvline(x=cluster_end, color='black',linestyle="--", linewidth=0.4)
        clustermap.ax_col_colors.text(x=cluster_end-(cur_count/2), y=0.5, s=cluster, fontsize=6)
        prev_count = cluster_end

    if title:
        clustermap.fig.suptitle(title, x = clustermap.ax_col_dendrogram.get_position().x0+(clustermap.ax_col_dendrogram.get_position().x1-clustermap.ax_col_dendrogram.get_position().x0)/2,
                               y = clustermap.ax_col_dendrogram.get_position().y1+0.05)
    if to_save and not sep_histograms:
        clustermap.figure.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    if sep_histograms:
        fig, axs = plt.subplots(nrows = len(cur_groups), figsize=figsize)
        axs = plot_histograms(axs[::-1], cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, div, 
                        color_by_cond, colors, max_per_condition, cluster_counts, plot_delta)
        fig.subplots_adjust(hspace=0)
        if to_save:
            fig.savefig(os.path.join(save_path, "only_histograms_" + filename),bbox_inches='tight', dpi=300)
    
    return None


def plot_hists_supp_1A(hist_df, labels, corr_with_clusters, hierarchical_order=None, sort=False, color_by_cond=False, plot_delta=False, 
                       to_save=False, colormap_name='viridis',figsize=(5,5), first_cond='stress', second_cond='Untreated', 
                       save_path=None, filename=None, colors = {"Untreated": "#52C5D5", 'stress': "#F7810F"}, to_mean=True, plot_cluster_lines=True, scale_max=False):
    real_labels = []
    for label in labels:
        if label not in np.unique(hist_df.label):
            real_labels += [real_label for real_label in np.unique(hist_df.label) if label in real_label]
        else:
            real_labels.append(label)
    hist_df_cur = hist_df[hist_df.label.isin(real_labels)]
    cur_groups = real_labels

    # Mean the histograms by labels 
    total_spectra_per_marker_ordered = hist_df_cur.copy()
    total_spectra_per_marker_ordered.set_index('label', inplace=True)
    if to_mean:
        total_spectra_per_marker_ordered = hist_df_cur.groupby('label').mean()

    #re-order by the indices order
    if sort:
        total_spectra_per_marker_ordered=total_spectra_per_marker_ordered[hierarchical_order]

    if color_by_cond and scale_max:
        # Set same y label limit for pairs to be compared
        tmp1 = pd.DataFrame(total_spectra_per_marker_ordered.max(axis=1)).reset_index()
        tmp1.columns = ['label', 'max']
        tmp1['label_s'] = tmp1['label'].str.split("_").str[0]
        tmp2 = tmp1.groupby('label_s').max()
        tmp2.columns = ['label', 'max']
        max_per_condition = tmp1[['label', 'label_s']].merge(tmp2['max'], right_index=True, left_on='label_s')
    else:
        max_per_condition = None
    if plot_delta:
        # Pairs to be compared
        cur_groups = labels
    
    # calc clusters locations
    cluster_counts = pd.DataFrame(corr_with_clusters.cluster.value_counts()).reset_index()
    cluster_counts.cluster = cluster_counts.cluster.str.replace('C','').astype('int')
    cluster_counts.sort_values(by='cluster', inplace=True)


    if to_mean:
        fig, axs = plt.subplots(nrows = len(cur_groups), figsize=figsize)
        if plot_delta:
            colors = sns.color_palette(colormap_name, n_colors=len(cur_groups)*2)[0::3]
        elif not color_by_cond:
            # create colors
            colors = [ListedColormap([colormap_name])(0)]*len(cur_groups)
        axs = plot_histograms(axs[::-1], cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, False, 
                    color_by_cond, colors, max_per_condition, cluster_counts, plot_delta, plot_cluster_lines, linewidth=2, show_yscale=False, scale_max=False)
    else:
        fig, axs = plt.subplots(nrows = len(cur_groups)*2, figsize=figsize)
        if not color_by_cond:
            # create colors
            colors = [ListedColormap([colormap_name])(0)]*len(cur_groups)*2
        axs = plot_histograms_not_mean(axs[::-1], cur_groups, first_cond, second_cond,total_spectra_per_marker_ordered,
                                 color_by_cond, colors, max_per_condition, cluster_counts, plot_cluster_lines, show_yscale=False)
    
    fig.subplots_adjust(hspace=0)
    if to_save:
        fig.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)

def plot_histograms_not_mean(axs, cur_groups, first_cond, second_cond, total_spectra_per_marker_ordered, 
                    color_by_cond, colors, max_per_condition, cluster_counts, plot_cluster_lines=True, show_yscale=True):
    # plot the histograms
    for i, label in enumerate(cur_groups[::-1]):
        d = total_spectra_per_marker_ordered.loc[label, :]
        for j in range(d.shape[0]):
            cur_d = d.iloc[j,:]
            ax = axs[i*d.shape[0] + j]
            if color_by_cond:
                if first_cond in label: 
                    cur_color = colors[first_cond]
                else:
                    cur_color = colors[second_cond]
                ax.fill_between(range(len(cur_d)), cur_d, color=cur_color, label=label, linewidth=2)
                # Set same y label limit for pairs to be compared
                # max_limit = max_per_condition.loc[max_per_condition['label']==label, 'max'].values[0]
                # ax.set_ylim(0, max_limit+(0.25*max_limit))
            else:
                ax.fill_between(range(len(cur_d)), cur_d, color=colors[i], label=label, linewidth=2)
        
            ax.margins(y=0.25)
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=4, length=0, pad=0.1)
            ax.text(1.02, 0.5, label.replace('_', ' '), transform=ax.transAxes,
                    rotation=0, va='center', ha='left')
            if plot_cluster_lines:
                # add cluster lines to histograms
                prev_count = 0
                for j, cluster in enumerate(cluster_counts.cluster):
                    cur_count = cluster_counts.iloc[j]['count']
                    cluster_end = cur_count + prev_count

                    if cur_count < 15:
                        prev_count = cluster_end
                        continue
                    ax.axvline(x=cluster_end, color='black',linestyle="--", linewidth=1)
                    prev_count = cluster_end

            ax.spines['bottom'].set_color('lightgray')
            ax.spines['top'].set_color('lightgray')
            ax.spines['right'].set_color('lightgray')
            ax.spines['left'].set_color('lightgray')
            ax.margins(x=0)
            if not show_yscale:
                ax.set_yticklabels([])
                ax.set_yticks([])
    return axs

def plot_heatmap_with_clusters_supp_1A(corr_with_clusters, save_path=None, to_save=False,                                            
                                             filename="codeword_idx_heatmap_and_histograms.tiff",
                                             figsize=(9,3), cmap=None):
    col_colors_df = {'C1': 'olivedrab', 'C2': 'teal', 'C3': 'goldenrod','C4':'purple'}
    # create the heatmap and dendrogram
    kws = dict(cbar_kws=dict(ticks=[-1,0,1]))
    clustermap = sns.clustermap(corr_with_clusters.drop(columns=['cluster']), center=0, cmap=cmap, vmin=-1, vmax=1, 
                                figsize=figsize, xticklabels=False, yticklabels=False, col_colors=corr_with_clusters['cluster'].map(col_colors_df), **kws)
    clustermap.ax_row_dendrogram.set_visible(False)

    # calc clusters locations
    cluster_counts = pd.DataFrame(corr_with_clusters.cluster.value_counts()).reset_index()
    cluster_counts.cluster = cluster_counts.cluster.str.replace('C','').astype('int')
    cluster_counts.sort_values(by='cluster', inplace=True)
    
    # fix the cbar appearance 
    clustermap.ax_cbar.set_position([clustermap.ax_col_dendrogram.get_position().x1+0.01, # x location 
                                     clustermap.ax_col_dendrogram.get_position().y0+0.01, # y location
                                     0.01,                                                # width
                                     clustermap.ax_col_dendrogram.get_position().height-0.05]) #height
    clustermap.ax_cbar.set_title('Pearson r',fontsize=6)
    clustermap.cax.tick_params(axis='y', labelsize=6, length=0, pad=0.1)
   
    # add cluster lines to the heatmap
    prev_count = 0
    for j, cluster in enumerate(cluster_counts.cluster):
        cur_count = cluster_counts.iloc[j]['count']

        cluster_end = cur_count + prev_count
        
        if cur_count < 15:
            prev_count = cluster_end
            continue
        # clustermap.ax_heatmap.axvline(x=cluster_end, color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([cluster_end,cluster_end], [prev_count,cluster_end], color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([prev_count,cluster_end], [cluster_end,cluster_end], color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([prev_count,prev_count], [prev_count,cluster_end], color='black',linestyle="--", linewidth=1)
        clustermap.ax_heatmap.plot([prev_count,cluster_end], [prev_count,prev_count], color='black',linestyle="--", linewidth=1)
        clustermap.ax_col_colors.text(x=cluster_end-(cur_count/2), y=0.8, s=cluster, fontsize=6)
        prev_count = cluster_end

    if to_save:
        clustermap.figure.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    return clustermap.dendrogram_col.reordered_ind

def find_rep_per_cluster(corr_with_clusters, hist_df_with_path, save_path, to_save=False, save_together = True,
                         filename="representative_images_per_cluster.eps", figsize=(4,32), use_second_max=False, top_images = 4):
    clusters = np.unique(corr_with_clusters.cluster)

    hist_per_cluster = pd.DataFrame(index = hist_df_with_path.index, columns = list(clusters) + ['label','path'])
    hist_per_cluster.label = hist_df_with_path.label
    hist_per_cluster.path = hist_df_with_path.path

    # for each cluster, get the indices and calc the sum of the histogram, then normalize by the cluster size
    for cluster_label, cluster_group in corr_with_clusters.groupby('cluster'):
        hist_per_cluster[cluster_label] = hist_df_with_path[cluster_group.index].sum(axis=1) / (cluster_group.index.size) #625 
    # Find the two largest values and corresponding columns (clusters) for each row
    top_clusters = hist_per_cluster.drop(['label', 'path'], axis=1).apply(lambda row: row.nlargest(2).index, axis=1)

    # Assign the first and second max clusters to new columns
    hist_per_cluster['max_cluster'] = top_clusters.apply(lambda x: x[0])
    hist_per_cluster['second_max_cluster'] = top_clusters.apply(lambda x: x[1])
    hist_per_cluster.max_cluster = hist_per_cluster.max_cluster.str.replace('C',"").astype(int)
    hist_per_cluster.second_max_cluster = hist_per_cluster.second_max_cluster.str.replace('C',"").astype(int)
    
    if save_together:
        fig, axs = plt.subplots(nrows=int(top_images/2)*np.unique(hist_per_cluster[['max_cluster', 'second_max_cluster']]).size, ncols=2, figsize=figsize)

    for i, cluster in enumerate(np.unique(hist_per_cluster[['max_cluster', 'second_max_cluster']])):
        max_cluster_group = hist_per_cluster[hist_per_cluster.max_cluster==cluster]
        max_cluster_column = f"C{cluster}"
        max_tiles_paths = max_cluster_group[[max_cluster_column,'path']].sort_values(by=max_cluster_column,ascending=False)[:top_images].path
        if max_tiles_paths.size == 0:
            if use_second_max:
                max_cluster_group = hist_per_cluster[hist_per_cluster.second_max_cluster == cluster]
                max_tiles_paths = max_cluster_group[[max_cluster_column,'path']].sort_values(by=max_cluster_column,ascending=False)[:top_images].path
                print(f'using second max cluster for {max_cluster_column}')
            else:
                continue
        for j, tile_path in enumerate(max_tiles_paths):
            cut = tile_path.rfind("_")
            real_path = tile_path[:cut]
            tile_number = int(tile_path[cut+1:])
            cur_site = np.load(real_path)
            if save_together:
                ax = axs[i * int(top_images/2) + j // 2, j%2]
            else:
                fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(cur_site[tile_number,:,:,0], cmap='gray',vmin=0,vmax=1)
            ax.axis('off')
            split_path=real_path.split(os.sep)
            marker = split_path[-2]
            condition = split_path[-3]
            if 'Untreated' in condition:
                condition = condition[:3]
            cell_line = split_path[-4]
            if 'FUS' in cell_line:
                cell_line = cell_line[:6]
            rep = split_path[-1].split("_")[0]
            label = f"{cell_line}_{condition}_\n{marker}_{rep}"
            if not save_together and to_save:
                os.makedirs(os.path.join(save_path, 'separated_images'), exist_ok=True)
                label = label.replace("\n","")
                plt.savefig(os.path.join(save_path, 'separated_images',f'{max_cluster_column}_{j}_{label}.eps'), 
                            bbox_inches='tight')
            ax.text(60,95,label, color='yellow', fontsize=6)
            if j==0:
                ax.text(-50,100, max_cluster_column, fontsize=15)
        if save_together and max_tiles_paths.size < top_images: # if found less then 4, add empty plots
            for k in range(j+1,top_images):
                axs[i * 2 + k // 2, k%2].axis('off')

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    if save_together and to_save:
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
    plt.show()

    # plot stacked bar plot of lables in each cluster
    colors = ListedColormap(sns.color_palette(cc.glasbey, n_colors=24)) #used for when we have 24 markers

    hist_per_cluster['short_label'] = hist_per_cluster.label.str.split('_').str[0:3:2].apply(lambda x: "_".join(x)) #include also condition in the label
    label_per_cluster = hist_per_cluster[['short_label','max_cluster']]
    stack=pd.DataFrame(label_per_cluster.groupby(['max_cluster','short_label']).short_label.count() *100 / label_per_cluster.groupby(['max_cluster']).short_label.count())
    stack = stack.rename(columns={'short_label': 'label_count'})
    stack = stack.reset_index()
    stack = stack.sort_values(by='max_cluster')
    df_pivot = stack.pivot(index='max_cluster', columns='short_label', values='label_count').fillna(0)
    base_cmap = plt.cm.get_cmap('Paired', 12) #used for when we have 3 markers and 2 conds
    cmap = ListedColormap([base_cmap(i) for i in range(0, len(df_pivot.columns))])

    ax=df_pivot.plot(kind='bar', stacked=True, cmap = cmap)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, (index, row) in enumerate(df_pivot.iterrows()):
        total_height = 0
        for value, color in zip(row, ax.patches[i::len(df_pivot)]):
            if round(value,0) == 0:
                continue
            ax.text(
                color.get_x() + color.get_width() / 2,
                total_height + value / 2,
                f'{round(value)}%',  # Format the value as needed
                ha='center',
                va='center',
                color='white' if value < 0.5 * max(row) else 'black'  # Choose text color based on value
            )
            total_height += value
    plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left',
            borderaxespad=-0.5, fontsize='x-small')
    plt.title('Stacked Bar Plot of Labels per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Label Percentage')
    plt.show()

    return None


def create_correlation_graph(correlation_matrix, top_positive=True, num_edges=2):
    graph = nx.Graph()
    for marker, row in correlation_matrix.iterrows():
        sorted_correlations = row.drop(marker).sort_values(ascending=not top_positive).head(num_edges)
        for other_marker, correlation in sorted_correlations.items():
            graph.add_edge(marker, other_marker, weight=abs(correlation), color=correlation)
    return graph

def create_bicorrelation_graph(correlation_matrix, num_edges=2):
    graph = nx.Graph()
    for marker, row in correlation_matrix.iterrows():
        # first take positive correlations
        sorted_correlations = row.drop(marker).sort_values(ascending=False).head(num_edges)
        for other_marker, correlation in sorted_correlations.items():
            graph.add_edge(marker, other_marker, weight=abs(correlation), color=correlation, label='positive')
        # then take negative correlations
        sorted_correlations = row.drop(marker).sort_values(ascending=True).head(num_edges)
        for other_marker, correlation in sorted_correlations.items():
            graph.add_edge(marker, other_marker, weight=abs(correlation), color=correlation, label='negative')
    return graph

def draw_correlation_graph(graph, title, cmap, vmin, vmax, save_path=None, filename="corr_graph.tiff", to_save=False):
    pos = nx.spring_layout(graph, weight='weight', seed=42) #shell_layout
    fig, ax = plt.subplots(figsize=(15, 8))
    nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=500, edgecolors='black', linewidths=0.5, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=6, ax=ax)
    edge_colors = [graph[u][v]['color'] for u,v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, orientation='vertical', label='Correlation Strength',shrink=0.2,pad=0)
    plt.title(title)
    plt.axis('off')
    if to_save:
        plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    plt.show()
    
def draw_bicorrelation_graph(graph, title, cmap_pos, cmap_neg, vmin_pos = 0, vmax_pos = 1,
                             vmin_neg=-1, vmax_neg=0, save_path=None, filename="corr_graph.tiff", to_save=False):
    pos = nx.spring_layout(graph, weight='weight', seed=42) #shell_layout
    fig, ax = plt.subplots(figsize=(15, 8))
    nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=1500, edgecolors='black', linewidths=0.5, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    #draw positive edges
    positive_edges = [edge for edge in edge_labels if edge_labels[edge]=='positive']
    positive_edge_colors = [graph[u][v]['color'] for u,v in positive_edges]
    nx.draw_networkx_edges(graph, pos, edge_color=positive_edge_colors, edgelist=positive_edges,
                                   edge_cmap=cmap_pos, edge_vmin=vmin_pos, edge_vmax=vmax_pos, ax=ax)
    #draw negative edges
    negative_edges = [edge for edge in edge_labels if edge_labels[edge]=='negative']
    negative_edge_colors = [graph[u][v]['color'] for u,v in negative_edges]
    nx.draw_networkx_edges(graph, pos, edge_color=negative_edge_colors, edgelist=negative_edges,
                                   edge_cmap=cmap_neg, edge_vmin=vmin_neg, edge_vmax=vmax_neg, ax=ax)
    
    sm_pos = plt.cm.ScalarMappable(cmap=cmap_pos, norm=plt.Normalize(vmin=vmin_pos, vmax=vmax_pos))
    sm_pos.set_array([])
    plt.colorbar(sm_pos, orientation='vertical', label='Positive Correlation Strength',shrink=0.2, pad=0)
    sm_neg = plt.cm.ScalarMappable(cmap=cmap_neg, norm=plt.Normalize(vmin=vmin_neg, vmax=vmax_neg))
    sm_neg.set_array([])
    plt.colorbar(sm_neg, orientation='vertical', label='Negative Correlation Strength',shrink=0.2, pad=0, location='left')
    plt.title(title)
    plt.axis('off')
    if to_save:
        plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    plt.show()

def create_lables_heatmap(df, title, save_path=None, to_save=False, filename=None, kl=False, method='pearson'):
    if not kl:
        corrs = df.T.corr(method=method)
        clustermap = sns.clustermap(corrs, center=0, cmap='bwr', vmin=-1, vmax=1, figsize=(5,5))

    if kl:
        corrs=kl_divergence_matrix((df.T/625))
        clustermap = sns.clustermap(corrs, cmap='plasma',vmin=0, vmax=3,figsize=(5,5))

    clustermap.fig.suptitle(title, y=1.05)
    row_order = clustermap.dendrogram_row.reordered_ind
    marker_order = corrs.columns[row_order]

    row_axes = clustermap.ax_heatmap.get_yaxis()
    col_axes = clustermap.ax_heatmap.get_xaxis()
    row_axes.set_ticks(np.arange(len(marker_order)) + 0.5)
    row_axes.set_ticklabels(marker_order, rotation=0, fontsize=6)
    col_axes.set_ticks(np.arange(len(marker_order)) + 0.5)
    col_axes.set_ticklabels(marker_order, rotation=90, fontsize=6)
    
    clustermap.ax_row_dendrogram.set_visible(False)
    clustermap.ax_cbar.set_position([clustermap.ax_col_dendrogram.get_position().x1+0.01, # x location 
                                     clustermap.ax_col_dendrogram.get_position().y0+0.01, # y location
                                     0.01,                                                # width
                                     clustermap.ax_col_dendrogram.get_position().height-0.05]) #height
    clustermap.ax_cbar.set_title(f'{method} r',fontsize=6)
    clustermap.cax.tick_params(axis='y', labelsize=6, length=0, pad=0.1) 
    clustermap.ax_heatmap.set_xlabel('Marker', fontsize=12)
    clustermap.ax_heatmap.set_ylabel('Marker', fontsize=12)
    
    if to_save:
        plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', dpi=300)
    plt.show()
    return marker_order, corrs


def analyse_deltas(df, markers_to_delta, first_cond, second_cond, div=False,
                  heatmap_title = None, graph_filename=None, save_path=None, 
                   to_save=False, heatmap_filename=None, plot_network=True):
    average_hist = df.groupby('label').mean()
    deltas = pd.DataFrame(index = markers_to_delta, columns=average_hist.columns)
    
    # calc delta and save in a new df
    for marker in markers_to_delta:
        first = average_hist.loc[f'{marker}_{first_cond}']
        second =  average_hist.loc[f'{marker}_{second_cond.replace("_Untreated","")}']
        if not div:
            deltas.loc[marker] =  first - second
        
        else: # div
            first = first.apply(lambda x: 1 if x < 1 else x)
            second = second.apply(lambda x: 1 if x == 0 else x)
            deltas.loc[marker] = np.log2((first / second))
            
    markers_order , marker_delta_corr = create_lables_heatmap(deltas, heatmap_title, save_path=save_path, 
                                                  to_save=to_save, filename=heatmap_filename)
    if plot_network:
        positive_graph = create_correlation_graph(marker_delta_corr, top_positive=True)
        draw_correlation_graph(positive_graph, 'Positive Correlation Graph', 
                               cmap=plt.get_cmap('Reds'), vmin=0, vmax=1,
                              save_path=save_path, filename=f"Positive_{graph_filename}", to_save=to_save)

        negative_graph = create_correlation_graph(marker_delta_corr, top_positive=False)
        draw_correlation_graph(negative_graph, 'Negative Correlation Graph', 
                               cmap=plt.get_cmap('Blues_r'), vmin=-1, vmax=0,
                              save_path=save_path, filename=f"Negative_{graph_filename}", to_save=to_save)
    return markers_order


from scipy.special import kl_div

def kl_divergence_matrix(df):
    """
    Compute the matrix of Kullback-Leibler (KL) divergence for each pair of columns in a DataFrame.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - pandas DataFrame containing the KL divergence between each pair of columns
    """
    columns = df.columns
    num_columns = len(columns)
    
    kl_matrix = np.zeros((num_columns, num_columns))
    
    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            kl_matrix[i, j] = np.sum(kl_div(df.iloc[:, i]+0.00001, df.iloc[:, j]+0.00001))
    
    # Filling the lower triangle of the matrix (since KL divergence is not symmetric)
    kl_matrix += kl_matrix.T

    return pd.DataFrame(kl_matrix, index=columns, columns=columns)
