import logging
import os
import pandas as pd
import numpy as np
import umap
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from collections import deque
from adjustText import adjust_text
from sklearn.cluster import KMeans
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image


def calc_umap_embvec(
        analytics,
        data=None,
        selected_ind=None,
        target_vq_layer=None,
        n_components=2,
        savepath=None,
        filename="vqvec_umap",
        verbose=True,
        seed=1
):
    """
    Compute umap of embedding vectors. (With fixed random state)
    :param data: embedding vector data
    :param selected_ind: only selected feature index will be used to compute umap
    :param target_vq_layer: the vq layer to output embedding vector;
    1 for local representation, 2 for global representation
    :param savepath: save path
    :param filename: file name
    :param verbose: verbosity of umap.UMAP
    """
    if data is None:
        if analytics.model.embvec:
            data = analytics.model.embvec
        else:
            analytics.model.calc_embvec(
                analytics.data_manager.test_data)  # ,  savepath="default", filename=f"{filename}_embvec_1u")
            data = analytics.model.embvec
    else:
        if not isinstance(data, list):
            data = [data]
    if selected_ind is not None:
        data = [d[:, i] for d, i in zip(data, selected_ind)]
    if target_vq_layer:
        if isinstance(target_vq_layer, int):
            target_vq_layer = [target_vq_layer]
        data = [
            d if i + 1 in target_vq_layer else np.array([])
            for i, d in enumerate(data)
        ]
    if analytics.model_vec_umap != []:
        logging.info(
            "vqvec models is not empty. vqvec models will be overwritten."
        )

    logging.info(f"Computing UMAP...")
    analytics.model_vec_umap = []
    analytics.vec_umap = []
    for v in data:
        if min(v.shape) > 0:
            reducer = umap.UMAP(n_components=n_components, verbose=verbose, random_state=seed)
            u = reducer.fit_transform(v.reshape(v.shape[0], -1))
        else:
            reducer = []
            u = np.array([])
        analytics.model_vec_umap.append(reducer)
        analytics.vec_umap.append(u)

    if savepath:
        if savepath == "default":
            savepath = analytics.model.savepath_dict["emb"]
        for i, v in enumerate(analytics.vec_umap):
            if min(v.shape) > 0:
                np.save(os.path.join(savepath, f"{filename}{i + 1}.npy"), v)
                try:
                    [
                        pickle.dump(
                            v,
                            open(
                                os.path.join(
                                    savepath, f"model_{filename}{i + 1}.ump"
                                ),
                                "wb",
                            ),
                            protocol=4,
                        )
                        for i, v in enumerate(analytics.model_vec_umap)
                        if v != []
                    ]
                except:
                    logging.info("UMAP model was not saved.")


def plot_umap(analytics,
              data=None,
              labels=None,
              embvec=None,
              alpha=1.0,
              target_vq_layer=2,
              cmap="tab20",
              colors_dict=None,
              annotations_font_size=11,
              to_annot=True,
              xlim=None,
              ylim=None,
              title=None,
              subplot_shape=None,
              show_legend=True,
              gt_table=None,
              s=10,
              savepath=None,
              plot_pca=False,
              figsize=None,
              seed=1):
    """Plot UMAP plot

    Args:
        analytics (_type_): Cytoself's Analytics object
        data (_type_, optional): The data (X). Defaults to None.
        labels (_type_, optional): The labels (y)_. Defaults to None.
        embvec (_type_, optional): Precomputed embedded vectors. Recompute if None. Defaults to None.
        alpha (float, optional): The alpha of the points in the plot. Defaults to 1.0.
        target_vq_layer (int, optional): Which vq layer to use for embedded vectors calculation. Defaults to 2.
        cmap (str, optional): cmap for the plot. Defaults to "tab20".
        colors_dict (_type_, optional): colors dictionary for the plot. Defaults to None.
        annotations_font_size (int, optional): The font size of the annotations. Defaults to 11.
        to_annot (bool, optional): Should annotate?. Defaults to True.
        xlim (_type_, optional): The limit of the x axis. Defaults to None.
        ylim (_type_, optional): The limit of the y axis. Defaults to None.
        title (string, optional): The title for the plot. Defaults to None.
        subplot_shape (_type_, optional): The shape for the subplot. Defaults to None.
        show_legend (bool, optional): Should show legend?. Defaults to True.
        gt_table (_type_, optional): The ground truth table for the data. Points will be colored according to it. Defaults to None.
        s (int, optional): Points' size. Defaults to 10.
        savepath (str, optional): The output file's path. Defaults to None.
        plot_pca (bool, optional): Should plot PCA. Defaults to False.
        figsize (_type_, optional): Figure's size. Defaults to None.

    Returns:
        _type_: data, label
    """

    label = labels.copy() if labels is not None else analytics.data_manager.test_label
    if isinstance(target_vq_layer, int):
        target_vq_layer = [target_vq_layer]

    gt_table = analytics.gt_table if gt_table is None else gt_table
    gt_name = gt_table.iloc[:, 0]
    uniq_group = np.unique(gt_table.iloc[:, 1])

    if data is None:
        analytics.model.embvec = embvec.copy() if embvec is not None else None
        calc_umap_embvec(analytics, target_vq_layer=target_vq_layer, seed=seed)
        data = analytics.vec_umap

    # make sure data is in a list
    data = data if isinstance(data, list) else [data]
    n_subplots = 0
    for d in data:
        if min(d.shape) > 0:
            n_subplots += 1
    # get subplot shape
    if subplot_shape is None:
        ncol = math.ceil(math.sqrt(n_subplots))
        nrow = math.ceil(n_subplots / ncol)
    else:
        nrow, ncol = subplot_shape

    plt.figure(figsize=figsize if figsize is not None else (2 * ncol, 2 * nrow))
    subplot_ind = 1
    annots = []
    for vqi, idht in enumerate(data):
        if vqi + 1 in target_vq_layer:
            logging.info(f"Plotting subplot{vqi} ...")
            plt.subplot(nrow, ncol, subplot_ind)
            if min(idht.shape) > 0:
                if colors_dict is None:
                    if type(cmap) == str:
                        colors = cm.get_cmap(cmap).colors
                    else:
                        # Array
                        colors = []
                        for c in cmap:
                            colors += cm.get_cmap(c).colors
                # plot each cluster layer
                for i, fname in enumerate(tqdm(uniq_group)):
                    group0 = gt_table[gt_table.iloc[:, 1] == fname]
                    ind = np.isin(label[:, 0], group0.iloc[:, 0])
                    data0 = idht[ind]
                    if colors_dict is None:
                        c = np.array(colors[i]).reshape(1, -1)
                    else:
                        c = np.array([*[colors_dict[fname]] * len(data0)])
                    sctrs = plt.scatter(
                        data0[:, 0],
                        data0[:, 1],
                        s=s,
                        c=c,
                        label=fname,
                        alpha=alpha,
                        edgecolors='none'
                    )
                    if to_annot:
                        km = KMeans(n_clusters=1, random_state=seed).fit(data0)
                        annot = plt.annotate(fname, (km.cluster_centers_[0, 0], km.cluster_centers_[0, 1]),
                                             fontsize=annotations_font_size)
                        annots.append(annot)
                if xlim is not None and xlim[vqi] is not None:
                    plt.xlim(xlim[vqi])
                if ylim is not None and ylim[vqi] is not None:
                    plt.ylim(ylim[vqi])

            hndls, names = sctrs.axes.get_legend_handles_labels()
            hndls = deque(hndls)
            names = deque(names)
            hndls.rotate(-1)
            names.rotate(-1)
            if show_legend:
                leg = plt.legend(
                    hndls,
                    names,
                    prop={"size": 10},
                    bbox_to_anchor=(1, 1),
                    loc="upper left",
                )
                for ll in range(len(names)):
                    leg.legendHandles[ll]._sizes = [6]
            if title is None:
                plt.title(f"Ground true samples vq{vqi + 1}")
            elif isinstance(title, str):
                plt.title(f"{title} vq{vqi + 1}")
            plt.ylabel("Umap 2")
            plt.xlabel("Umap 1")
            plt.xticks([])
            plt.yticks([])
            # plt.tight_layout(pad=0.5)

    if to_annot:
        adjust_text(annots, arrowprops=dict(arrowstyle="-", color='black', lw=1.5))

    if savepath is not None:
        plt.savefig(savepath, dpi=300, facecolor="white",bbox_inches='tight')
    plt.show()

    if plot_pca:
        embs = analytics.model.embvec[1]
        embs = embs.reshape(embs.shape[0], -1)
        embs = StandardScaler().fit_transform(embs.copy())
        pca = PCA()
        principal_components = pca.fit_transform(embs)
        principal_df = pd.DataFrame(data=principal_components[:, [0, 1]],
                                    columns=['principal component 1', 'principal component 2'])
        y = pd.DataFrame(label, columns=['group'])
        final_df = pd.concat([principal_df, y], axis=1)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(f'Principal Component 1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)', fontsize=15)
        ax.set_ylabel(f'Principal Component 2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)', fontsize=15)
        ax.set_title(f'PCA', fontsize=20)
        for group in names:
            indicesToKeep = final_df['group'] == group
            ax.scatter(final_df.loc[indicesToKeep, 'principal component 1']
                       , final_df.loc[indicesToKeep, 'principal component 2']
                       , s=s)
        ax.legend(names)
        plt.show()
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
                label='Individual explained variance')
        plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
                 label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    return data, label


def plot_feature_spectrum_from_image(
        analytics,
        data,
        target_vq_layer=1,
        take_mean=True,
        savepath=None,
        filename="Feature_spectrum",
        title=None,
        color='blue',
        figsize=None
):
    """
    Plot feature spectrum from image.
    :param data: image data; make sure it has 4 dimensions (i.e. batch, x, y, channel).
    :param target_vq_layer: 1 for local representation, 2 for global representation
    :param take_mean: take mean spectrum if multiple images were inputted, otherwise plot multiple subplots.
    :param savepath: save path
    :param filename: file name
    :param title: plot title
    """
    logging.info(data.shape)
    embindhist = analytics.model.calc_embindhist(data, do_return=True)
    if len(analytics.dendrogram_index) == 0:
        ValueError("No dendrogram_index found. Load dendrogram_index first.")
    embindhist = embindhist[target_vq_layer - 1][:, analytics.dendrogram_index[0]]
    plot_feature_spectrum_from_vqindhist(
        analytics,
        embindhist,
        take_mean=take_mean,
        savepath=savepath,
        filename=filename,
        title=title,
        color=color,
        figsize=figsize,
        target_vq_layer=target_vq_layer
    )


def plot_feature_spectrum_from_vqindhist(
        analytics,
        embindhist,
        take_mean=True,
        savepath=None,
        filename="Feature_spectrum",
        title=None,
        color=None,
        figsize=None,
        target_vq_layer=1
):
    """
    Plot feature spectrum from vq index histogram.
    :param embindhist: vq index histogram data; make sure it has 2 dimensions (i.e. batch, index histogram).
    :param take_mean: take mean spectrum if multiple images were inputted, otherwise plot multiple subplots.
    :param savepath: save path
    :param filename: file name
    :param title: plot title
    """

    n_row = 1 if take_mean else embindhist.shape[0]
    if take_mean:
        embindhist = np.mean(embindhist, axis=0, keepdims=True)

    n_index = embindhist.shape[1]
    plt.figure(figsize=(10 * n_index / 136.5, 3 * n_row) if figsize is None else figsize)
    for i in range(n_row):
        plt.subplot(n_row, 1, i + 1)
        plt.plot(np.arange(n_index), embindhist[i], color=color)
        if title:
            if i == 0:
                plt.title(title)
        plt.ylabel("Counts")
        plt.xlim([0, n_index])
        # plt.xticks(np.arange(0, n_index, 100))
    plt.xlabel("Feature index")
    if target_vq_layer == 1:
        plt.tight_layout()
    if savepath:
        if savepath == "default":
            savepath = analytics.model.savepath_dict["ft"]
        plt.savefig(os.path.join(savepath, f"{filename}.png"), dpi=300, facecolor="white",bbox_inches='tight')
    else:
        plt.show()


def plot_markers_umap(analytics,
                      data=None,
                      labels=None,
                      embvec=None,
                      alpha=1.0,
                      target_vq_layer=2,
                      filename="umap_gt",
                      cmap="tab20",
                      colors_dict=None,
                      annotations_font_size=11,
                      to_annot=True,
                      xlim=None,
                      ylim=None,
                      titles=None,
                      subplot_shape=None,
                      gt_table=None,
                      s=10,
                      savefig=False,
                      output_filename="umap",
                      plot_pca=False,
                      seed=1):
    """Plot UMAP in a clean version"""

    label = labels.copy() if labels is not None else analytics.data_manager.test_label
    if isinstance(target_vq_layer, int):
        target_vq_layer = [target_vq_layer]

    gt_table = analytics.gt_table if gt_table is None else gt_table
    gt_name = gt_table.iloc[:, 0]
    uniq_group = np.unique(gt_table.iloc[:, 1])

    if data is None:
        analytics.model.embvec = embvec.copy() if embvec is not None else None
        calc_umap_embvec(analytics, target_vq_layer=target_vq_layer)
        data = analytics.vec_umap

    # make sure data is in a list
    data = data if isinstance(data, list) else [data]
    n_subplots = 0
    for d in data:
        if min(d.shape) > 0:
            n_subplots += 1
    # get subplot shape
    if subplot_shape is None:
        ncol = math.ceil(math.sqrt(n_subplots))
        nrow = math.ceil(n_subplots / ncol)
    else:
        nrow, ncol = subplot_shape
    plt.figure(figsize=(2 * ncol, 2 * nrow))
    subplot_ind = 1
    annots = []
    for vqi, idht in enumerate(data):
        if vqi + 1 in target_vq_layer:
            logging.info(f"Plotting {filename} subplot{vqi} ...")
            plt.subplot(nrow, ncol, subplot_ind)
            if min(idht.shape) > 0:
                if colors_dict is None:
                    if type(cmap) == str:
                        colors = cm.get_cmap(cmap).colors
                    else:
                        # Array
                        colors = []
                        for c in cmap:
                            colors += cm.get_cmap(c).colors
                # plot each cluster layer
                for i, fname in enumerate(tqdm(uniq_group)):
                    group0 = gt_table[gt_table.iloc[:, 1] == fname]
                    ind = np.isin(label[:, 0], group0.iloc[:, 0])
                    data0 = idht[ind]
                    fname = fname.split("_")[1]
                    if colors_dict is None:
                        c = np.array(colors[i]).reshape(1, -1)
                    else:
                        c = np.array([*[colors_dict[fname]] * len(data0)])
                    sctrs = plt.scatter(
                        data0[:, 0],
                        data0[:, 1],
                        s=s,
                        c=c,
                        label=fname,
                        alpha=alpha
                    )
                    if to_annot:
                        km = KMeans(n_clusters=1, random_state=seed).fit(data0)
                        annot = plt.annotate(fname, (km.cluster_centers_[0, 0], km.cluster_centers_[0, 1]),
                                             fontsize=annotations_font_size)
                        annots.append(annot)
                if xlim is not None and xlim[vqi] is not None:
                    plt.xlim(xlim[vqi])
                if ylim is not None and ylim[vqi] is not None:
                    plt.ylim(ylim[vqi])
            hndls, names = sctrs.axes.get_legend_handles_labels()
            hndls = deque(hndls)
            names = deque(names)
            hndls.rotate(-1)
            names.rotate(-1)
            if titles is None:
                plt.title(f"Ground true samples vq{vqi + 1}")
            elif isinstance(titles, str):
                plt.title(f"{titles}")
            else:
                plt.title(titles[vqi])
            # plt.ylabel("Umap 2")
            # plt.xlabel("Umap 1")
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout(pad=0.5)

    if savefig:
        plt.savefig(f"./umaps/{output_filename}_umap.png")
    plt.show()

    if plot_pca:
        embs = analytics.model.embvec[1]
        embs = embs.reshape(embs.shape[0], -1)
        embs = StandardScaler().fit_transform(embs.copy())
        pca = PCA()
        principal_components = pca.fit_transform(embs)
        principal_df = pd.DataFrame(data=principal_components[:, [0, 1]],
                                    columns=['principal component 1', 'principal component 2'])
        y = pd.DataFrame(label, columns=['group'])
        final_df = pd.concat([principal_df, y], axis=1)
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(f'PC 1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)', fontsize=15)
        ax.set_ylabel(f'PC 2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)', fontsize=15)
        ax.set_title(titles, fontsize=15)
        final_df['group'] = final_df['group'].str.split('_', 1, expand=True)[1]
        for group in names:
            indicesToKeep = final_df['group'] == group
            ax.scatter(final_df.loc[indicesToKeep, 'principal component 1']
                       , final_df.loc[indicesToKeep, 'principal component 2']
                       , s=50,
                       c=np.array([*[colors_dict[group]] * len(final_df.loc[indicesToKeep, 'principal component 1'])]))
        if savefig:
            plt.savefig(f"./umaps/{output_filename}_pca.png", bbox_inches='tight')
        plt.show()

        # pca_=PCA()
        # pca.fit_transform(emb)
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
                label='Individual explained variance')
        plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
                 label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.title(titles)
        plt.legend(loc='best')
        plt.tight_layout()
        if savefig:
            plt.savefig(f"./umaps/{output_filename}_pca_variance.png", bbox_inches='tight')
        plt.show()

    return data, label


def plot_kmeans_pca(analytics, labels, cluster_sizes):
    # label = labels.copy() if labels is not None else analytics.data_manager.test_label
    # gt_table = analytics.gt_table if gt_table is None else gt_table
    # uniq_group = np.unique(gt_table.iloc[:, 1])
    calc_umap_embvec(analytics, target_vq_layer=2)

    label = labels.copy()
    label = pd.DataFrame(label, columns=["full_label"])
    label[['marker', 'cell_line']] = label['full_label'].str.split('_', 1, expand=True)
    x = analytics.model.embvec[1]
    x = x.reshape(x.shape[0], -1)

    wcss = []
    for i in cluster_sizes:
        model = KMeans(n_clusters=i, init="k-means++")
        model.fit(x)
        wcss.append(model.inertia_)
    plt.figure(figsize=(10, 10))
    plt.plot(cluster_sizes, wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    pca = PCA(2)
    data = pca.fit_transform(x)
    plt.figure(figsize=(5, 5))
    var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    lbls = [str(x) for x in range(1, len(var) + 1)]
    plt.bar(x=range(1, len(var) + 1), height=var, tick_label=lbls)
    plt.ylabel("explained variance")
    plt.show()

    clustering_options = {4: 'cell_line', 28: 'marker', 112: 'full_label'}
    clusters_results = []
    for i in cluster_sizes:
        # find clusters and plot
        model = KMeans(n_clusters=i, init="k-means++")
        clusters = model.fit_predict(data)
        clusters_results.append(clusters)
        plt.figure(figsize=(5, 5))
        uniq = np.unique(clusters)
        centroids = model.cluster_centers_
        for j, uni in enumerate(uniq):
            plt.scatter(data[clusters == uni, 0], data[clusters == uni, 1], label=uni, s=20)
            if i <= 28:
                plt.annotate(uni, (centroids[j, 0], centroids[j, 1]))
        # plt.legend()
        plt.title(f"{i} Clusters")
        plt.show()

        # plot ground truth
        principal_df = pd.DataFrame(data=data,
                                    columns=['principal component 1', 'principal component 2'])
        y = label[clustering_options[i]]
        final_df = pd.concat([principal_df, y], axis=1)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(f'PC 1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)')
        ax.set_ylabel(f'PC 2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)')
        ax.set_title(f'PCA of {clustering_options[i]}')
        # final_df['group'] = final_df['group'].str.split('_', 1, expand=True)[1]
        for group in np.unique(y):
            indicesToKeep = final_df[clustering_options[i]] == group
            ax.scatter(final_df.loc[indicesToKeep, 'principal component 1']
                       , final_df.loc[indicesToKeep, 'principal component 2']
                       , s=20)
        # ax.legend(np.unique(y))
        # c=np.array([*[colors_dict[group]] * len(final_df.loc[indicesToKeep, 'principal component 1'])]))
        plt.tight_layout()
        plt.show()

    return clusters_results


def plot_kmeans_umap(analytics, cluster_sizes):
    calc_umap_embvec(analytics, target_vq_layer=2)
    data = analytics.vec_umap

    clusters_results = []
    two_umaps = data[1][:, :2]  # extract global representation and then extract 2 first umaps
    for i in cluster_sizes:
        # find clusters and plot
        model = KMeans(n_clusters=i, init="k-means++")
        clusters = model.fit_predict(two_umaps)
        clusters_results.append(clusters)
        plt.figure(figsize=(5, 5))
        uniq = np.unique(clusters)
        centroids = model.cluster_centers_
        for j, uni in enumerate(uniq):
            plt.scatter(two_umaps[clusters == uni, 0], two_umaps[clusters == uni, 1], label=uni, s=20)
            if i <= 28:
                plt.annotate(uni, (centroids[j, 0], centroids[j, 1]))
        # plt.legend()
        plt.title(f"{i} Clusters")
        plt.show()

    return clusters_results


def arrange_plots(input_folder, nrows:int, ncols:int, file_name_contains=None, order=None):
    """Arrange multiple plots in a single plot

    Args:
        input_folder (string): Path to the input folder containing the plots
        nrows (int): Number of rows in the final plot
        ncols (int): Number of columns in the final plot
        file_name_contains (string, optional): Take only files containing this str in their name. Defaults to None.
        order (list, optional): The order the files should be taken by. Defaults to None.
    """
    plots = []
    if not order:
        files = sorted(filter(lambda x: os.path.isfile(os.path.join(input_folder, x)),
                              os.listdir(input_folder)))
    else:
        tmp_files = os.listdir(input_folder)
        files = []
        for cur_order in order:
            files += [i for i in tmp_files if cur_order in i]
    for file in files:
        if file_name_contains:
            if file_name_contains not in file:
                continue
        if "png" not in file or 'arranged' in file:
            continue
        plots.append(Image.open(os.path.join(input_folder, file)))
    if len(plots) == 0:
        logging.info("found no plots")
        return
    if nrows * ncols != len(plots):
        logging.info("number of plots doesn't match nrows*ncols")
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 7.5), dpi=300)
    for row in range(nrows):
        for col in range(ncols):
            axs[row, col].imshow(plots[row * ncols + col])
            for spine in ['top', 'right', 'bottom', 'left']:
                axs[row, col].spines[spine].set_visible(False)
            axs[row, col].get_xaxis().set_ticks([])
            axs[row, col].get_yaxis().set_ticks([])
    plt.tight_layout(w_pad=0, pad=0)
    plt.savefig(f"{input_folder}/arranged_{file_name_contains.replace('.', '')}.png")
    plt.show()
