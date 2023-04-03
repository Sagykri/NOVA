import os
import random
import numpy as np
import pandas as pd
from cytoself_custom import calc_umap_embvec, plot_umap, plot_markers_umap
from lib.metrics import calc_metrics
from metrics import plot_metrics
from umap import UMAP
import re
from matplotlib import cm
import matplotlib.pyplot as plt
from configs.model_config import SEED, COLORS_MAPPING, TEMR_LEGEND_MAPPING
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def multiplex(model, 
              groups_terms,match=False,
              plot_pca=False, is_comb=False, legend_inside=True,
              y1=None, cmap1=["tab20", "tab20b"], title1="protein localization (before SM)", annot_font_size1=None,
              colors_dict1=None, s1=None, alpha=0.5,
              title2="Proteins localization", s2=30, show1=True, show=True, reset_embvec=True, verbose=True,
              filename="multiplexed_data", savefig=True, first_umap_output_name="umap", second_umap_figsize=(4, 4),
              first_umap_figsize=(8, 6)):
    
    
    labels_changepoints = model.test_labels_changepoints
    X                   = model.test_data
    y                   = model.test_label
    markers             = model.markers
    markers_order       = model.test_markers_order
    analytics           = model.analytics
    
    
    if match and labels_changepoints is None:
        raise "label_changepoints can't be None if match is True"

    markers = [m for m in markers_order if m in markers]
    is_contains_re = np.vectorize(lambda y, x: re.match(x, y))

    if y1 is None:
        y1 = y

    y = y.copy()
    y1 = y1.copy()

    ########################## TODO: DELETE ##########################

    # Set all labels other than markers to None so they will be skipped
    # labels_pd = pd.Series(y.reshape(-1,))
    # markers_indexes = labels_pd.index

    # markers_regex = '|'.join(list(map(lambda x: f"^{x}$|^{x}_", markers)))
    # markers_indexes = labels_pd[labels_pd.str.contains(markers_regex)].index

    # all_markers = labels_pd.value_counts().index
    # excluded_markers = all_markers[~all_markers.str.contains(markers_regex)].tolist()
    # print(f"Excluded: {excluded_markers} (Count: {len(y)-len(markers_indexes)})")
    # y1[~np.isin(np.arange(len(y1)), markers_indexes)] = None
    # y[~np.isin(np.arange(len(y)), markers_indexes)] = None

    ####################################################################

    if reset_embvec:
        # Take only images of these markers and recalculate the embvec
        analytics.model.embvec = None
        analytics.model.calc_embvec(X, savepath="default", filename=f"{filename}_embvec_1u")
        calc_umap_embvec(analytics, target_vq_layer=2, savepath="default", filename=f"{filename}_umap_1u")

    # Concatenate:
    groups = None
    labels_counts = np.zeros((len(groups_terms),), dtype=int)
    if is_comb:
        y = pd.Series(y.reshape(-1, )).replace(
            {"NCL": "Nucleolin", "SNCA": "SCNA", "syto12": "Syto12", "phalloidin": "Phalloidin"}, regex=True).to_numpy().reshape(-1,1)
    for j, term in enumerate(tqdm(groups_terms)):

        y_group = np.where(is_contains_re(y, f'.*{term}.*'), y, None)

        if all(y_group == None):
            if verbose:
                print(f"{term} not found")
            continue
        group_size = pd.Series(y_group.reshape(-1, )).value_counts().min()
        for i in range(group_size):
            group = []
            selected_index = None
            selected_index_paired = None

            for l, m in enumerate(markers):

                # Get all options for current marker
                # inner_indexes = np.where(y_group == f"{m}{term}")[0]
                inner_indexes = np.where(pd.Series(y_group[:, 0]).str.contains(f'{m}{term}'))[0]
                if len(inner_indexes) == 0:
                    if verbose:
                        print(f"Skipping {m}{term}")
                    continue

                # Select image
                if match:
                    if selected_index is None:
                        selected_index = random.choice(inner_indexes)
                        selected_index_paired = _get_tiles_ids(selected_index, len(markers), labels_changepoints)

                    index = selected_index_paired[l]
                else:
                    index = random.choice(inner_indexes)

                # Remove this choice from ever to be chosen again
                y_group[index] = None
                # Concatenate current image's embvec1 with the previous images' embvec
                group.append(analytics.model.embvec[1][index].reshape(-1, ))

            # Flat
            group = [item for sublist in group for item in sublist]
            group = np.array(group)

            if groups is None:
                groups = group.copy()
            else:
                groups = np.vstack((groups, group))
            labels_counts[j] += 1

    if verbose:
        print(f"Groups Counts: {labels_counts}")

    reducer = UMAP(random_state=SEED)
    X_transformed = reducer.fit_transform(groups)
    y_transformed = [[groups_terms[i]] * labels_counts[i] for i in range(len(labels_counts))]
    # Flat
    y_transformed = [item for sublist in y_transformed for item in sublist]
    # To numpy array
    y_transformed = np.array(y_transformed)

    if not show:
        return X_transformed, y_transformed

    # Plot first UMAP

    if show1:
        d1, l1 = plot_umap(analytics, data=analytics.vec_umap, cmap=cmap1, s=s1, annotations_font_size=annot_font_size1,
                           colors_dict=colors_dict1, titles=title1, savefig=savefig,
                           output_filename=first_umap_output_name,
                           plot_pca=plot_pca, figsize=first_umap_figsize)
        plt.show()

    # Plot second (grouped) UMAP

    n_clusters = len(groups_terms)
    plt.figure(figsize=second_umap_figsize)
    scs = []
    i = 0
    for c in range(n_clusters):
        sc = plt.scatter(
            X_transformed[i:i + labels_counts[c], 0],
            X_transformed[i:i + labels_counts[c]:, 1],
            s=s2,
            c=np.array([*[COLORS_MAPPING[groups_terms[c]]] * labels_counts[c]]),
            alpha=alpha
        )
        scs.append(sc)
        i += labels_counts[c]

    legend = np.array([TEMR_LEGEND_MAPPING[l] for l in groups_terms])
    if legend_inside:
        plt.legend(scs, legend, loc='upper left', ncol=2, borderaxespad=3)
    else:
        plt.legend(scs, legend)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.xticks([])
    plt.yticks([])
    plt.title(title2)

    plt.tight_layout(pad=0.5)
    plt.show()

    print("Metrics:")
    print("(1)")
    embvec_flatten = analytics.model.embvec[1].copy()
    embvec_flatten = embvec_flatten.reshape(embvec_flatten.shape[0], -1)
    plot_metrics(embvec_flatten, y, n_clusters=len(np.unique(y)))
    print("(2)")
    plot_metrics(groups, y_transformed, n_clusters=n_clusters)

    np.save(os.path.join(analytics.model.savepath_dict["emb"], f"{filename}_embvec_2u.npy"), groups)
    np.save(os.path.join(analytics.model.savepath_dict["emb"], f"{filename}_umap_2u.npy"), X_transformed)

    # plot PCA
    if plot_pca:
        pca = PCA()
        x = StandardScaler().fit_transform(groups.copy())
        principal_components = pca.fit_transform(x)
        principal_df = pd.DataFrame(data=principal_components[:, [0, 1]],
                                    columns=['principal component 1', 'principal component 2'])
        y = pd.DataFrame(y_transformed, columns=['group'])
        final_df = pd.concat([principal_df, y], axis=1)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(f'PC 1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)', fontsize=15)
        ax.set_ylabel(f'PC 2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)', fontsize=15)
        for cell_line in groups_terms:
            indicesToKeep = final_df['group'] == cell_line
            ax.scatter(final_df.loc[indicesToKeep, 'principal component 1']
                       , final_df.loc[indicesToKeep, 'principal component 2']
                       , c=COLORS_MAPPING[cell_line]
                       , s=50)
        # ax.legend(legend)
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
    return X_transformed, y_transformed


def _get_tiles_ids(tile_id, n_markers, mapping):
    mapping_np = np.array(mapping)
    transition_index = np.array([mapping_np[i] for i in range(len(mapping_np)) if i % n_markers == 0])
    start_index_transition = np.argwhere(tile_id >= transition_index)[-1][0]
    end_index_transition = np.argwhere(tile_id < transition_index)[0][0]
    start_index_mapping = np.argwhere(mapping_np == transition_index[start_index_transition])[0][0]
    end_index_mapping = np.argwhere(mapping_np == transition_index[end_index_transition])[0][0]

    blocks = []
    for i in range(start_index_mapping, end_index_mapping):
        blocks.append(mapping[i])

    nearest_block = np.argwhere(tile_id >= mapping_np)[-1][0]
    delta = tile_id - mapping[nearest_block]

    indexes = []
    for i in range(len(blocks)):
        indexes.append(blocks[i] + delta)

    # return indexes
    indexes.sort()
    return np.unique(indexes)



def calc_bootstrapping(model, groups_terms, n_runs=1000, save_folder=None):

    X                   = model.test_data
    y                   = model.test_label
    markers             = model.markers
    markers_order       = model.test_markers_order
    analytics           = model.analytics
    
    
    n_clusters = 2
    reset_embvec = False
    metrics_random = []
    metrics_match = []
    
    if not os.path.exists(save_folder):
        print(f"{save_folder} wasn't found. Creating it..")
        os.makedirs(save_folder)

    for i in range(n_runs):
        print(f"{i}/{(n_runs-1)}", flush=True)
        data, labels = multiplex(analytics, X=X,
                                y=y, markers_order=markers_order, groups_terms=groups_terms,
                                markers = markers, show=False,
                                match=False,reset_embvec=(reset_embvec & (i==0))
                                )

        adjusted_rand_score_val_j, silhouette_score_val_j = calc_metrics(data, labels.reshape(-1,), n_clusters=n_clusters)

        metrics_random.append([adjusted_rand_score_val_j, silhouette_score_val_j])

        if i == 0:
            data, labels = multiplex(analytics, X=X,
                                    y=y, markers_order=markers_order, groups_terms=groups_terms,
                                    markers = markers,show=False,
                                        match=True,reset_embvec=False
                                    )

            adjusted_rand_score_val_j, silhouette_score_val_j = calc_metrics(data, labels.reshape(-1,), n_clusters=n_clusters)

            metrics_match[i].append([adjusted_rand_score_val_j, silhouette_score_val_j])

        if i % 10 == 0:
            print(f"Saving {i}", flush=True)
            with open(os.path.join(save_folder,'scores_random-{i}.npy'), 'wb') as f:
                np.save(f, np.array(metrics_random))

            with open(os.path.join(save_folder,'scores_match-{i}.npy'), 'wb') as f:
                np.save(f, np.array(metrics_match))
            
    print(f"Saving Final", flush=True)
    with open(model.conf.METRICS_RANDOM_PATH, 'wb') as f:
        np.save(f, np.array(metrics_random))

    with open(model.conf.METRICS_MATCH_PATH, 'wb') as f:
        np.save(f, np.array(metrics_match))

    return np.array(metrics_random), np.array(metrics_match)

def plot_boostrapping(metrics_random=None, metrics_match=None, metrics_random_path=None, metrics_match_path=None):
    metrics_random = metrics_random if metrics_random is not None else np.load(metrics_random_path)
    metrics_match = metrics_match if metrics_match is not None else np.load(metrics_match_path)
    
    n_runs = len(metrics_random)
    
    plt.title("Bootstrapping")
    plt.plot(np.arange(n_runs), [metrics_match[0,0]]*len(metrics_random), color='red')
    plt.scatter(np.arange(n_runs), metrics_random[:,0], c='grey')
    plt.xlabel("Bootstrap sample")
    plt.ylabel("ARI")
    plt.show()

    plt.title("Bootstrapping")
    plt.plot(np.arange(n_runs), [metrics_match[0,1]]*len(metrics_random), color='red')
    plt.scatter(np.arange(n_runs), metrics_random[:,1], c='grey')
    plt.xlabel("Bootstrap sample")
    plt.ylabel("Silhouette")
    plt.show()
