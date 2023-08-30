import datetime
import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import logging
import random
import numpy as np
import pandas as pd
from umap import UMAP
import re
import matplotlib.pyplot as plt
from src.common.lib.model import Model
from src.common.lib.utils import flat_list_of_lists, get_if_exists


def multiplex(model: Model, embeddings_type='testset',
                    title=None,
                    colormap='Set1',
                    alpha=0.8,
                    s=0.8,
                    output_layer='vqvec2'):
    assert model is not None, "Model is None"
    assert model.test_loader is not None, "model.test_loader is None, please first load dataloaders"
    
    dataset_conf = model.test_loader.dataset.conf
    calc_embeddings = get_if_exists(dataset_conf, 'CALCULATE_EMBEDDINGS', False)
    logging.info(f"calc_embeddings is set to {calc_embeddings}")
    
    embeddings, labels = __get_embeddings(model, embeddings_type, calc_embeddings, output_layer)
    logging.info(f"[Before concat] Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
    
    df = __embeddings_to_df(embeddings, labels, dataset_conf)
    embeddings, label_data, unique_groups = __get_multiplexed_embeddings(df, random_state=dataset_conf.SEED)
    logging.info(f"[After concat] Embeddings shape: {embeddings.shape}, Labels shape: {label_data.shape}")
    
    logging.info("Loading analytics..")
    model.load_analytics()
    
    logging.info("Plot umap..")
    model.plot_umap(colormap=colormap,
                    alpha=alpha,
                    s=s,
                    label_data=label_data,
                    id2label=None,
                    title=title if title is not None else __generate_plot_title(model.conf, dataset_conf),
                    unique_groups=unique_groups,
                    embedding_data=embeddings,
                    output_layer=output_layer)

def __generate_plot_title(model_conf, dataset_conf):
    return 'SM_' + f"{'_'.join([os.path.basename(f) for f in dataset_conf.INPUT_FOLDERS])}_{datetime.datetime.now().strftime('%d%m%y_%H%M%S_%f')}_{os.path.splitext(os.path.basename(model_conf.MODEL_PATH))[0]}"

def __get_multiplexed_embeddings(embeddings_df, random_state=None):
    grouped_by_pheno = embeddings_df.groupby('Pheno')
    common_markers = set.intersection(*map(set, grouped_by_pheno['Marker'].unique()))
    logging.info(f"[SM] Common markers: {common_markers}")
    
    def __apply_func(df):
        df = df[df['Marker'].isin(common_markers)]
        return __concatenate_embeddings_by_group(df, random_state=random_state)
    
    # # Group by "Pheno" and apply the custom function to each group
    result_df = grouped_by_pheno.apply(__apply_func).reset_index(drop=True)
    
    embeddings = np.vstack(result_df['Embeddings'].to_numpy())
    unique_groups = result_df['Pheno'].to_numpy().reshape(-1)
    
    label_data = []
    for _, row in result_df.iterrows():
        label_data.append([row['Pheno']] * row['Embeddings'].shape[0])
    
    label_data = flat_list_of_lists(label_data)
    label_data = np.asarray(label_data).reshape(-1,1)
    
    return embeddings, label_data, unique_groups

def __embeddings_to_df(embeddings, labels, dataset_conf):
    calc_embeddings = get_if_exists(dataset_conf, 'CALCULATE_EMBEDDINGS', False)
    if calc_embeddings:
        labels_df = pd.DataFrame([s.split('_', 1) for s in labels], columns=['Marker', 'Pheno'])
    else:
        labels_df = pd.DataFrame([(s.split('_')[-1], '_'.join(s.split('_')[-4:-2 + int(dataset_conf.ADD_REP_TO_LABEL)])) for s in labels], columns=['Marker', 'Pheno'])
    embeddings_series = pd.DataFrame({"Embeddings": [*embeddings]})
    df = pd.merge(labels_df, embeddings_series, left_index=True, right_index=True)
    return df

def __get_embeddings(model, embeddings_type, calc_embeddings, output_layer: str = 'vqvec2'):
    if calc_embeddings:
        logging.info("Calculating embeddings...")
        embeddings, labels_ids = model.model.infer_embeddings(model.test_loader, output_layer)
        labels = model.test_loader.dataset.id2label(labels_ids)
    else:
        logging.info("Loading embeddings...")
        embeddings, labels = model.load_embeddings(embeddings_type)
        labels = np.asarray(labels).reshape(-1,)
    return embeddings,labels

def __concatenate_embeddings_by_group(group, random_state=None):
    group_copy = group.copy()
    pheno = group['Pheno'].iloc[0]
    logging.info(f"Pheno: {pheno}")
    
    #Shuffle
    group_copy.sample(frac=1, random_state=random_state)
    
    n_subgroups = min(group_copy['Marker'].value_counts())
    logging.info(f"Detected {n_subgroups} subgroups")
    embeddings = []
    for i in range(n_subgroups):
        logging.info(f"{i+1}/{n_subgroups}")
        logging.info(f"[{i+1}/{n_subgroups}] Shape: {group_copy.shape}")
        subgroup = group_copy\
                    .groupby('Marker')\
                    .sample(n=1, replace=False, random_state=random_state)
        
        subgroup.sort_values('Marker', inplace=True)
        
        __subgroup_embeddings = np.stack(subgroup['Embeddings'].to_numpy(), axis=0)
        subgroup_embeddings = np.concatenate([e.reshape(-1) for e in __subgroup_embeddings])
        embeddings.append(subgroup_embeddings)
        
        group_copy.drop(index=subgroup.index, inplace=True)
    
    embeddings = np.stack(embeddings, axis=0)
    
    logging.info(f"[{pheno}] embeddings shape: {embeddings.shape}")
    
    return pd.Series({
        'Pheno': pheno, 
        'Embeddings': embeddings
    })

def old_calc_bootstrapping(model:Model, groups_terms, n_runs=1000, save_folder=None):
    """Calculate metrics (ARI and silhouette) with bootstrapping

    Args:
        model (Model): The model
        groups_terms ([string]): list of terms to group by
        n_runs (int, optional): Number of repeated runs in the bootstrapping. Defaults to 1000.
        save_folder (string, optional): Where to save the results. Defaults to None.

    Returns:
        _type_: (Scores for random matching, Scores for actual matching)
    """

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
        logging.info(f"{save_folder} wasn't found. Creating it..")
        os.makedirs(save_folder)

    for i in range(n_runs):
        logging.info(f"{i}/{(n_runs-1)}")
        data, labels = multiplex(analytics, X=X,
                                y=y, markers_order=markers_order, groups_terms=groups_terms,
                                markers = markers, show2=False,
                                match=False,reset_embvec=(reset_embvec & (i==0))
                                )

        adjusted_rand_score_val_j, silhouette_score_val_j = calc_clustering_validation(data, labels.reshape(-1,), n_clusters=n_clusters)

        metrics_random.append([adjusted_rand_score_val_j, silhouette_score_val_j])

        if i == 0:
            data, labels = multiplex(analytics, X=X,
                                    y=y, markers_order=markers_order, groups_terms=groups_terms,
                                    markers = markers,show2=False,
                                        match=True,reset_embvec=False
                                    )

            adjusted_rand_score_val_j, silhouette_score_val_j = calc_clustering_validation(data, labels.reshape(-1,), n_clusters=n_clusters)

            metrics_match[i].append([adjusted_rand_score_val_j, silhouette_score_val_j])

        if i % 10 == 0:
            logging.info(f"Saving {i}")
            with open(os.path.join(save_folder,'scores_random-{i}.npy'), 'wb') as f:
                np.save(f, np.array(metrics_random))

            with open(os.path.join(save_folder,'scores_match-{i}.npy'), 'wb') as f:
                np.save(f, np.array(metrics_match))
            
    logging.info(f"Saving Final")
    with open(model.conf.METRICS_RANDOM_PATH, 'wb') as f:
        np.save(f, np.array(metrics_random))

    with open(model.conf.METRICS_MATCH_PATH, 'wb') as f:
        np.save(f, np.array(metrics_match))

    return np.array(metrics_random), np.array(metrics_match)

def old_plot_boostrapping(metrics_random=None, metrics_match=None, metrics_random_path=None, metrics_match_path=None):
    """Plot bootstrapping results

    Args:
        metrics_random (list, optional): The metrics for random matching. Defaults to None.
        metrics_match (list, optional): The metrics for actual matching. Defaults to None.
        metrics_random_path (int, optional): Path to the file holds the metrics from random matching. Defaults to None.
        metrics_match_path (string, optional): Path to the file holds the metrics from actual matching. Defaults to None.
    """
    
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
