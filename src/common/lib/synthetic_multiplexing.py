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
                    output_layer='vqvec2',
                    savepath='default',
                    map_labels_function=None,
                    config_data=None):
    assert model is not None, "Model is None"
    assert config_data is not None, 'config_data is None'
    
    dataset_conf = config_data

    embeddings, labels = __get_embeddings(model, embeddings_type, config_data=dataset_conf, vq_type=output_layer)
    logging.info(f"[Before concat] Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
    
    df = __embeddings_to_df(embeddings, labels, dataset_conf,  vq_type=output_layer)
    embeddings, label_data, unique_groups = __get_multiplexed_embeddings(df, random_state=dataset_conf.SEED)
    logging.info(f"[After concat] Embeddings shape: {embeddings.shape}, Labels shape: {label_data.shape}")
    
    if map_labels_function is not None:
        logging.info("Applyging map_labels_function from the config on the unique_groups")
        unique_groups = map_labels_function(unique_groups)    
    
    logging.info("Generating dummy analytics..")
    model.generate_dummy_analytics()
    
    logging.info("Plot umap..")
    model.plot_umap(colormap=colormap,
                    alpha=alpha,
                    s=s,
                    label_data=label_data,
                    id2label=None,
                    title=title if title is not None else __generate_plot_title(model.conf, dataset_conf),
                    unique_groups=unique_groups,
                    embedding_data=embeddings,
                    output_layer=output_layer,
                    savepath=savepath,
                    map_labels_function=map_labels_function,
                    config_data=config_data)

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

def __format_labels_to_marker_and_pheno(label, config_data, vq_type):        
    label_split = label.split('_')
    
    if vq_type in ['vqindhist1', 'vqindhist2']:
        pheno = label_split[-4 :-2 + int(config_data.ADD_BATCH_TO_LABEL)]
        if config_data.ADD_REP_TO_LABEL:
            pheno += [label_split[-1]]
        return (label_split[0], '_'.join(pheno))
    
    if vq_type in ['vqvec1', 'vqvec2']:
        return (label_split[-1], '_'.join(label_split[-4 + int(not config_data.ADD_REP_TO_LABEL):-1]))
    
    raise f"Invalid vq type {vq_type} [The options are: 'vqvec1', 'vqvec2', 'vqindhist1', 'vqindhist2']"


def __embeddings_to_df(embeddings, labels, dataset_conf, vq_type='vqvec2'):
    labels_df = pd.DataFrame([__format_labels_to_marker_and_pheno(s, dataset_conf, vq_type) for s in labels], columns=['Marker', 'Pheno'])
    embeddings_series = pd.DataFrame({"Embeddings": [*embeddings]})
    df = pd.merge(labels_df, embeddings_series, left_index=True, right_index=True)
    return df

def __get_embeddings(model, embeddings_type, config_data, vq_type='vqvec2'):
    logging.info(f"Loading embeddings... (vq type: {vq_type})")
    
    if vq_type in ['vqindhist1', 'vqindhist2']:
        loading_func = lambda: model.load_indhists(embeddings_type, config_data)
    elif vq_type in ['vqvec1', 'vqvec2']:
        loading_func = lambda: model.load_embeddings(embeddings_type, config_data)
    else:
        raise f"Invalid vq type {vq_type} [The options are: 'vqvec1', 'vqvec2', 'vqindhist1', 'vqindhist2']"
    
    embeddings, labels = loading_func()
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
