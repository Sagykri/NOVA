import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import pandas as pd
import logging
import  torch
import datetime

from src.common.lib.dataset import Dataset
from src.common.lib.utils import get_if_exists, load_config_file
from src.common.lib.model import Model
from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.feature_spectra_utils import *

def generate_umaps():
    
    if len(sys.argv) < 3:
        raise ValueError("Invalid config path. Must supply model config and data config.")
    
    config_path_model = sys.argv[1]
    config_path_data = sys.argv[2]

    config_model = load_config_file(config_path_model, 'model')
    config_data = load_config_file(config_path_data, 'data', config_model.CONFIGS_USED_FOLDER)
    output_folder_path = sys.argv[3] #if len(sys.argv) > 3 else config_model.MODEL_OUTPUT_FOLDER
    delta = True if len(sys.argv)>4 else False
    
    if not os.path.exists(output_folder_path):
        logging.info(f"{output_folder_path} doesn't exists. Creating it")
        os.makedirs(output_folder_path)
    
    # assert os.path.isdir(output_folder_path) and os.path.exists(output_folder_path), f"{output_folder_path} is an invalid output folder path or doesn't exists"

    logging.info("init")
    logging.info("[Generate UMAP1]")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
        
    __unique_labels_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, "unique_labels.npy")
    if os.path.exists(__unique_labels_path):
        logging.info(f"unique_labels.npy files has been detected - using it. ({__unique_labels_path})")
        unique_markers = np.load(__unique_labels_path)
    else:
        logging.warn(f"Couldn't find unique_labels file: {__unique_labels_path}")
        raise Exception(f"Couldn't find unique_labels file: {__unique_labels_path}")
    
    logging.info("Init model")
    model = Model(config_model)
    
    __generate_with_load(config_model, config_data, model, output_folder_path, delta)
    return None

def calculate_difference(group, first_cond = 'stress', second_cond='Untreated'):
	# Separate the DataFrame into stress and untreated groups
	first_df = group[group['label'].str.endswith(f'_{first_cond}')]
	second_df = group[group['label'].str.endswith(f'_{second_cond}')]
	# Randomly sort the DataFrames
	first_df = first_df.sample(frac=1).reset_index(drop=True)
	second_df = second_df.sample(frac=1).reset_index(drop=True)
    # remove labels columns
	first_df.drop(columns=['label','marker', 'path'], inplace=True)
	second_df.drop(columns=['label','marker', 'path'], inplace=True)
 
	# Cut the DataFrames to have the same size
	min_size = min(len(first_df), len(second_df))
	first_df = first_df.head(min_size)
	second_df = second_df.head(min_size)
	# Calculate the difference and save it
	delta = first_df - second_df
	return delta

def generate_deltas(embeddings, labels, first_cond = 'stress', second_cond='Untreated'):
    df = create_vqindhists_df([embeddings], [labels], [labels], arange_labels=False)
    df['label'] = df['label'].str.split("_").str[0:3:2].apply(lambda x: '_'.join(x)) # merging different batches and reps -> label == marker_cond
    # # first, create the mean deltas for ref
    # total_spectra_per_marker_ordered = df.groupby('label').mean()
    # total_spectra_per_marker_ordered['marker'] = total_spectra_per_marker_ordered.index.str.split('_').str[0]
    # average_deltas = total_spectra_per_marker_ordered.groupby('marker').diff(axis=0).dropna()
    # mean_embeddings = np.array(average_deltas)
    # mean_labels = average_deltas.index.str.split('_').str[0].to_list()
    # mean_labels = np.array([label + "_mean" for label in mean_labels])

    # now, we want to randomly choose couples from the same marker and to diff them
    df['marker'] = df.label.str.split('_').str[0]
    deltas = []
    for marker, marker_group in df.groupby('marker'):
        cur_delta = calculate_difference(marker_group, first_cond = first_cond, second_cond=second_cond)
        cur_delta['marker'] = marker
        deltas.append(cur_delta)
    deltas = pd.concat(deltas)
    deltas_embeddings = np.array(deltas.drop(columns=['marker']))
    deltas_labels = np.array(deltas.marker)
    embeddings = deltas_embeddings #np.concatenate((mean_embeddings, deltas_embeddings))
    labels = deltas_labels #np.concatenate((mean_labels, deltas_labels))

    return embeddings, labels


def __generate_with_load(config_model, config_data, model, output_folder_path, delta=False):
    logging.info("Clearing cache")
    torch.cuda.empty_cache()
    
    __now = datetime.datetime.now()
    
    model.generate_dummy_analytics()
    embeddings, labels = model.load_indhists(embeddings_type='testset' if config_data.SPLIT_DATA else 'all',
                                               config_data=config_data)
    logging.info(f'[__generate_with_load]: embeddings shape: {embeddings.shape}, labels shape: {labels.shape}')
    if delta:
        embeddings, labels = generate_deltas(embeddings, labels)   
        logging.info(f'[__generate_with_load]: doing deltas: embeddings shape: {embeddings.shape}, labels shape: {labels.shape}, unique labels: {np.unique(labels)}')

    logging.info(f"Plot umap...")
    title = f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}"
    savepath = os.path.join(output_folder_path,\
                            'UMAPs',\
                            'UMAP1'
                                f'{__now.strftime("%d%m%y_%H%M%S_%f")}_{os.path.splitext(os.path.basename(config_model.MODEL_PATH))[0]}',\
                                    f'{title}')
        
    __savepath_parent = os.path.dirname(savepath)
    if not os.path.exists(__savepath_parent):
        os.makedirs(__savepath_parent)
        
    colormap = get_if_exists(config_data, 'COLORMAP', 'Set1')
    size = get_if_exists(config_data, 'SIZE', 0.8)
    alpha = get_if_exists(config_data, 'ALPHA', 0.7)
    map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
    if map_labels_function is not None:
        map_labels_function = eval(map_labels_function)(config_data)

    ordered_marker_names = ["DAPI", 'TDP43', 'PEX14', 'NONO', 'ANXA11', 'FUS', 'Phalloidin', 
                            'PURA', 'mitotracker', 'TOMM20', 'NCL', 'Calreticulin', 'CLTC', 'KIF5A', 'SCNA', 'SQSTM1', 'PML',
                            'DCP1A', 'PSD95', 'LAMP1', 'GM130', 'NEMO', 'CD41', 'G3BP1']
    ordered_names = [config_data.UMAP_MAPPINGS[marker]['alias'] for marker in ordered_marker_names]
    model.plot_umap(embedding_data=embeddings,
                    label_data=labels,
                    title=title,
                    savepath=savepath,
                    colormap=colormap,
                    alpha=alpha,
                    s=size,
                    reset_umap=True,
                    map_labels_function=map_labels_function,
                    config_data=config_data,
                    ordered_names = ordered_names,
                    outliers_fraction=0.01)
    
    logging.info(f"UMAP saved successfully to {savepath}")
    return None

if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        generate_umaps()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")