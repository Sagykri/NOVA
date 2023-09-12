import logging
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.lib import synthetic_multiplexing
from src.common.lib.embeddings_utils import load_embeddings
from src.common.lib.utils import get_if_exists, load_config_file

def generate_features():
    if len(sys.argv) < 4:
        raise ValueError("Invalid config path. Must supply model config, data config and n_components for UMAP.")
    
    config_path_model = sys.argv[1]
    config_path_data = sys.argv[2]
    n_components = sys.argv[3]

    config_model = load_config_file(config_path_model, 'model')
    config_data = load_config_file(config_path_data, 'data', config_model.CONFIGS_USED_FOLDER)

    try:
        n_components = int(n_components)
    except Exception as e:
        logging.error("n_components is not a valid number")
        raise e

    logging.info("init")
    logging.info("[Generate Features]")
    
    # Load embeddings
    embeddings, labels = __load_embeddings(config_data, config_model)
    logging.info(np.unique(labels))

    # Synthetic multiplexing    
    folder_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'features')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    embeddings_layer = get_if_exists(config_data, 'EMBEDDINGS_LAYER', 'vqvec2')
    savepath_postfix = f"{'_'.join([os.path.basename(p) for p in config_data.INPUT_FOLDERS])}_{embeddings_layer}"

    embeddings_savepath = os.path.join(folder_path, f"sm_embeddings_{savepath_postfix}")
    labels_savepath = os.path.join(folder_path, f"sm_labels_{savepath_postfix}")
    
    if os.path.exists(f"{embeddings_savepath}.npy") and os.path.exists(f"{labels_savepath}.npy"):
        sm_embeddings, sm_y = np.load(f"{embeddings_savepath}.npy"), np.load(f"{labels_savepath}.npy")
    else:
        sm_embeddings, sm_y, _ = __synthetic_multiplexing(embeddings, labels, config_data)
        
        # Save sm embeddings to file        
        __save_sm_embeddings(folder_path, embeddings_savepath, labels_savepath, sm_embeddings, sm_y)

    
    # Calc umap
    X = __calc_umap(n_components, config_data, sm_embeddings)
    
    # Save umap to file
    __save_umap_to_file(folder_path, savepath_postfix, n_components, X)

def __load_embeddings(config_data, config_model):
    logging.info("Loading embeddings...")
    embeddings, labels = load_embeddings(embeddings_type='testset' if config_data.SPLIT_DATA else 'all', config_model=config_model, config_data=config_data)
    
    return embeddings, labels

def __synthetic_multiplexing(embeddings, labels, config_data):
    logging.info("Applying Synthetic Multiplexing...")
    df = synthetic_multiplexing.__embeddings_to_df(embeddings, labels.reshape(-1,), dataset_conf=config_data)
    sm_embeddings, y, unique_groups = synthetic_multiplexing.__get_multiplexed_embeddings(df, random_state=config_data.SEED)
    logging.info(f"embeddings shape: {embeddings.shape}, y shape: {y.shape}")
    
    return sm_embeddings, y, unique_groups

def __save_sm_embeddings(folder_path, embeddings_savepath, labels_savepath, embeddings, y):    
    logging.info(f"Saving Synthetic Multipelxing embeddings to files: {embeddings_savepath}, {labels_savepath}")
    np.save(embeddings_savepath, embeddings)
    np.save(labels_savepath, y)
    
def __calc_umap(n_components, config_data, embeddings):
    logging.info(f"Calculating UMAPs (n_components = {n_components})")
    reducer = UMAP(n_components=n_components, random_state=config_data.SEED)
    X = reducer.fit_transform(embeddings.reshape(embeddings.shape[0], -1))
    
    return X

def __save_umap_to_file(folder_path, savepath_postfix, n_components, X):
    X_savepath = os.path.join(folder_path, f"sm_embeddings_{savepath_postfix}_UMAP{n_components}")
    logging.info(f"Saving Synthetic Multiplexing {n_components} UMAPs embeddings to file: {X_savepath}")
    np.save(X_savepath, X)    

if __name__ == "__main__":
    print("Starting generating features...")
    try:
        generate_features()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")