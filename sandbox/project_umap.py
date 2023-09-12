import logging
import sys
import os
import numpy as np
from umap import UMAP

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.lib.utils import init_logging


###########################  CONFIG  ##########################


log_file_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/project_umap_log.log"

embeddings_train_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/outputs/models_outputs_batch78_nods_tl_ep23/features/sm_embeddings_batch9_16bit_no_downsample_vqvec2.npy"
embeddings_test_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/outputs/models_outputs_batch78_nods_tl_ep23/features/sm_embeddings_batch5_16bit_no_downsample_vqvec2.npy"

random_state = 1

umap_n_components = 100

save_folder_path = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/sandbox/projection"
save_path_train = os.path.join(save_folder_path, f"train_emb9_n{umap_n_components}.npy")
save_path_test = os.path.join(save_folder_path, f"test_emb5_n{umap_n_components}.npy")


###############################################################

def project_umap():
    # Load embeddings
    embeddings_train = __load_embeddings(embeddings_train_path)    
    embeddings_test = __load_embeddings(embeddings_test_path)

    # Project
    X_train, X_test = __project(embeddings_train, embeddings_test)
    
    # Save to files
    if not os.path.exists(save_folder_path):
        logging.info(f"Save folder {save_folder_path} doesn't exists.. Creating it" )
        os.makedirs(save_folder_path)
        
    __save_to_file(X_train, save_path_train)
    __save_to_file(X_test, save_path_test)
    
    
def __save_to_file(emb, save_path):
    logging.info(f"Saving umap proj to file {save_path}.npy")
    with open(save_path, 'wb') as f:
        np.save(f, emb)
    
def __project(embeddings_train, embeddings_test):
    logging.info("Projecting...")
    
    embeddings_train = embeddings_train.reshape(embeddings_train.shape[0], -1)
    embeddings_test = embeddings_test.reshape(embeddings_test.shape[0], -1)
    
    logging.info(f"embeddings_train shape: {embeddings_train.shape}, embeddings_test shape: {embeddings_test.shape}")
    
    logging.info(f"Constructing UMAP (n_components={umap_n_components}, random_state={random_state})")
    reducer = UMAP(n_components=umap_n_components, random_state=random_state, verbose=True)
    logging.info("Fitting UMAP to train embeddings...")
    reducer.fit(embeddings_train)
    
    logging.info("Transforming UMAP to train embeddings...")
    X_train = reducer.transform(embeddings_train)
    logging.info("Transforming UMAP to test embeddings...")
    X_test = reducer.transform(embeddings_test)
    
    logging.info(f"Transformed X_train shape: {X_train.shape}, Transformed X_test shape: {X_test.shape}")
    
    return X_train, X_test
    

def __load_embeddings(emb_path):
    logging.info("Loading embeddings")
    embeddings = np.load(emb_path)
    logging.info(f"Embedding shape: {embeddings.shape}")

    return embeddings

if __name__ == "__main__":    
    init_logging(log_file_path)
    logging.info("Starting projecting umap...")
    try:
        project_umap()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
    