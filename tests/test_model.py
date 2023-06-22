import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import tensorflow as tf
import logging
import numpy as np

from src.common.lib.data_loader import DataLoader
from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.utils import load_config_file
from src.common.lib.model import Model

def test_load_config():
    config = load_config_file(sys.argv[1], 'model')
    print(f"!!!!!!!! LOGS_FOLDER: {config.LOGS_FOLDER} !!!!!!!!")
    logging.info(f"SEED: {config.SEED}")
    
    return config

def test_init_model(config):
    logging.info("init")
    
    logging.info("Creating model")    
    model = Model(config)
    
    return model
    
def test_load_data(model):
    logging.info("Loading test data")
    model.load_data()

    return model

def test_load_with_dataloader(model, batch_index=0):
    logging.info("Loading with dataloader")
    
    logging.info(f"Loading data config: {sys.argv[2]}")
    config_data = load_config_file(sys.argv[2], 'data', model.conf.CONFIGS_USED_FOLDER)
    
    
    logging.info(f"Is GPU available: {tf.test.is_gpu_available()}")
    
    logging.info("Init dataset")
    
    dataset = DatasetSPD(config_data)
    dataset.flip, dataset.rot = False, False
    test_indexes = None
    
    if config_data.SPLIT_DATA:
        logging.info("Split data...")
        _, _, test_indexes = dataset.split()
    
    
    batch_size = model.batch_size
    
    logging.info(f"Init dataloader (batch_size: {batch_size})")
    dataloader = DataLoader(dataset, batch_size=batch_size, indexes=test_indexes, tpe='test')
    
    logging.info(f"Getitem ({batch_index})")
    X_batch, y_batch = dataloader.get_batch(batch_index)
    
    model.test_data = X_batch
    model.test_label = y_batch
    
    logging.info(f"test size: {np.asarray(model.test_data).shape}")
    
    return model
    
def test_load_model(model):
    logging.info("Loading model")
    model.load_model()
    
    return model

def test_load_analytics(model, X=None, y=None):
    logging.info("Loading Analytics")
    model.load_analytics(X, y)
    
    return model

def test_plot_umap(model):
    logging.info("Plotting UMAP")
    model.plot_umap(s=5, alpha=0.7,
                    # cmap=['tab20', 'tab20b', 'Dark2', 'Set2'],
                    cmap='Set1',
                    savepath=os.path.join(model.conf.MODEL_OUTPUT_FOLDER, "umaps", "umap_test4_FMRP.png"), to_annot=False)

if __name__ == "__main__":
    try:
        print("-----------------------Start-----------------")
        config = test_load_config()
        model  = test_init_model(config)
        # model  = test_load_data(model)
        model = test_load_with_dataloader(model, batch_index=0)
        
        model  = test_load_model(model)
        model  = test_load_analytics(model)
        test_plot_umap(model)
        print("***********************END********************")
    except Exception as e:
        logging.exception(e)
        raise e
    logging.info("Done")
    
    