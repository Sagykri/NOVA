import tensorflow as tf
import sys
import os
import logging

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.lib.utils import load_config_file
from src.common.lib.model import Model

def test_load_config():
    config = load_config_file(sys.argv[1], '_test')
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
    
def test_load_model(model):
    logging.info("Loading model")
    model.load_model()
    
    return model

def test_load_analytics(model):
    logging.info("Loading Analytics")
    model.load_analytics()
    
    return model

def test_plot_umap(model):
    logging.info("Plotting UMAP")
    model.plot_umap(savepath="umap_test.png", to_annot=False)

if __name__ == "__main__":
    print("-----------------------Start-----------------")
    config = test_load_config()
    model  = test_init_model(config)
    model  = test_load_data(model)
    
    model  = test_load_model(model)
    model  = test_load_analytics(model)
    test_plot_umap(model)
    print("***********************END********************")
    logging.info("Done")
    
    