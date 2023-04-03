import tensorflow as tf
import sys
from src.common.lib.utils import load_config_file
from src.common.lib.model import Model
import logging

def test():
    config_test = load_config_file(sys.argv[1], '_test')
    
    logging.info("init")
    
    logging.info(f"Is GPU available: {tf.test.is_gpu_available()}")
    
    logging.info("Creating model")    
    model = Model(config_test)
       
    logging.info("Loading test data")
    model.load_data()
    
    logging.info("Loading model")
    model.load_model()
    
    logging.info("Loading Analytics")
    model.load_analytics()
    
    logging.info("Plotting UMAP")
    model.plot_umap({"savepath": "umap_test.png"})

if __name__ == "__main__":
    test()
    logging.info("Done")
    
    