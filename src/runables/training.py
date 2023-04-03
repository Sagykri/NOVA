import tensorflow as tf
import sys
from src.common.lib.utils import load_config_file
from src.common.lib.model import Model
import logging

def train():
    
    
    # Importing customize config for this run
    config_path_train, config_path_val, config_path_test = sys.argv[1], sys.argv[2], sys.argv[3]
    config_train, config_val, config_test = load_config_file(config_path_train, '_train'),\
                                            load_config_file(config_path_val, '_val'),\
                                            load_config_file(config_path_test, '_test')
    
    logging.info("init")
    
    logging.info(f"Is GPU available: {tf.test.is_gpu_available()}")
    
    logging.info("Creating model")
    
    model = Model(config_train)
    logging.info("Loading training data")
    model.load_data()
    
    logging.info("Loading validation data")
    model.set_params(config_val)           
    model.load_data()
    
    logging.info("Loading test data")
    model.set_params(config_test)           
    model.load_data()

    logging.info("[Start] Training..")

    model.train()
                
    logging.info("[End] Training..")

if __name__ == "__main__":
    train()
    logging.info("Done")
    
    