import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


import logging
import tensorflow as tf
from src.common.lib.utils import load_config_file
from src.common.lib.model import Model

def train():
    
    # Importing customize config for this run
    if len(sys.argv) == 2:
        config_path_train, config_path_val, config_path_test = sys.argv[1], sys.argv[1], sys.argv[1]
    elif len(sys.argv) == 3:
        config_path_train, config_path_val, config_path_test = sys.argv[1], sys.argv[2], sys.argv[2]
    elif len(sys.argv) == 4:
        config_path_train, config_path_val, config_path_test = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        raise ValueError("Invalid config paths. Must specify at least one config path.")
    
    config_train, config_val, config_test = load_config_file(config_path_train, '_train'),\
                                            load_config_file(config_path_val, '_val'),\
                                            load_config_file(config_path_test, '_test')
    logging.info("init")
    
    
    logging.info("Set split_by_set to True and data_set_type accordingly")
    config_train.DATA_SET_TYPE, config_train.SPLIT_BY_SET = "train", True
    config_val.DATA_SET_TYPE, config_val.SPLIT_BY_SET = "val", True
    config_test.DATA_SET_TYPE, config_test.SPLIT_BY_SET = "test", True
    
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
    
    