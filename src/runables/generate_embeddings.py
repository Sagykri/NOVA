import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
sys.path.insert(1,'/home/labs/hornsteinlab/Collaboration/MOmaps/') # Nancy

import logging
import src.common.lib.utils as utils
from src.common.configs.model_config import ModelConfig


def generate_embeddings():
    run_config: ModelConfig = utils.load_config_file(sys.argv[1], '_embeddings')
    
    utils.generate_embeddings(run_config)
    
    
if __name__ == "__main__":
    generate_embeddings()
    logging.info("Done")
    
    
    
    