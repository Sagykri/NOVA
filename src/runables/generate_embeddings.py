import os
import sys
import importlib
import numpy as np
import logging
import common.lib.utils as utils
from common.configs.model_config import ModelConfig


def generate_embeddings():
    run_config: ModelConfig = utils.load_config_file(sys.argv[1], '_embeddings')
    
    utils.generate_embeddings(run_config)
    
    
if __name__ == "__main__":
    generate_embeddings()
    logging.info("Done")
    
    
    
    