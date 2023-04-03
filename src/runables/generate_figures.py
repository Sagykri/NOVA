import os
import sys
import importlib
import numpy as np
import logging
from common.configs.figure_config import FigureConfig
from common.lib.utils import load_config_file
import figures.V5.figures as figures


def generate_figures():
    run_config: FigureConfig = load_config_file(sys.argv[1], '_figures')
    
    figures.get_figures(run_config)
    
    
if __name__ == "__main__":
    generate_figures()
    logging.info("Done")
    
    