import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


import logging
from src.common.lib.figures_handler import FiguresHandler
from src.common.configs.figure_config import FigureConfig
from src.common.lib.utils import get_class, load_config_file

def generate_figures():
    run_config: FigureConfig = load_config_file(sys.argv[1], '_figures')
    
    logging.info(f"Importing figures handler class.. {run_config.HANDLER_CLASS_PATH}")
    figures_handler_class: FiguresHandler = get_class(run_config.HANDLER_CLASS_PATH)
    
    logging.info(f"Instantiate figures handler {type(figures_handler_class)}")
    figures_handler: FiguresHandler = figures_handler_class(run_config)
    
    figures_handler.get_figures(run_config)
    
    
if __name__ == "__main__":
    generate_figures()
    logging.info("Done")
    
    