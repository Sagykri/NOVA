import json
import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging
import datetime

from src.common.lib.utils import get_if_exists, load_config_file
from src.common.lib.embeddings_utils import load_embeddings
from src.common.lib.plotting import _plot_marker_ranking
from src.common.lib.utils import handle_log

from src.Analysis.analyzer_distances_ari import AnalyzerDistancesARI

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig

def generate_distances():
    if len(sys.argv) < 3:
        raise ValueError("Invalid arguments. Must supply trainer config and data config!")
    
    config_path_trainer = sys.argv[1]
    config_trainer = load_config_file(config_path_trainer, 'data')
    model_output_folder = config_trainer.OUTPUTS_FOLDER #TODO: change this to the right name
    handle_log(model_output_folder)

    config_path_data = sys.argv[2]
    config_data = load_config_file(config_path_data, 'data')
    
    train_batches = get_if_exists(config_data, 'TRAIN_BATCHES', None)
    if train_batches:
        logging.info(f'config_data.TRAIN_BATCHES: {train_batches}')

    embeddings, labels = load_embeddings(model_output_folder, config_data, train_batches)
    
    logging.info("[Generate distances (vit)]")
    d = AnalyzerDistancesARI(config_trainer, config_data)
    d.calculate(embeddings, labels)
    d.save()
        

if __name__ == "__main__":
    print("Starting generating distances...")
    try:
        generate_distances()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")