import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging

from src.common.lib.utils import load_config_file
from src.common.lib.plotting import plot_distances_plots
from src.common.lib.utils import handle_log

from src.analysis.analyzer_distances_ari import AnalyzerDistancesARI


def plot_distances():
    if len(sys.argv) < 3:
        raise ValueError("Invalid arguments. Must supply trainer config and data config!")
    
    config_path_trainer = sys.argv[1]
    config_trainer = load_config_file(config_path_trainer, 'data')
    model_output_folder = config_trainer.OUTPUTS_FOLDER #TODO: change this to the right name
    handle_log(model_output_folder)

    config_path_data = sys.argv[2]
    config_data = load_config_file(config_path_data, 'data')
       
    logging.info("[Generate distances (vit)]")
    d = AnalyzerDistancesARI(config_trainer, config_data)
    d.load()
    output_folder_path = os.path.join(model_output_folder, 'figures', d.experiment_type, 'distances')
    plot_distances_plots(distances=d.features, config_data=config_data, output_folder_path=output_folder_path)

        
if __name__ == "__main__":
    print("Starting plotting distances...")
    try:
        plot_distances()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
