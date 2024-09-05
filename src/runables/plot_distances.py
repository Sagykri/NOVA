import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import logging

from src.common.lib.utils import load_config_file
from src.common.lib.plotting import plot_distances_plots
from src.common.lib.utils import handle_log

from src.analysis.analyzer_distances_ari import AnalyzerDistancesARI


def plot_distances(config_path_trainer:str, config_path_data:str)->None:
    config_trainer = load_config_file(config_path_trainer, 'data')
    model_output_folder = config_trainer.OUTPUTS_FOLDER #TODO: change this to the right name
    handle_log(model_output_folder)

    config_data = load_config_file(config_path_data, 'data')
       
    logging.info("[Generate distances (vit)]")
    d = AnalyzerDistancesARI(config_trainer, config_data)
    d.load()
    output_folder_path = d.get_saving_folder(feature_type='distances')
    plot_distances_plots(distances=d.features, config_data=config_data, output_folder_path=output_folder_path)

        
if __name__ == "__main__":
    print("Starting plotting distances...")
    try:
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply trainer config and data config!")
        
        config_path_trainer = sys.argv[1]
        config_path_data = sys.argv[2]
        plot_distances(config_path_trainer, config_path_data)
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
