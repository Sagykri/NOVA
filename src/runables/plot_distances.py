import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import logging

from src.common.lib.utils import load_config_file
from src.common.lib.distances_plotting import plot_distances_plots
from src.common.configs.dataset_config import DatasetConfig

from src.analysis.analyzer_distances_ari import AnalyzerDistancesARI


def plot_distances(output_folder_path:str, config_path_data:str)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = output_folder_path
    logging.info("[Plot distances]")
    analyzer_distances = AnalyzerDistancesARI(config_data, output_folder_path)
    analyzer_distances.load()
    plot_output_folder_path = analyzer_distances.get_saving_folder(feature_type='distances')
    plot_distances_plots(distances=analyzer_distances.features, config_data=config_data, output_folder_path=plot_output_folder_path)

        
if __name__ == "__main__":
    print("Starting plotting distances...")
    try:
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply output folder path and data config!")
        
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        plot_distances(output_folder_path, config_path_data)
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
