import os
import sys


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging

from src.common.utils import load_config_file
from src.figures.distances_plotting import plot_distances_plots
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_config import PlotConfig

from src.analysis.analyzer_distances_ari import AnalyzerDistancesARI


def plot_distances(output_folder_path:str, config_path_data:str, config_path_plot:str)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = output_folder_path
    logging.info("[Plot distances]")
    config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')
    analyzer_distances = AnalyzerDistancesARI(config_data, output_folder_path)
    analyzer_distances.load()
    plot_output_folder_path = analyzer_distances.get_saving_folder(feature_type='distances')
    os.makedirs(plot_output_folder_path, exist_ok=True)
    plot_distances_plots(distances=analyzer_distances.features, config_data=config_data, config_plot=config_plot,
                         saveroot=plot_output_folder_path)

        
if __name__ == "__main__":
    print("Starting plotting distances...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply output folder path, data config and plot_config!")
        
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        config_path_plot = sys.argv[3]
        plot_distances(output_folder_path, config_path_data, config_path_plot)
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
