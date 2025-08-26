import copy
import os
import sys


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging

from src.common.utils import load_config_file, save_config
from src.figures.effect_size_plotting import plot_multiplex_forestplot
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_config import PlotConfig

from src.analysis.analyzer_effects_alyssa_NEW_multiplex import AnalyzerEffectsAlyssaNEWMultiplex


def plot_effect_sizes(output_folder_path:str, config_path_data:str, config_path_plot:str)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = output_folder_path
    logging.info("[Plot effect sizes]")
    config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')
    
    analyzer_distances = AnalyzerEffectsAlyssaNEWMultiplex(config_data, output_folder_path)
    analyzer_distances.load()
    plot_output_folder_path = analyzer_distances.get_saving_folder(feature_type='effects_multiplex')

    if plot_output_folder_path:
        os.makedirs(plot_output_folder_path, exist_ok=True)
        save_config(config_data, plot_output_folder_path)
        save_config(config_plot, plot_output_folder_path)
    
    plot_multiplex_forestplot(*analyzer_distances.features, plot_output_folder_path, config_plot, combine_on='plate')
                
if __name__ == "__main__":
    print("Starting plotting distances...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply output folder path, data config and plot_config!")
        
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        config_path_plot = sys.argv[3]
        plot_effect_sizes(output_folder_path, config_path_data, config_path_plot)
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
