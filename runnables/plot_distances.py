import os
import sys


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging

from src.common.utils import load_config_file
from src.analysis.analyzer_distances import AnalyzerDistances
from src.datasets.dataset_config import DatasetConfig
from src.figures.distances_plotting import plot_distances_heatmap

def parse_args(argv):
    """
    Parse arguments.

    Args:
        argv (List[str]): List of command-line arguments.

    Returns:
        dict: Parsed values.
    """
    if len(argv) < 3:
        raise ValueError("Usage: calculate_distances.py <model_outputs_folder> <config_path_data> [rep_effect] [multiplexed] [detailed_stats] [normalize] ([] optional)")

    return {
        'model_outputs_folder' : sys.argv[1],
        'config_path_data' : sys.argv[2],
        'rep_effect': True if "rep_effect" in sys.argv else False,
        'multiplexed': True if "multiplexed" in sys.argv else False,
        'detailed_stats': True if "detailed" in sys.argv else False,
    }

def plot_distances(
    model_outputs_folder: str,
    config_path_data: str,
    metric: str = "euclidean",
    detailed_stats: bool = False,
    multiplexed: bool = False,
    rep_effect:bool = False):

    # Load config and embeddings
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = model_outputs_folder

    logging.info(f"Parameters: data config path:{config_path_data}, \
        Model outputs folder:{model_outputs_folder}, Multiplexed:{multiplexed}, Detailed stats:{detailed_stats}")

    logging.info(f"[Load distances]")
    d = AnalyzerDistances(config_data, model_outputs_folder, rep_effect, multiplexed, detailed_stats, metric)
    d.load()
    
    logging.info(f"[Plot distances]")

    plot_distances_heatmap(df = d.features,
        metric_col = "p50",
        method = "average",
        cmap = "viridis",
        figsize = (10,10),
        highlight_thresh = None,
        savepath = d.get_saving_folder(feature_type='distances'),
        fmt = ".2f",
        text_color = "black",
        do_cluster = True,
        annotate_values = True)

if __name__ == "__main__":
    print("Starting plotting distances...")
    try:
        args = parse_args(sys.argv)

        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply config path and embeddings folder!")

        model_outputs_folder = args['model_outputs_folder']
        config_path_data = args['config_path_data']
        rep_effect = args['rep_effect'] # optional flag: True if "rep_effect" in sys.argv else False
        multiplexed = args['multiplexed'] # optional flag: True if "multiplexed" in sys.argv else False
        detailed_stats = args['detailed_stats'] # optional flag: True if "detailed" in sys.argv else False
        metric = "euclidean"  # Default metric, other option is "cosine"
        
        plot_distances(
                model_outputs_folder=model_outputs_folder,
                config_path_data=config_path_data,
                metric=metric,
                multiplexed=multiplexed,
                detailed_stats=detailed_stats,
                rep_effect=rep_effect)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
