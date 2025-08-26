import logging
import os
import sys
import numpy as np
import time
import pandas as pd

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

from src.embeddings.embeddings_config import EmbeddingsConfig
from src.common.utils import load_config_file
from src.embeddings.embeddings_utils import load_embeddings
from src.analysis.analyzer_distances import AnalyzerDistances

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
        'normalize': True if "normalize" in sys.argv else False,
    }

def generate_distances(
    model_outputs_folder: str,
    config_path_data: str,
    metric: str = "euclidean",
    detailed_stats: bool = False,
    multiplexed: bool = False,
    rep_effect:bool = False,
    normalize_embeddings:bool = False):

    # Load config and embeddings
    config_data:EmbeddingsConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = model_outputs_folder

    logging.info(f"Parameters: data config path:{config_path_data}, \
        Model outputs folder:{model_outputs_folder}, Multiplexed:{multiplexed}, Detailed stats:{detailed_stats}")

    logging.info(f"[Load embeddings] Loading embeddings from {model_outputs_folder}")
    if rep_effect:
        detailed_stats= True  # If rep_effect is enabled, detailed stats are also enabled
        config_data.ADD_REP_TO_LABEL = True # Force adding rep to label for distance calculation
    embeddings, labels, _ = load_embeddings(model_outputs_folder, config_data)

    logging.info(f"[Calculate distances]")
    d = AnalyzerDistances(config_data, model_outputs_folder, rep_effect, multiplexed, detailed_stats, metric, normalize_embeddings)
    d.calculate(embeddings, labels)
    logging.info(f"[Saving distances]")
    d.save()
    

if __name__ == "__main__":
    print("Starting calculating distances...")
    try:
        args = parse_args(sys.argv)

        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply config path and embeddings folder!")

        model_outputs_folder = args['model_outputs_folder']
        config_path_data = args['config_path_data']
        rep_effect = args['rep_effect'] # optional flag: True if "rep_effect" in sys.argv else False
        multiplexed = args['multiplexed'] # optional flag: True if "multiplexed" in sys.argv else False
        detailed_stats = args['detailed_stats'] # optional flag: True if "detailed" in sys.argv else False
        metric = "euclidean"  # Default metric
        normalize_embeddings = args['normalize']  # optional flag: True if "normalize" in sys.argv else False

        generate_distances(
                model_outputs_folder=model_outputs_folder,
                config_path_data=config_path_data,
                metric=metric,
                detailed_stats=detailed_stats,
                multiplexed=multiplexed,
                rep_effect=rep_effect,
                normalize_embeddings=normalize_embeddings
            )
        logging.info("Distance calculation completed.")

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
