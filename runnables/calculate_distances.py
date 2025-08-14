import logging
import os
import sys
from embeddings.embeddings_config import EmbeddingsConfig
import numpy as np
import time
import pandas as pd


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

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
        raise ValueError("Usage: calculate_distances.py <model_outputs_folder> <config_path_data> [rep_effect] [multiplexed] [detailed_stats] ([] optional)")

    return {
        'model_outputs_folder' : sys.argv[1],
        'config_path_data' : sys.argv[2],
        'ref_effect': True if "ref_effect" in sys.argv else False,
        'multiplexed': True if "multiplexed" in sys.argv else False,
        'detailed_stats': True if "detailed" in sys.argv else False,
    }

def generate_distances(
    model_outputs_folder: str,
    config_path_data: str,
    metric: str = "euclidean",
    detailed_stats: bool = False,
    multiplexed: bool = False,
    rep_effect:bool = False):

    # Load config and embeddings
    config_data:EmbeddingsConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = model_outputs_folder
    embeddings, labels, _ = load_embeddings(model_outputs_folder, config_data)

    logging.info(f"[Calculate distances]")
    d = AnalyzerDistances(config_data, output_folder_path, rep_effect, multiplexed, detailed_stats, metric)
    d.calculate(embeddings, labels)
    d.save()
    

if __name__ == "__main__":
    print("Starting calculating distances...")
    try:
        args = parse_args(sys.argv)

        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply config path and embeddings folder!")

        config_path_data = args['config_path_data']
        model_outputs_folder = args['embeddings_folder']
        rep_effect = args['ref_effect'] # optional flag: True if "ref_effect" in sys.argv else False
        multiplexed = args['multiplexed'] # optional flag: True if "multiplexed" in sys.argv else False
        detailed_stats = args['detailed_stats'] # optional flag: True if "detailed" in sys.argv else False
        metric = "euclidean"  # Default metric

        logging.info(f"Parameters: data config path:{config_path_data}, \
            Model outputs folder:{model_outputs_folder}, Multiplexed:{multiplexed}, Detailed stats:{detailed_stats}")

        generate_distances(
                model_outputs_folder=model_outputs_folder,
                config_path_data=config_path_data,
                metric=metric,
                detailed_stats=detailed_stats,
                multiplexed=multiplexed,
                rep_effect=rep_effect
            )
        logging.info("Distance calculation completed.")

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
