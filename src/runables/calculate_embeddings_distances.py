import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.lib.embeddings_distances_utils import calc_embeddings_distances, unite_batches
from src.common.lib.utils import load_config_file, get_if_exists
import logging

def run_calc_embeddings_distances(config_path_model, config_path_data, embeddings_type):
    # Get configs of model (trained model) 
    config_model = load_config_file(config_path_model, 'model')
    logging.info('[Calc Embeddings Distances]')
    # Get dataset configs (as to be used in the desired UMAP)
    config_data = load_config_file(config_path_data, 'data')

    experiment_type = get_if_exists(config_data, 'EXPERIMENT_TYPE', None)
    assert experiment_type is not None, "EXPERIMENT_TYPE can't be None"    
    embeddings_layer = get_if_exists(config_data, 'EMBEDDINGS_LAYER', 'vqvec2')

    distances_main_folder = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'distances', experiment_type, embeddings_layer)
    os.makedirs(distances_main_folder, exist_ok=True)
    logging.info(f'Saving results in {distances_main_folder}')
    calc_embeddings_distances(config_model, config_data, distances_main_folder, embeddings_type)
    unite_batches(distances_main_folder, config_data.INPUT_FOLDERS, files=['between_cell_lines_conds_similarities_rep'])



if __name__ == "__main__":

    if len(sys.argv) != 4:
        raise ValueError("Invalid config path. Must supply model and data config, and embeddings_type")
    try:
        run_calc_embeddings_distances(config_path_model= sys.argv[1], config_path_data=sys.argv[2], embeddings_type=sys.argv[3])
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done!")