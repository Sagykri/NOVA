import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.lib.embeddings_distances_utils_vit import calc_embeddings_distances_for_vit, plot_distances_plot
from src.common.lib.utils import load_config_file, get_if_exists, init_logging
import logging
import datetime
import torch
from src.common.lib.embeddings_utils import load_vit_features


def __load_vit_features(model_output_folder, config_data, training_batches=['batch7','batch8']):
    logging.info("Clearing cache")
    torch.cuda.empty_cache()
    
    embeddings, labels = load_vit_features(model_output_folder, config_data, training_batches=training_batches)
    return embeddings, labels

def run_calc_embeddings_distances():
    model_output_folder = sys.argv[1]
     
    jobid = os.getenv('LSB_JOBID')
    __now = datetime.datetime.now()
    logs_folder = os.path.join(model_output_folder, "logs")
    os.makedirs(logs_folder, exist_ok=True)
    log_file_path = os.path.join(logs_folder, __now.strftime("%d%m%y_%H%M%S_%f") + f'_{jobid}_dist.log')
    init_logging(log_file_path)

    logging.info('[Calc Embeddings Distances ViT]')

    config_path_data = sys.argv[2]
    config_data = load_config_file(config_path_data, 'data')
    
    output_folder_path = os.path.join(model_output_folder, 'figures', config_data.EXPERIMENT_TYPE, 'distances')
    if not os.path.exists(output_folder_path):
        logging.info(f"{output_folder_path} doesn't exists. Creating it")
        os.makedirs(output_folder_path, exist_ok=True)

    if len(sys.argv) >3:
        suff = sys.argv[3]
    else:
        suff=''
    
    train_batches = get_if_exists(config_data, 'TRAIN_BATCHES', None)
    
    embeddings, labels = __load_vit_features(model_output_folder, config_data, train_batches)

    logging.info(f'Saving distances plots in {output_folder_path}')
    compare_identical_reps=False
    calc_embeddings_distances_for_vit(embeddings, labels, config_data, output_folder_path, suff, compare_identical_reps=compare_identical_reps)
    plot_distances_plot(output_folder_path,
                    convert_markers_names_to_organelles=True, suff=suff,
                    compare_identical_reps=compare_identical_reps)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        raise ValueError("Invalid config path. Must supply model output folder and data config")
    try:
        run_calc_embeddings_distances()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done!")