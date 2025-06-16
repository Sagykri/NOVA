import json
import os
import logging
import sys


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

from src.embeddings.embeddings_config import EmbeddingsConfig
from src.common.utils import load_config_file
from src.models.utils.eval_model_utils import run_evaluation, save_plots

def parse_positional_args(argv):
    """
    Parse positional arguments.

    Args:
        argv (List[str]): List of command-line arguments.

    Returns:
        dict: Parsed values.
    """
    if len(argv) < 3:
        raise ValueError("Usage: evalulate_models.py <save_dir> <config_path_embeddings> <models_paths_filepath> [k] [neg_k] [sample_fraction] [precomputed_dists_paths_filepath]  ([] optional)")

    return {
        'save_dir': sys.argv[1],
        'config_path_embeddings': sys.argv[2],
        'models_paths_filepath': sys.argv[3],
        'k': int(argv[4]) if len(argv) > 4 else 5,
        'neg_k': int(argv[5]) if len(argv) > 5 else 20,
        'sample_fraction': float(argv[6]) if len(argv) > 6 else 1.0,
        'precomputed_dists_paths_filepath': argv[7] if len(argv) > 7 else None
    }

if __name__ == "__main__":
    args = parse_positional_args(sys.argv)

    save_dir= args['save_dir']
    config_path_embeddings = args['config_path_embeddings']
    models_paths_filepath = args['models_paths_filepath']
    k = args['k']
    neg_k = args['neg_k']
    sample_fraction = args['sample_fraction']
    precomputed_dists_paths_filepath = args['precomputed_dists_paths_filepath']

    embeddings_config:EmbeddingsConfig = load_config_file(config_path_embeddings, "embeddings")
    embeddings_config.OUTPUTS_FOLDER = save_dir

    experiment_type = embeddings_config.EXPERIMENT_TYPE
    batches = "_".join([folder.split(os.sep)[-1] for folder in embeddings_config.INPUT_FOLDERS])

    with open(models_paths_filepath, 'r') as f:
        model_folders_dict = json.load(f)

    precomputed_dists_paths = None
    if precomputed_dists_paths_filepath is not None:
        with open(precomputed_dists_paths_filepath, 'r') as f:
            precomputed_dists_paths = json.load(f)

    logging.info(f"save_dir: {save_dir}; config_path_embeddings: {config_path_embeddings}, models_paths_filepath: {model_folders_dict}; precomputed_dists_paths_filepath: {precomputed_dists_paths_filepath} k: {k}; neg_k: {neg_k}; Sample fraction: {sample_fraction}")
    
    try:
        results = run_evaluation(model_folders_dict, embeddings_config=embeddings_config, precomputed_dists_paths=precomputed_dists_paths, save_dir=save_dir, k=k, neg_k=neg_k, sample_fraction=sample_fraction)
        save_plots(results, save_dir, experiment_type=embeddings_config.EXPERIMENT_TYPE, batches=batches, k=k, neg_k=neg_k)
        logging.info("Done")
    except Exception as e:
        logging.exception(f"Error during evaluation {str(e)}")
        raise


    # Example for models_paths_filepath json file:
    # {
    # "pretrained": "/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/pretrained_model",  
    # "finetuned_CL": "/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model",  
    # "finetuned_CL_nofreeze": "/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model_no_freeze",  
    # "finetuned_CE_nofreeze": "/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model_classification_with_batch_no_freeze",
    # "finetuned_CE": "/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model_classification_with_batch_freeze"
    # }

    # Example for precomputed_dists_paths_filepath json file:
    # {
    #         "pretrained": "PATH_TO_METRICS_CSV_FILE",  
    #         "finetuned_CL": "PATH_TO_METRICS_CSV_FILE",  
    #         "finetuned_CL_nofreeze": "PATH_TO_METRICS_CSV_FILE",  
    #         "finetuned_CE_nofreeze": "PATH_TO_METRICS_CSV_FILE",
    #         "finetuned_CE": "PATH_TO_METRICS_CSV_FILE"
    #  }
