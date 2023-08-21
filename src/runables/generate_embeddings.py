import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import logging
import  torch

from src.common.lib.embeddings_utils import init_model_for_embeddings, load_dataset_for_embeddings, load_model_with_dataloader, calc_embeddings


def generate_embeddings():
    """This function expect to get 2 sys arguments
    1. Path to the config file of a src.common.lib.model.Model object (e.g., ./src/models/neuroself/configs/model_config/NeuroselfB78)
    2. Path to the config file of src.datasets.dataset_spd.DatasetSPD (./src/datasets/configs/train_config/TrainB78DatasetConfig)
    The output (embeddings) are generate under the model path, in a dedicated folder called "embeddings"

    Returns:
        None
    """
    
    # Init model and model configuration 
    model, config_model =  init_model_for_embeddings(config_path_model=sys.argv[1])
    
    logging.info("---------------Start---------------")
    logging.info("Starting to generate VQVAE2 embeddings...")
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    # Get dataset 
    datasets_list = load_dataset_for_embeddings(config_path_data=sys.argv[2], batch_size=100)
    # Set the output folder (where to save the embeddings)
    embeddings_folder = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'embeddings', 'neurons')
    # Get trained model    
    trained_model = load_model_with_dataloader(model, datasets_list)
    
    calc_embeddings(trained_model, datasets_list, embeddings_folder, save=True)
    
    return None
    

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        raise ValueError("Invalid config path. Must supply model config and data config.")
    try:
        generate_embeddings()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done!")

# Example how to run:    
# ./bash_commands/run_py.sh ./src/runables/generate_embeddings -g -m 40000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB78BIT16ShuffleTLTrainingConfig ./src/datasets/configs/train_config/TrainB78BIT16DatasetConfig 

# ./bash_commands/run_py.sh ./src/runables/generate_embeddings -g -m 40000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB78BIT16ShuffleTrainingConfig ./src/datasets/configs/train_config/TrainB78BIT16DatasetConfig 


    
