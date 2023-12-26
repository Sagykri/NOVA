import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.lib.utils import get_if_exists, load_config_file
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
    
    config_path_model=sys.argv[1]

    # Get dataset configs (as used in trainig the model)
    config_path_data = sys.argv[2]
    
    # Init model and model configuration 
    # Note: This line must be called before loading the data config file, otherwise no logs will be written to the model folder
    model, config_model =  init_model_for_embeddings(config_path_model=config_path_model)
    
    config_data = load_config_file(config_path_data, 'data') 
    embeddings_layer = get_if_exists(config_data, 'EMBEDDINGS_LAYER', 'vqvec2')

    logging.info("---------------Start---------------")
    logging.info("[Generate embeddings]")
    logging.info(f"Starting to generate {embeddings_layer} embeddings...")
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")

    logging.info(f"Init datasets {config_data} from {config_path_data}")
    experiment_type = get_if_exists(config_data, 'EXPERIMENT_TYPE', None)
    assert experiment_type is not None, "EXPERIMENT_TYPE can't be None"

    logging.info(f"experiment_type = {experiment_type}")
    logging.info(f"embeddings_layer = {embeddings_layer}")
    
    # Get dataset 
    # Note: batch_size is set to 1 to make sure each embeddings npy file is corresponding to a single site npy file
    datasets_list = load_dataset_for_embeddings(config_data=config_data, batch_size=1, config_model=config_model)
    # Set the output folder (where to save the embeddings)
    embeddings_folder = os.path.join(config_model.MODEL_OUTPUT_FOLDER, 'embeddings', experiment_type)
    # Get trained model    
    trained_model = load_model_with_dataloader(model, datasets_list)
    
    calc_embeddings(trained_model, datasets_list, embeddings_folder, save=True, embeddings_layer=embeddings_layer)
    
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


    
