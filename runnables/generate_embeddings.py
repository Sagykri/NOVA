import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import numpy as np
import logging

from src.models.architectures.NOVA_model import NOVAModel
from src.embeddings.embeddings_utils import generate_embeddings, save_embeddings
from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME


def generate_embeddings_with_model(outputs_folder_path:str, config_path_data:str,batch_size:int=700)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    
    chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
    model = NOVAModel.load_from_checkpoint(chkp_path)

    embeddings, labels = generate_embeddings(model, config_data, batch_size=batch_size)
    save_embeddings(embeddings, labels, config_data, outputs_folder_path)

if __name__ == "__main__":
    print("Starting generate embeddings...")
    try:
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply outputs folder path and data config.")
        outputs_folder_path = sys.argv[1]
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder.")
        if not os.path.exists(os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)):
            raise ValueError(f"Invalid outputs folder. Must contain a {CHECKPOINTS_FOLDERNAME} folder, and inside a {CHECKPOINT_BEST_FILENAME} file.")
        
        config_path_data = sys.argv[2]

        if len(sys.argv)==4:
            try:
                batch_size = int(sys.argv[3])
            except ValueError:
                raise ValueError("Invalid batch size, must be integer")
        else:
            batch_size = 700
        generate_embeddings_with_model(outputs_folder_path, config_path_data, batch_size)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
