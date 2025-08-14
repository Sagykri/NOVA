import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")
working_dir = os.getcwd()
sys.path.append(working_dir)
print(f"working_dir: {working_dir}")

import logging
from src.embeddings.embeddings_utils import generate_multiplexed_embeddings, save_embeddings
from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig


def generate_multiplexed_embeddings_with_config(outputs_folder_path:str, config_path_data:str)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_data.OUTPUTS_FOLDER = outputs_folder_path

    # generate multiplex embeddings (using already generated single-marker embeddings)
    embeddings, labels, paths = generate_multiplexed_embeddings(outputs_folder_path, config_data)
  
    # save multiplex vectors
    save_embeddings(embeddings, labels, paths, config_data, outputs_folder_path, multiplex = True)

if __name__ == "__main__":
    print("Starting generate multiplex embeddings...")
    try:
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply outputs folder path and data config.")
        outputs_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]

        generate_multiplexed_embeddings_with_config(outputs_folder_path, config_path_data)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
