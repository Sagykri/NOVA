import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging

from src.common.lib.models.NOVA_model import NOVAModel
from src.common.lib.embeddings_utils import generate_embeddings, save_embeddings
from src.common.lib.utils import load_config_file, handle_log

def generate_embeddings_with_model():
    if len(sys.argv) < 3:
        raise ValueError("Invalid arguments. Must supply trained model path (.pth) and data config.")
    
    model = NOVAModel.load_from_checkpoint(pth=sys.argv[1])
    model_output_folder = model.config_trainer.OUTPUTS_FOLDER
    handle_log(model_output_folder)

    config_path_data = sys.argv[2]
    config_data = load_config_file(config_path_data)

    embeddings, labels = generate_embeddings(model, config_path_data)
    save_embeddings(model, embeddings, labels, config_data)

if __name__ == "__main__":
    print("Starting generate embeddings...")
    try:
        generate_embeddings_with_model()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
