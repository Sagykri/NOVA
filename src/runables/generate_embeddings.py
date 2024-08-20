import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging

import Model ## SAGY ### TODO imp and add import ######

def generate_embeddings_with_model():
     if len(sys.argv) < 3:
        raise ValueError("Invalid arguments. Must supply trained model path (.pth) and data config.")

    model = Model.load_from_pth(pth=sys.argv[1])
    generate_embeddings(model, config_path_data = sys.argv[2])

if __name__ == "__main__":
    print("Starting generate embeddings...")
    try:
        generate_embeddings_with_model()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
