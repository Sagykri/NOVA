import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import logging

from src.common.lib.utils import load_config_file
from src.common.lib.embeddings_utils import load_embeddings
from src.common.configs.trainer_config import TrainerConfig
from src.common.configs.dataset_config import DatasetConfig
from src.analysis.analyzer_distances_ari import AnalyzerDistancesARI

def generate_distances(config_path_trainer:str, config_path_data:str )->None:
   
    config_trainer:TrainerConfig = load_config_file(config_path_trainer, 'train')
    model_output_folder = config_trainer.OUTPUTS_FOLDER

    config_data:DatasetConfig = load_config_file(config_path_data, 'data')

    embeddings, labels = load_embeddings(model_output_folder, config_data)
    
    logging.info("[Generate distances]")
    d = AnalyzerDistancesARI(config_trainer, config_data)
    d.calculate(embeddings, labels)
    d.save()
        

if __name__ == "__main__":
    print("Starting generating distances...")
    try:
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply trainer config and data config!")
        config_path_trainer = sys.argv[1]
        config_path_data = sys.argv[2]

        generate_distances(config_path_trainer, config_path_data)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
