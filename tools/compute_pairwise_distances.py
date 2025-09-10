import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging

from src.common.utils import load_config_file
from src.embeddings.embeddings_utils import load_embeddings
from src.datasets.dataset_config import DatasetConfig
from src.analysis.analyzer_pair_wise_distances import AnalyzerPairwiseDistances

def generate_distances(output_folder_path:str, config_path_data:str, config_path_pairwise )->None:
    
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = output_folder_path
    config_pairwise = load_config_file(config_path_pairwise, 'data')
    embeddings, labels, paths = load_embeddings(output_folder_path, config_data)
    logging.info("[Generate distances]")
    d = AnalyzerPairwiseDistances(config_data,config_pairwise, output_folder_path)
    
    d.calculate(embeddings, labels, paths)
    d.save()
        

if __name__ == "__main__":
    print("Starting generating distances...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply output folder path, data config and pair-wise config!")
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        config_path_pairwise = sys.argv[3]

        generate_distances(output_folder_path, config_path_data, config_path_pairwise)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
