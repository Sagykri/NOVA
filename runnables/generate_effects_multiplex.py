import os
import sys


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging
import numpy as np

from src.common.utils import load_config_file
from src.embeddings.embeddings_utils import load_embeddings
from src.effects.effects_config import EffectConfig
from src.analysis.analyzer_effects_multiplex import AnalyzerEffectsMultiplex
    
def generate_effects(output_folder_path:str, config_path_data:str )->None:
    
    config_data:EffectConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = output_folder_path
    logging.info(f"[Generate multiplex effects]")
    
    multiplexed_embeddings, multiplexed_labels, _ = load_embeddings(output_folder_path, config_data, multiplex=True)
    logging.info(f"multiplexed_embeddings.shape: {multiplexed_embeddings.shape}, multiplexed_labels.shape: {multiplexed_labels.shape}")

    d = AnalyzerEffectsMultiplex(config_data, output_folder_path)
    d.calculate(multiplexed_embeddings, multiplexed_labels,n_boot=config_data.N_BOOT)
    d.save()
        

if __name__ == "__main__":
    print("Starting generating effects...")
    try:
        for i in range(len(sys.argv)):
            print(sys.argv[i])
            
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply output folder path and data config!")
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]

        generate_effects(output_folder_path, config_path_data)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
