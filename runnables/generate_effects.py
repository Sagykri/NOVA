import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging

from src.common.utils import load_config_file
from src.embeddings.embeddings_utils import load_embeddings
from src.effects.effects_config import EffectConfig
from src.analysis.analyzer_effects_dist_ratio import AnalyzerEffectsDistRatio

def generate_effects(output_folder_path:str, config_path_data:str )->None:
    
    config_data:EffectConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = output_folder_path
    logging.info(f"[Generate effects]")
    logging.info(f"MIN_REQUIRED: {config_data.MIN_REQUIRED}, N_BOOT: {config_data.N_BOOT}, SUBSAMPLE_FRACTION: {config_data.SUBSAMPLE_FRACTION}, BOOTSTRAP_TRIMMING_ALPHA: {config_data.BOOTSTRAP_TRIMMING_ALPHA}")
    config_data.CELL_LINES = list(set([config_data.BASELINE.split('_')[0],config_data.PERTURBATION.split('_')[0]]))
    config_data.CONDITIONS = list(set([config_data.BASELINE.split('_')[1],config_data.PERTURBATION.split('_')[1]]))
    embeddings, labels, paths = load_embeddings(output_folder_path, config_data)
    d = AnalyzerEffectsDistRatio(config_data, output_folder_path)
    d.calculate(embeddings, labels, paths, n_boot=config_data.N_BOOT)
    d.save()
        

if __name__ == "__main__":
    print("Starting generating effects...")
    try:
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply output folder path and data config!")
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]

        generate_effects(output_folder_path, config_path_data)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
