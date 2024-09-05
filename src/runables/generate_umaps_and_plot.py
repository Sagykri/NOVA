import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import logging

from src.common.lib.utils import load_config_file
from src.common.lib.embeddings_utils import load_embeddings
from src.common.lib.plotting import plot_umap0, plot_umap1, plot_umap2
from src.common.lib.utils import handle_log

from src.analysis.analyzer_umap0 import AnalyzerUMAP0
from src.analysis.analyzer_umap1 import AnalyzerUMAP1
from src.analysis.analyzer_umap2 import AnalyzerUMAP2
from src.analysis.analyzer_umap import AnalyzerUMAP

# Mapping between umap_type and corresponding Analyzer classes and plotting functions
analyzer_mapping = {
    0: (AnalyzerUMAP0, plot_umap0, f"[Generate {AnalyzerUMAP.UMAP_type(0).name} UMAP]"),
    1: (AnalyzerUMAP1, plot_umap1, f"[Generate {AnalyzerUMAP.UMAP_type(1).name} UMAP]"),
    2: (AnalyzerUMAP2, plot_umap2, f"[Generate {AnalyzerUMAP.UMAP_type(2).name} UMAP]")
}

def generate_umaps(config_path_trainer:str, config_path_data:str, umap_idx:int)->None:
    config_trainer = load_config_file(config_path_trainer, 'data')
    model_output_folder = config_trainer.OUTPUTS_FOLDER
    handle_log(model_output_folder)

    config_data = load_config_file(config_path_data, 'data')

    embeddings, labels = load_embeddings(model_output_folder, config_data)

    if umap_idx not in analyzer_mapping:
        raise ValueError(f"Invalid UMAP index: {umap_idx}. Must be one of {list(analyzer_mapping.keys())}.")
    
    AnalyzerClass, plot_function, log_message = analyzer_mapping[umap_idx]
    logging.info(log_message)

     # Create the analyzer instance and calculate the UMAP embeddings
    analyzer = AnalyzerClass(config_trainer, config_data)
    umap_embeddings, labels = analyzer.calculate(embeddings, labels)

    # Define the output folder path and plot the UMAP
    saveroot = analyzer.get_saving_folder(feature_type='UMAPs', umap_type = analyzer.UMAP_type(umap_idx).name)
    plot_function(umap_embeddings, labels, config_data, saveroot)
        

if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply trainer config and data config and UMAP idx! (0,1,2).")
        config_path_trainer = sys.argv[1]
        config_path_data = sys.argv[2]
        umap_idx = int(sys.argv[3])

        generate_umaps(config_path_trainer, config_path_data, umap_idx)

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
