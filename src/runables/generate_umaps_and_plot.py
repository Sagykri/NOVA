import json
import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging
import datetime

from src.common.lib.utils import get_if_exists, load_config_file
from src.common.lib.embeddings_utils import load_embeddings
from src.common.lib.plotting import plot_umap0, plot_umap1, plot_umap2
from src.common.lib.utils import handle_log

from src.analysis.analyzer_umap0 import AnalyzerUMAP0
from src.analysis.analyzer_umap1 import AnalyzerUMAP1
from src.analysis.analyzer_umap2 import AnalyzerUMAP2

def generate_umaps():
    if len(sys.argv) < 4:
        raise ValueError("Invalid arguments. Must supply trainer config and data config and UMAP type! ('umap0','umap1','umap2').")
    
    config_path_trainer = sys.argv[1]
    config_trainer = load_config_file(config_path_trainer, 'data')
    model_output_folder = config_trainer.OUTPUTS_FOLDER
    handle_log(model_output_folder)

    config_path_data = sys.argv[2]
    config_data = load_config_file(config_path_data, 'data')
    
    train_batches = get_if_exists(config_data, 'TRAIN_BATCHES', None)
    if train_batches:
        logging.info(f'config_data.TRAIN_BATCHES: {train_batches}')

    embeddings, labels = load_embeddings(model_output_folder, config_data, train_batches)

    umap_type = sys.argv[3]
    if umap_type=='umap0':
        logging.info("[Generate UMAPs 0]")
        u = AnalyzerUMAP0(config_trainer, config_data)
        umap_embeddings, labels = u.calculate(embeddings, labels)
        output_folder_path = os.path.join(model_output_folder, 'figures', u.experiment_type,'UMAPs', umap_type)
        plot_umap0(umap_embeddings, labels, config_data, output_folder_path)

    elif umap_type=='umap1':
        logging.info("[Generate UMAP 1]")
        u = AnalyzerUMAP1(config_trainer, config_data)
        umap_embeddings, labels = u.calculate(embeddings, labels)
        output_folder_path = os.path.join(model_output_folder, 'figures', u.experiment_type,'UMAPs', umap_type)
        plot_umap1(umap_embeddings, labels, config_data, output_folder_path)

    elif umap_type=='umap2':
        logging.info("[Generate SM (umap2)]")
        u = AnalyzerUMAP2(config_trainer, config_data)
        umap_embeddings, labels = u.calculate(embeddings, labels)
        output_folder_path = os.path.join(model_output_folder, 'figures', u.experiment_type,'UMAPs', umap_type)
        plot_umap2(umap_embeddings, labels, config_data, output_folder_path)
        

if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        generate_umaps()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
