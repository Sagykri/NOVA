import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging

from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig
from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.label_utils import get_markers_from_labels
from src.embeddings.embeddings_utils import load_embeddings
from tools.pretrained_model_accuracy_utils import build_probs_dataframe, compute_localization_probs, evaluate_classification_accuracy, multiclass_roc

import torch
import pandas as pd
import numpy as np

def calculate_accuracy(outputs_folder_path:str, config_path_data:str, anot_file_path:str)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    config_data.MARKERS_TO_EXCLUDE = []
    logging.info("[Calcaulte pretrained accuracy]")    
    embeddings, labels, _ = load_embeddings(outputs_folder_path, config_data)  

    dataset = DatasetNOVA(config_data)
    unique_labels = get_markers_from_labels(dataset.unique_labels)
    
    outputs = torch.from_numpy(embeddings)

    probs = torch.nn.functional.softmax(outputs, dim=1).numpy()

    probs_df = build_probs_dataframe(probs, labels, unique_labels, anot_file_path)
    anot = pd.read_csv(anot_file_path)
    sum_probs = compute_localization_probs(probs_df, anot)
    
    class_names = list(np.unique(anot.localization)) + ['Other']

    save_path = os.path.join(outputs_folder_path,'figures', config_data.EXPERIMENT_TYPE, 'accuracy')
    evaluate_classification_accuracy(sum_probs, class_names, save_path)
    multiclass_roc(sum_probs, class_names, save_path)

if __name__ == '__main__':
    print("Starting to calculate pre-trained model accuracy...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply output folder path, data config and anot_file_path!")
        outputs_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        anot_file_path = sys.argv[3]

        calculate_accuracy(outputs_folder_path, config_path_data, anot_file_path)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")


### TO RUN:
# python tools/test_pretrained_model_accuracy.py /home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/pretrained_model ./manuscript/embeddings_config/EmbeddingsOpenCellDatasetConfig /home/projects/hornsteinlab/Collaboration/MOmaps_Noam/MOmaps/tools/OpenCell_annotated_proteins.csv