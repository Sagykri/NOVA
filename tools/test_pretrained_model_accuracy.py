import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging

from src.common.utils import load_config_file
from src.datasets.dataset_config import DatasetConfig
from src.datasets.dataset_NOVA import DatasetNOVA
from src.embeddings.embeddings_utils import generate_embeddings
from src.models.architectures.NOVA_model import NOVAModel
from src.models.utils.consts import CHECKPOINT_BEST_FILENAME, CHECKPOINTS_FOLDERNAME
from tools.pretrained_model_accuracy_utils import build_probs_dataframe, compute_localization_probs, evaluate_classification_accuracy, multiclass_roc

import torch
import pandas as pd
import numpy as np

"""
Script to evaluate the accuracy of the pretrained model using embeddings, labels, 
and protein-localcization annotations.

Run Example:
    $NOVA_HOME/runnables/run.sh $NOVA_HOME/tools/test_pretrained_model_accuracy -g -m 30000 \
    -b 10 -a /home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/pretrained_model \
    ./manuscript/dataset_config/OpenCellDatasetConfig \
    /home/projects/hornsteinlab/Collaboration/MOmaps_Noam/MOmaps/tools/OpenCell_annotated_proteins.csv \
    -q short-gpu -j acc_pretrain
    
    """


def calculate_accuracy(outputs_folder_path:str, config_path_data:str, anot_file_path:str)->None:
    """
    Evaluates classification accuracy of a pretrained model.

    Parameters:
    - outputs_folder_path (str): Path to the folder containing model output embeddings.
    - config_path_data (str): Path to the dataset configuration file.
    - anot_file_path (str): Path to the annotation CSV file containing ground truth labels.

    This function:
    - Loads dataset configuration and embeddings.
    - Computes softmax probabilities over the model outputs.
    - Builds a dataframe aligning predictions with true labels.
    - Computes per-class localization probabilities.
    - Evaluates accuracy and generates a multiclass ROC curve.
    """
     
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    logging.info("[Calculate pretrained accuracy]")    
    dataset = DatasetNOVA(config_data)
    unique_labels = dataset.unique_labels
        
    chkp_path = os.path.join(outputs_folder_path, CHECKPOINTS_FOLDERNAME, CHECKPOINT_BEST_FILENAME)
    model = NOVAModel.load_from_checkpoint(chkp_path)

    all_embeddings, all_labels, _ = generate_embeddings(model, config_data)
    test_embeddings, test_labels = all_embeddings[-1], all_labels[-1]
    logging.info(f'np.unique(test_labels).shape: {np.unique(test_labels).shape} (should be 1311)')
    outputs = torch.from_numpy(test_embeddings)
    probs = torch.nn.functional.softmax(outputs, dim=1).numpy()

    probs_df = build_probs_dataframe(probs, test_labels, unique_labels, anot_file_path)
    anot = pd.read_csv(anot_file_path)
    sum_probs = compute_localization_probs(probs_df, anot)
    
    class_names = list(np.unique(anot.localization)) + ['Other']

    save_path = os.path.join(outputs_folder_path,'figures', 'Opencell', 'accuracy')
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
