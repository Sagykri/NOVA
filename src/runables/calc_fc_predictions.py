import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging
import torch
from scipy.special import softmax

from src.common.lib.utils import load_config_file
from src.common.lib.embeddings_utils import init_model_for_embeddings, load_dataset_for_embeddings, load_model_with_dataloader


def predict_using_fc():
    
    if len(sys.argv) < 3:
        raise ValueError("Invalid config path. Must supply model config and data config.")
    
    config_path_model = sys.argv[1]
    config_path_data = sys.argv[2]

    model, config_model =  init_model_for_embeddings(config_path_model=config_path_model)
    config_data = load_config_file(config_path_data, 'data', config_model.CONFIGS_USED_FOLDER)
    
    output_folder_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, "pretext", "predictions")
    if not os.path.exists(output_folder_path):
        logging.info(f"{output_folder_path} doesn't exists. Creating it")
        os.makedirs(output_folder_path)

    jobid = os.getenv('LSB_JOBID')
    logging.info(f"init (jobid: {jobid})")
    logging.info("[Predict label with fc]")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")

    __unique_labels_path = os.path.join(config_model.MODEL_OUTPUT_FOLDER, "unique_labels.npy")
    if os.path.exists(__unique_labels_path):
        logging.info(f"unique_labels.npy files has been detected - using it. ({__unique_labels_path})")
        unique_markers = np.load(__unique_labels_path)
        logging.info(f'unique markers shape:{unique_markers.shape}')
        logging.info(f'unique markers:{unique_markers}')
    else:
        logging.warn(f"Couldn't find unique_labels file: {__unique_labels_path}")
        raise Exception(f"Couldn't find unique_labels file: {__unique_labels_path}")
    
    # batch_size=1 to make sure each batch has only one site - so we can keep tile indices
    datasets_list = load_dataset_for_embeddings(config_data=config_data, batch_size=1, config_model=config_model)
    trained_model = load_model_with_dataloader(model, datasets_list)

    __predict_with_dataloader(trained_model, output_folder_path, datasets_list)

def __predict_with_dataloader(model, output_folder_path, datasets_list):

    # Parser to get the image's batch/cell_line/condition/rep/marker
    def label(full_path):
        path_list = full_path.split(os.sep)
        cell_line_condition_marker_list = path_list[-4:-1]
        cell_line_condition_marker = '_'.join(cell_line_condition_marker_list)
        return cell_line_condition_marker
    get_label = np.vectorize(label)
    
    def do_label_prediction(images_batch, all_predictions, all_paths, all_labels):
        # images_batch is torch.Tensor of size(n_tiles, n_channels, 100, 100)
        labels = get_label(images_batch['image_path'])
        path_with_tiles = [f'{path}_{n_tile}' for n_tile, path in enumerate(images_batch['image_path'])]
        logging.info(images_batch['image'].numpy().shape)
        predictions = model.model.infer_embeddings(images_batch['image'].numpy(), output_layer='fc1')
        logging.info(f'predictions shape: {predictions.shape}')
        all_predictions.append(predictions)
        all_paths.extend(path_with_tiles),
        all_labels.extend(labels)
        return all_predictions, all_paths, all_labels
    
    def predict_for_set(set_type, set_index, datasets_list):
        logging.info(f"Predict label - {set_type} set")
        all_predictions, all_paths, all_labels = [], [], []
        for i, images_batch in enumerate(datasets_list[set_index]):
            all_predictions, all_paths, all_labels = do_label_prediction(images_batch, all_predictions, all_paths, all_labels)
        all_predictions = np.concatenate(all_predictions, axis=1)
        logging.info(f'all_predictions shape: {all_predictions.shape}')
        all_predictions_softmax = softmax(all_predictions, axis=-1)
        np.save(os.path.join(output_folder_path,f'{set_type}_pred.npy'), all_predictions_softmax)
        np.save(os.path.join(output_folder_path,f'{set_type}_paths.npy'), np.array(all_paths))
        np.save(os.path.join(output_folder_path,f'{set_type}_true_labels.npy'), np.array(all_labels))
        logging.info(f'Finished {set_type} set, saved in {output_folder_path}')
        return None
    
    if len(datasets_list)==3:
        predict_for_set('testset', 2 , datasets_list)
        predict_for_set('valset', 1 , datasets_list)
        predict_for_set('trainset', 0 , datasets_list)
        
    elif len(datasets_list)==1:
        predict_for_set('all', 0 , datasets_list)
    
    else:
        logging.exception("[Generate Embeddings] Load model: List of datasets is not supported.")
    
    return None
        

if __name__ == "__main__":
    print("Starting predicting using fc...")
    try:
        predict_using_fc()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
