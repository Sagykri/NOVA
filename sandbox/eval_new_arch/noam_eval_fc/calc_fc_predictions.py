import os
import random
import sys



sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging
import torch

from src.common.lib.data_loader import get_dataloader
from src.datasets.dataset_spd import DatasetSPD
from src.common.lib.dataset import Dataset
from src.common.lib.utils import load_config_file

# from sandbox.eval_new_arch.dino4cells.main_vit import infer_pass
from sandbox.eval_new_arch.dino4cells.main_vit_contrastive import infer_pass
from sandbox.eval_new_arch.dino4cells.utils import utils
from sandbox.eval_new_arch.dino4cells.archs import vision_transformer as vits

import torch.backends.cudnn as cudnn

class DictToObject:
    def __init__(self, dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to objects
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

def predict_using_fc():
    return_cls_token = True
        
    config = {
        'seed': 1,
        'embedding': {
            'image_size': 100
        },
        'patch_size': 14,
        'num_channels': 2,
        
        'epochs': 300,
        
        'lr':0.0008,
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        
        'weight_decay': 0.04,
        'weight_decay_end': 0.4,
        'local_crops_number': None,
        
        'batch_size_per_gpu': 200,#300,#3,#65,
        'num_workers': 6,
        
        'accumulation_steps': 1,
    
        'early_stopping_patience': 10,
        
    }
    
    config = DictToObject(config)
    config_path_data = sys.argv[1]

    config_data = load_config_file(config_path_data)
    
    logging.info('Not doing softmax!!')
    output_folder_path = sys.argv[2]

    if not os.path.exists(output_folder_path):
        logging.info(f"{output_folder_path} doesn't exists. Creating it")
        os.makedirs(output_folder_path)

    jobid = os.getenv('LSB_JOBID')
    logging.info(f"init (jobid: {jobid})")
    logging.info("[Predict label with fc]")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    
    
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    cudnn.benchmark = False
    random.seed(config.seed)

    config_data = load_config_file(config_path_data)

    dataset = DatasetSPD(config_data, None)
    _, _, test_indexes = None, None, None


    if config_data.SPLIT_DATA:
        logging.info("Split data...")
        _, _, test_indexes = dataset.split()
        dataset_test_subset = Dataset.get_subset(dataset, test_indexes)
    else:
        dataset_test_subset = dataset
    
    data_loader_test = get_dataloader(dataset_test_subset, config.batch_size_per_gpu, num_workers=config.num_workers)
    logging.info(f"Data loaded: there are {len(dataset)} images.")

    model = vits.vit_tiny(
            img_size=[config.embedding.image_size, config.embedding.image_size],
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            num_classes=128
    ).cuda()
    
    
    
    chkp_path = sys.argv[3]
    model = utils.load_model_from_checkpoint(chkp_path, model)
    
    predictions, labels, cls_tokens = infer_pass(model, data_loader_test, return_cls_token=return_cls_token)
    set_type = 'testset'
    np.save(os.path.join(output_folder_path,f'{set_type}_pred.npy'), predictions)
    np.save(os.path.join(output_folder_path,f'{set_type}_true_labels.npy'), np.array(labels))
    if cls_tokens is not None:
        np.save(os.path.join(output_folder_path,f'{set_type}_cls_tokens.npy'), cls_tokens)

    logging.info(f'Finished {set_type} set, saved in {output_folder_path}')

if __name__ == "__main__":
    print("Starting predicting using fc...")
    try:
        predict_using_fc()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
