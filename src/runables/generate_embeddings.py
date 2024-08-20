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

from sandbox.eval_new_arch.dino4cells.main_vit_fine_tuning import infer_pass
from sandbox.eval_new_arch.dino4cells.utils import utils
from sandbox.eval_new_arch.dino4cells.archs import vision_transformer as vits
import torch.backends.cudnn as cudnn

from src.common.lib.utils import load_config_file, init_logging
import datetime

class DictToObject:
    def __init__(self, dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to objects
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

def save_embeddings_with_dataloader(dataset, config, data_config, model, output_folder_path, set_type='testset'):
    data_loader = get_dataloader(dataset, config.batch_size_per_gpu, num_workers=config.num_workers, drop_last=False)
    logging.info(f"Data loaded: there are {len(dataset)} images.")

    embeddings, _, _, labels = infer_pass(model, data_loader)
    logging.info(f'total embeddings: {embeddings.shape}')
    
    unique_batches = np.unique([label.split('_')[0] for label in labels])
    logging.info(f'unique_batches: {unique_batches}')
    
    __dict_temp = {value: [index for index, item in enumerate(labels) if item.split('_')[0] == value] for value in unique_batches}
    for batch, batch_indexes in __dict_temp.items():
        # create folder if needed
        batch_save_path = os.path.join(output_folder_path, 'embeddings', data_config.EXPERIMENT_TYPE, batch)
        os.makedirs(batch_save_path, exist_ok=True)
        
        logging.info(f"Saving {len(batch_indexes)} in {batch_save_path}")
        
        np.save(os.path.join(batch_save_path,f'{set_type}_labels.npy'), np.array(labels[batch_indexes]))
        np.save(os.path.join(batch_save_path,f'{set_type}.npy'), embeddings[batch_indexes])

        logging.info(f'Finished {set_type} set, saved in {batch_save_path}')
    

def generate_embeddings():
    jobid = os.getenv('LSB_JOBID')
    __now = datetime.datetime.now()
    output_folder_path = sys.argv[2]
    logs_folder = os.path.join(output_folder_path, "logs")
    os.makedirs(logs_folder, exist_ok=True)
    init_logging(os.path.join(logs_folder, __now.strftime("%d%m%y_%H%M%S_%f") + f'_{jobid}_embeddings.log'))

    config = {
        'seed': 1,
        'embedding': {
            'image_size': 100
        },
        'patch_size': 14,
        'num_channels': 2,
        'num_classes': int(sys.argv[3]),
        
        'batch_size_per_gpu': 700,#300,#3,#65,
        'num_workers': 6,  

        'vit_version':'tiny'      
    }
    
    config = DictToObject(config)
    config_path_data = sys.argv[1]

    config_data = load_config_file(config_path_data)
    

    if not os.path.exists(output_folder_path):
        logging.info(f"{output_folder_path} doesn't exists. Creating it")
        os.makedirs(output_folder_path)

    jobid = os.getenv('LSB_JOBID')
    logging.info(f"init (jobid: {jobid})")
    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}")
    logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    cudnn.benchmark = False
    random.seed(config.seed)

    create_vit = vits.vit_base
    if config.vit_version == 'base':
        create_vit = vits.vit_base
    elif config.vit_version == 'small':
        create_vit = vits.vit_small
    elif config.vit_version == 'tiny':
        create_vit = vits.vit_tiny
    
    model = create_vit(
            img_size=[config.embedding.image_size],
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            num_classes=config.num_classes
    ).cuda()

    
    chkp_path = sys.argv[4]
    model = utils.load_model_from_checkpoint(chkp_path, model)
    
    if config_data.SPLIT_DATA: # we need to load all the training markers (remove DAPI), then split, then load only DAPI and split, then concat them, This is because DAPI wasn't in the training
        config_data.MARKERS_TO_EXCLUDE = config_data.MARKERS_TO_EXCLUDE + ['DAPI']
        dataset = DatasetSPD(config_data)
        logging.info("Split data...")
        train_indexes, val_indexes, test_indexes = dataset.split()
        
        for idx, set_type in zip([train_indexes, val_indexes, test_indexes],['trainset','valset','testset']):
            dataset_subset = Dataset.get_subset(dataset, idx)
            logging.info(f'running on {set_type}')
            
            if set_type=='testset':
                config_data = load_config_file(config_path_data)
                config_data.MARKERS = ['DAPI']
                dataset_DAPI = DatasetSPD(config_data)
                _, _, test_DAPI_indexes = dataset_DAPI.split()
                dataset_DAPI_subset = Dataset.get_subset(dataset_DAPI, test_DAPI_indexes) 
                dataset_subset.unique_markers = np.concatenate((dataset_subset.unique_markers, dataset_DAPI_subset.unique_markers), axis=1)
                dataset_subset.label = np.concatenate((dataset_subset.label, dataset_DAPI_subset.label), axis=0)
                dataset_subset.X_paths = np.concatenate((dataset_subset.X_paths, dataset_DAPI_subset.X_paths), axis=0)
                dataset_subset.y = np.concatenate((dataset_subset.y, dataset_DAPI_subset.y), axis=0)
            save_embeddings_with_dataloader(dataset_subset, config, config_data, model, output_folder_path, set_type)
            
    else:
        dataset_subset = DatasetSPD(config_data)
        save_embeddings_with_dataloader(dataset_subset, config, config_data, model, output_folder_path, set_type='all')

if __name__ == "__main__":
    print("Starting generate embeddings...")
    try:
        generate_embeddings()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
