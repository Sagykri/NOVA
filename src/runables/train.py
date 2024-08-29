import os
import sys
from typing import Dict


sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import logging
import torch

from src.common.lib.utils import get_class, load_config_file
from src.common.configs.model_config import ModelConfig
from src.common.lib.models.trainers.trainer_base import TrainerBase
from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
from src.common.lib.models.NOVA_model import NOVAModel
from src.common.lib.data_loader import init_dataloaders_with_config

def __extracting_params_from_sys()->Dict:
    """Exctract paras from sys

    Returns:
        Dict: The extracted values
    """
    assert len(sys.argv) == 4, f"Invalid config paths. You must specify: model config path, trainer config path, dataset config path. ({len(sys.argv)}: {sys.argv})"
    
    model_config_path:str = sys.argv[1]
    trainer_config_path:str = sys.argv[2]
    dataset_config_path:str = sys.argv[3]
    
    return {
        'model_config_path':model_config_path,
        'trainer_config_path':trainer_config_path,
        'dataset_config_path':dataset_config_path
    }

def __train(model_config_path:str, trainer_config_path:str, dataset_config_path:str)->None:
    model_config:ModelConfig = load_config_file(model_config_path, 'model_config')
    trainer_config:TrainerConfig = load_config_file(trainer_config_path, 'training_config', model_config.CONFIGS_USED_FOLDER)
    dataset_config:DatasetConfig = load_config_file(dataset_config_path, 'dataset_config', model_config.CONFIGS_USED_FOLDER)
    
    logging.info("[Training] Init")    
    logging.info(f"Is GPU available: {torch.cuda.is_available()}; Num GPUs Available: {torch.cuda.device_count()}")
    
    logging.info("Initializing the dataloaders")
    dataloader_train, dataloader_val, _ = init_dataloaders_with_config(trainer_config, dataset_config)
    
    logging.info("Creating the NOVA model")
    nova_model = NOVAModel(model_config)
    
    logging.info(f"Creating the trainer (from class {trainer_config.TRAINER_CLASS_PATH})")
    trainer_class: TrainerBase = get_class(trainer_config.TRAINER_CLASS_PATH)
    
    logging.info(f"Instantiate trainer from class {trainer_class.__name__}")
    trainer: TrainerBase = trainer_class(trainer_config, nova_model)
    
    logging.info("Training...")
    trainer.train(dataloader_train, dataloader_val)


if __name__ == "__main__":    
    print("Calling the training func...")
    try:
        args = __extracting_params_from_sys()
        __train(**args)
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
    
# Example how to run:    
# ./bash_commands/run_py.sh ./src/runables/train -g -m 40000 -b 44 -j train -a ./src/common/configs/model_config/ClassificationModelConfig ./src/common/configs/trainer_config/ClassificationTrainerConfig  ./src/datasets/configs/training_data_config/OpenCellTrainDatasetConfig
