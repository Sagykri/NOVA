import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.base_config import BaseConfig


class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.OUTPUTS_FOLDER = os.path.join('home', 'labs', 'hornsteinlab', 'Collaboration', 'MOmaps', 'outputs') # added by Nancy
        self.MODEL_OUTPUT_FOLDER = None # for example: os.path.join(self.OUTPUTS_FOLDER, 'models', 'model_name')
        self.CONFIGS_USED_FOLDER = None # for example: os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Important - define "self.LOGS_FOLDER" in your config. ###### SAGY, if I define it here with some default value, it doesn't work.  

        # Transfer learning model
        self.PRETRAINED_MODEL_PATH = None # added by Nancy

        # Load model from checkpoint - continuous training upon crash
        self.LAST_CHECKPOINT_PATH = None # for example: os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        # Trained model for inference (fill this after training complete)
        self.MODEL_PATH = None

        # Variance of the training set (calc with src/common/lib/calc_dataset_variance.py)
        self.DATA_VAR = None

        # Training parameters
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4 # Nancy changed this to actual batch size we can use with Wexac
        self.MAX_EPOCH = 100
        self.REDUCELR_PATIENCE = 3
        self.REDUCELR_INCREMENT = 0.1

        # Architecture parameters
        self.EMB_SHAPES = ((25, 25), (4, 4))
        self.INPUT_SHAPE = (2, 100, 100)
        self.OUTPUT_SHAPE = (2, 100, 100)
        self.FC_ARGS = {'num_layers': 2}
        self.FC_OUTPUT_IDX = "all"
        self.VQ_ARGS = [{'num_embeddings': 2048, 'embedding_dim': 64},
                        {'num_embeddings': 2048, 'embedding_dim': 64, 'channel_split':9}]
        self.FC_INPUT_TYPE = 'vqvec'
