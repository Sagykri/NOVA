import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.model_config import ModelConfig

class NeuroselfConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'neuroself_models_outputs')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 32
        self.MAX_EPOCH = 100
        
        self.DATA_VAR = None
        
        self.EMB_SHAPES = ((25, 25), (4, 4))
        self.INPUT_SHAPE = (2, 100, 100)
        self.OUTPUT_SHAPE = (2, 100, 100)
        self.FC_ARGS = {'num_layers': 2}
        self.FC_OUTPUT_IDX = "all"
        self.VQ_ARGS = [{'num_embeddings': 2048, 'embedding_dim': 64},
                        {'num_embeddings': 2048, 'embedding_dim': 64, 'channel_split':9}]
        self.FC_INPUT_TYPE = 'vqvec'
        self.REDUCELR_PATIENCE = 3
        self.REDUCELR_INCREMENT = 0.1
        
