import datetime
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.models.neuroself.configs.config import NeuroselfConfig

MODEL_OUTPUT = '/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/'

class ExampleNeuroselfModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch_test')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 4
        self.MAX_EPOCH = 100

        # Was calculated based in XX images per marker (26) from batch XX
        self.DATA_VAR = 0.01

class TLep23NeuroselfB78ModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch78_tl_ep23')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.PRETRAINED_MODEL_PATH = os.path.join(self.OUTPUTS_FOLDER, "models_outputs_cytoself_qsplit9", "checkpoints", "checkpoint_ep23.chkp") 
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints', 'checkpoint_ep18.chkp')
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 40162 images were sampled. Variance: 0.011433005990307048
        self.DATA_VAR = 0.00939656708666626 # 7_16bit: 0.00939656708666626
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig


class TLNeuroselfB78NoDSModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.OUTPUTS_FOLDER = MODEL_OUTPUT ## TODO: remove, temp fix
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch78_nods_tl_ep23')
        
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.PRETRAINED_MODEL_PATH = os.path.join(self.OUTPUTS_FOLDER,"models_outputs_cytoself_qsplit9", "checkpoints", "checkpoint_ep23.chkp") 
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints', 'checkpoint_ep21.chkp')
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch7
        self.DATA_VAR =0.016091262813612773 # 7_16bit_nods: 0.016091262813612773
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig
        
class TLNeuroselfdeltaNLSB25ModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.OUTPUTS_FOLDER = MODEL_OUTPUT ## TODO: remove, temp fix
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_deltaNLS_tl_neuroself_sep_TDP43')
        
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.PRETRAINED_MODEL_PATH = os.path.join(self.OUTPUTS_FOLDER, "models_outputs_batch78_nods_tl_ep23", "checkpoints", "checkpoint_ep21.chkp") 
        self.MODEL_PATH = None 
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4
        self.MAX_EPOCH = 100

        # batch2 var:  0.0074132928873828965
        # batch3 var:  0.007107947532662469
        # batch4 var:  0.0070460634463774836
        # batch5 var:  0.006884715302232348
        self.DATA_VAR = 0.00714900409 # the mean of B2 and B5  