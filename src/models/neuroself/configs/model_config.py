import datetime
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.models.neuroself.configs.config import NeuroselfConfig

# TODO: (210823) Clean!

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

class NeuroselfB78ModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch78_16bit_shuffle')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints', 'checkpoint_ep19.chkp')
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#8
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 40162 images were sampled. Variance: 0.011433005990307048
        self.DATA_VAR = 0.00939656708666626 # 7_16bit: 0.00939656708666626
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig

class TLNeuroselfB78ModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch78_16bit_shuffle_tl')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.PRETRAINED_MODEL_PATH = os.path.join(self.OUTPUTS_FOLDER, "models_outputs_cytoself_qsplit9", "checkpoints", "checkpoint_ep6.chkp") 
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints', 'checkpoint_ep7.chkp')
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#8
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 40162 images were sampled. Variance: 0.011433005990307048
        self.DATA_VAR = 0.00939656708666626 # 7_16bit: 0.00939656708666626
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig

class TLep23NeuroselfEP7B78ModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch78_tl_ep23')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.PRETRAINED_MODEL_PATH = os.path.join(self.OUTPUTS_FOLDER, "models_outputs_cytoself_qsplit9", "checkpoints", "checkpoint_ep23.chkp") 
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints', 'checkpoint_ep7.chkp')
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#8
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 40162 images were sampled. Variance: 0.011433005990307048
        self.DATA_VAR = 0.00939656708666626 # 7_16bit: 0.00939656708666626
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig


class TLep23NeuroselfEP13B78ModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch78_tl_ep23')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.PRETRAINED_MODEL_PATH = os.path.join(self.OUTPUTS_FOLDER, "models_outputs_cytoself_qsplit9", "checkpoints", "checkpoint_ep23.chkp") 
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints', 'checkpoint_ep13.chkp')
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#8
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 40162 images were sampled. Variance: 0.011433005990307048
        self.DATA_VAR = 0.00939656708666626 # 7_16bit: 0.00939656708666626
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig

class TLep23NeuroselfPretextB78ModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch78_tl_ep23_pretext')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.PRETRAINED_MODEL_PATH = os.path.join(self.OUTPUTS_FOLDER, "models_outputs_cytoself_qsplit9", "checkpoints", "checkpoint_ep23.chkp") 
        self.MODEL_PATH = None#os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints', 'checkpoint_ep13.chkp')
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#8
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 40162 images were sampled. Variance: 0.011433005990307048
        self.DATA_VAR = 0.00939656708666626 # 7_16bit: 0.00939656708666626
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig

class TLep23NeuroselfPretext2NODSB78ModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch78_tl_ep23_pretext2_nods')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.PRETRAINED_MODEL_PATH = os.path.join(self.OUTPUTS_FOLDER, "models_outputs_cytoself_qsplit9", "checkpoints", "checkpoint_ep23.chkp") 
        self.MODEL_PATH = None#os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints', 'checkpoint_ep13.chkp')
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#8
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 40162 images were sampled. Variance: 0.011433005990307048
        self.DATA_VAR = 0.016091262813612773 # 7_16bit_nods: 0.016091262813612773
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig


class TLep23NeuroselfPretext2NOUntreatedDSB78ModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch78_tl_ep23_pretext2_nods_untreated')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.PRETRAINED_MODEL_PATH = os.path.join(self.OUTPUTS_FOLDER, "models_outputs_cytoself_qsplit9", "checkpoints", "checkpoint_ep23.chkp") 
        self.MODEL_PATH = None#os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints', 'checkpoint_ep13.chkp')
        self.LAST_CHECKPOINT_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, 'checkpoints')
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#8
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 40162 images were sampled. Variance: 0.011433005990307048
        self.DATA_VAR = 0.016091262813612773 # 7_16bit_nods: 0.016091262813612773
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig


class NeuroselfBatch2DLModelConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch2_dl')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')

        
        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 8 #32 # = 2*~16 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based on all training data of batch2
        self.DATA_VAR = 0.00675623276886494
