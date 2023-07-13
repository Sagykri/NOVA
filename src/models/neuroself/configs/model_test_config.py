import datetime
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.models.neuroself.configs.config import NeuroselfConfig

class Neuroself2Config(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_2torch')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs', "test")
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", "test", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))
        
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "model_24.pt")


        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#100000#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfB8STConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_second_try')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "checkpoint_ep6.chkp")

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfB8PiecewiseTestingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_piecewise')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "model_19.pt") #"checkpoints", "checkpoint_ep9.chkp")
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 16#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfB8PiecewiseAlsoStressTestingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_piecewise_also_stress')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "checkpoint_ep4.chkp")
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132



class NeuroselfB8STAlsoStressConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
       
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_second_try_also_stress')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "model_15.pt") #"checkpoints", "checkpoint_ep5.chkp")

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132



class NeuroselfB8SmallConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
       
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_small')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "checkpoint_ep31.chkp")

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfB8PiecewiseVQINDHISTTestingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_piecewise_vqindhist')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "checkpoint_ep8.chkp")
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        self.FC_INPUT_TYPE = 'vqindhist'

class NeuroselfB8PiecewiseNoESTestingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_piecewise_no_earlystop')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "checkpoint_ep4.chkp")
        
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100
        self.EARLY_STOP_PATIENCE = self.MAX_EPOCH

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfB2WTUnstressedTestingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch2_WT_untresssed4')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "model_21.pt")# "checkpoints", "checkpoint_ep11.chkp")
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 8e-5
        self.BATCH_SIZE = 2#20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        self.DATA_VAR = 0.00675623276886494


class NeuroselfB2SmallConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
       
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch2_small2')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "model_16.pt")# "checkpoints", "checkpoint_ep6.chkp")

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16
        self.MAX_EPOCH = 100

        self.DATA_VAR = 0.00675623276886494



class CytoTestingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_cyto')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs', "test")
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", "test", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))
        
        self.MODEL_PATH = self.PRETRAINED_MODEL_PATH#os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "model_weights.0059.h5")



        
        self.BATCH_SIZE = 64#100000#16 # 4= 4*~8 tiles per site -> 32 tiles~



class NeuroselfB7TestingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch7_scale_intensities')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = os.path.join(self.MODEL_OUTPUT_FOLDER, "checkpoints", "checkpoint_ep3.chkp")
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 16#8
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 41471 images were sampled. Variance: 0.011492688980808567
        self.DATA_VAR = 0.011492688980808567
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig
