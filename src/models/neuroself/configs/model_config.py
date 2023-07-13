import datetime
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.models.neuroself.configs.config import NeuroselfConfig


class NeuroselfTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfB7TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch7_scale_intensities')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#8
        self.MAX_EPOCH = 100

        # Was calculated based on 200 images per marker (26) from batch8 # Total of 41471 images were sampled. Variance: 0.011492688980808567
        self.DATA_VAR = 0.011492688980808567
        # ./bash_commands/run_py.sh ./src/runables/training -g -m 15000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB7TrainingConfig ./src/datasets/configs/train_config/TrainB7DatasetConfig


class NeuroselfB8PiecewiseVQINDHISTTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_piecewise_vqindhist')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        self.FC_INPUT_TYPE = 'vqindhist'

class NeuroselfB8PiecewiseNoESTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_piecewise_no_earlystop')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100
        self.EARLY_STOP_PATIENCE = self.MAX_EPOCH

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfB8PiecewiseAlsoStressTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_piecewise_also_stress')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfB8STAlsoStressTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_second_try_also_stress')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class Neuroself2TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        """
        2 layers for fc
        128 emb vectors
        """
       
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_2torch')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class Neuroself3TrainingConfig(NeuroselfConfig):
    """
        2 layers for fc
        128 emb vectors
        Fixed lr reduced counter and early stop counter
    """
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_3torch')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfB6789SmallTrainingConfig(NeuroselfConfig):
    """
        2 layers for fc
        128 emb vectors
        Fixed lr reduced counter and early stop counter
    """
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch6789_small')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 2#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfB6789Small3232TrainingConfig(NeuroselfConfig):
    """
        2 layers for fc
        128 emb vectors
        Fixed lr reduced counter and early stop counter
    """
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch6789_small_3232')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 2#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # Model Params
        self.EMB_SHAPES = ((32, 32), (4, 4))

class NeuroselfB6789SmallVQINDHISTTrainingConfig(NeuroselfConfig):
    """
        2 layers for fc
        128 emb vectors
        Fixed lr reduced counter and early stop counter
    """
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch6789_small_vqindhist')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 2#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # Model Params
        self.FC_INPUT_TYPE = 'vqindhist'

class NeuroselfB8SmallTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
       
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_small')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16#20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfArchConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_testing_arch')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16#20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        
        # self.EMB_SHAPES = ((25, 25), (12, 12))
        
class NeuroselfB8SmallVQINDHISTTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_small_VQINDHIST')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16#20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # Model Params
        self.FC_INPUT_TYPE = 'vqindhist'

class NeuroselfVQINDHIST8TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_VQINDHIST')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfTRANSFORMERSTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batchWT6789_transformers')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfTRANSFORMERS8TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batchWT8_transformers')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfTRANSFORMERS8ALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8ALL_transformers')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfTRANSFORMERSALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_ALL_transformers')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfVQINDHIST3232TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_VQINDHIST_3232')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfVQINDHISTALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_ALL_VQINDHIST')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfVQINDHISTALLALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_ALL_ALL_VQINDHIST')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 20#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBS256TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs256')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 32#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBatch678BS4TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs4')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBatch678BS256TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
      
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs256')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None


        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 32#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBatch678BS4TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
       
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs4')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBS128ALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs128_all')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 16#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBS128ALLBatchesTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs128_all_batches')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 16#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBS400TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs400')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 8e-5
        self.BATCH_SIZE = 50#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBS320TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs320')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 8e-5
        self.BATCH_SIZE = 40#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBS1024TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
    
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs1024')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 7e-5
        self.BATCH_SIZE = 128#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs16_all')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16 
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfTBKTDP16TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs16_TDP43_TBK1')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 16 
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfTBKTDP50TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs50_TDP43_TBK1')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 50
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfBS32ALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs32_all')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 32 
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfBS50ALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs50_nancy_test')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 50 
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class Neuroself_B8_BS10_TBK1_TDP_TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs10_TBK1_TDP')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 10 
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class Neuroself_B8_BS10_WT_TDP_TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs10_WT_TDP')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 10 
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfQSPLIT3TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_qsplit3')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 8#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100
        self.Q_SPLITS = [1,3]

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfQSPLIT1TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
       
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs4_qsplit1')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100
        self.Q_SPLITS = [1,1]

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfQSPLIT30TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs4_qsplit30')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100
        self.Q_SPLITS = [1,30]

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfQSPLIT15TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
       
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_qsplit15')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 8#16 # 4= 4*~8 tiles per site -> 32 tiles~
        self.MAX_EPOCH = 100
        self.Q_SPLITS = [1,15]

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132


class NeuroselfBatch8BS10TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs10')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 10
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

class NeuroselfStressBatch8BS10TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs10_stress')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 10
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132
    
class NeuroselfBatch8BS4NoAugTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs4_noAUG')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132
    
class NeuroselfAllBatch8BS4TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_all_lines_bs4')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132
        
        