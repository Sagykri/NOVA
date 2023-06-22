import datetime
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.models.neuroself.configs.config import NeuroselfConfig

class NeuroselfTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]


class NeuroselfBS256TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs256')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 32#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]

class NeuroselfBatch678BS4TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs4')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]

class NeuroselfBatch678BS256TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs256')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 32#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]



class NeuroselfBatch678BS4TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs4')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 10
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]


class NeuroselfBS128ALLTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs128_all')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 16#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]

class NeuroselfBS128ALLBatchesTrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs128_all_batches')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 9e-5
        self.BATCH_SIZE = 16#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]



class NeuroselfBS400TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs400')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 8e-5
        self.BATCH_SIZE = 50#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]

class NeuroselfBS320TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs320')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 8e-5
        self.BATCH_SIZE = 40#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]




class NeuroselfBS1024TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_bs1024')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 7e-5
        self.BATCH_SIZE = 128#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]


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
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_qsplit3')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 8#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100
        self.Q_SPLITS = [1,3]

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]

class NeuroselfQSPLIT1TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs4_qsplit1')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100
        self.Q_SPLITS = [1,1]

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]

class NeuroselfQSPLIT30TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch678_bs4_qsplit30')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 2e-4
        self.BATCH_SIZE = 4#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100
        self.Q_SPLITS = [1,30]

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]

        
class NeuroselfQSPLIT15TrainingConfig(NeuroselfConfig):
    def __init__(self):
        super().__init__()
        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
        #                 ["batch8"]]

        
        self.MODEL_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'models_outputs_batch8_qsplit15')
        self.LOGS_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.MODEL_OUTPUT_FOLDER, "configs_used", datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f"))

        # Models
        self.MODEL_PATH = None

        # self.MARKERS = ["CD41", "CLTC", "FMRP", "G3BP1", "KIF5A", "NONO", "Phalloidin", \
        #     "PSD95", "PURA", "SQSTM1", "TDP43", "TIA1", "NEMO", "DCP1A", \
        #     "TOMM20", "ANXA11", "Calreticulin", "FUS", "LAMP1", \
        #     "mitotracker", "Nucleolin", "SNCA", \
        #     "GM130", "PEX14", "PML", "DAPI"]
        
        # self.MARKERS = None

        # self.MARKERS_TO_EXCLUDE = ['DAPI']#, 'lysotracker', 'Syto12']
        # self.CELL_LINES = None
        # self.CONDITIONS = None
        # self.SPLIT_DATA = True
        # self.DATA_SET_TYPE = 'train'
        # self.MARKERS_FOR_DOWNSAMPLE = None
        # self.TRAIN_PCT = 0.7
        # self.SHUFFLE = True
        # self.ADD_CONDITION_TO_LABEL = True 
        # self.ADD_LINE_TO_LABEL = True
        # self.ADD_TYPE_TO_LABEL = False
        # self.ADD_BATCH_TO_LABEL = False
        
        # self.SPLIT_BY_SET_FOR = None
        # self.SPLIT_BY_SET_FOR_BATCH = None
        
        self.EARLY_STOP_PATIENCE = 20
        self.LEARN_RATE = 1e-4
        self.BATCH_SIZE = 8#16 # 4= 4*~8 tiles per site -> 32 tiles~
        # self.BATCH_SIZE = 128
        self.MAX_EPOCH = 100
        self.Q_SPLITS = [1,15]

        # Was calculated based 50 images per marker (26) from batch8
        self.DATA_VAR = 0.008237136228436984 #train var for WT stressed&unstressed: 0.00791205461967132

        # self.GROUPS_TERMS_CONDITION = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.GROUPS_TERMS_LINE = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]


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
        
        