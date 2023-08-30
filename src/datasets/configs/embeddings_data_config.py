import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig

class EmbeddingsExampleDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        # All possible options:
        # ---------------------
        # self.CELL_LINES = ['WT', 'TDP43', 'TBK1', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant', 'SCNA', 'OPTN', ]
        # self.CONDITIONS = ['Untreated', 'stress']
        # self.MARKERS = ['ANXA11', 'Calreticulin', 'CD41', 'CLTC', 'DAPI', 'DCP1A', 'FMRP', 'FUS', 'G3BP1', GM130, KIF5A, LAMP1,
        #                 'mitotracker', 'NCL', 'NEMO', 'NONO', 'PEX14', 'Phalloidin', 'PML', 'PSD95', 'PURA', 'SCNA', 'SQSTM1', 'TDP43',
        #                 'TIA1', 'TOMM20']
        # self.REPS = ['rep1', 'rep2']
        
        
        # Set this var to True if you 'input_folders_names' contains batches that the model used for training (ex. batch7/batch8), otherwise set to False
        self.SPLIT_DATA = False
        
        
        self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['TOMM20','mitotracker'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

    
        # Which type to load: ['trainset', 'valset', 'testset', 'all']
        self.EMBEDDINGS_TYPE_TO_LOAD = 'testset'
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
        # Should we add rep (rep1/rep2) to the label?
        self.ADD_REP_TO_LABEL = False
        
        # Should we add batch to the label?
        self.ADD_BATCH_TO_LABEL = False
        
        # How much percentage to sample from the dataset. Set to 1 or None for taking all dataset.
        # Valid values are: 0<SAMPLE_PCT<=1 or SAMPLE_PCT=None (identical to SAMPLE_PCT=1)
        self.SAMPLE_PCT = 1
        
        # Determines the subfolder (inside the embeddings folder) in which the embeddings will be stored
        self.EXPERIMENT_TYPE = None

############################################################
# Neurons
############################################################        
class EmbeddingsB78DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7_16bit_no_downsample", "batch8_16bit_no_downsample"]]
        
        self.SPLIT_DATA = True
        self.EXPERIMENT_TYPE = 'neurons'    

class EmbeddingsB9DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False  
        self.EXPERIMENT_TYPE = 'neurons'

class EmbeddingsB6DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'

class EmbeddingsB5DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch5_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'

class EmbeddingsB4DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch4_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'

class EmbeddingsB3DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch3_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'
        
############################################################
# Open Cell (cytoself data)
############################################################        
class EmbeddingsOpenCellDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["OpenCell"]]
        
        self.SPLIT_DATA = True        
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.EXPERIMENT_TYPE = 'opencell'
        

############################################################
# Perturbations 
############################################################
class EmbeddingsPertConfocalDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                                ["Perturbations_spd_format"]]
        
        
        self.SPLIT_DATA = False        
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['FUS']#, 'FMRP']
        
        # self.SAMPLE_PCT = 0.1 
        self.EXPERIMENT_TYPE = 'perturbations'
        
class EmbeddingsPertSPDDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["Perturbations"]]
                

        self.SPLIT_DATA = False        
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['FUS']#, 'FMRP'] 
        
        # self.SAMPLE_PCT = 0.1
        self.EXPERIMENT_TYPE = 'perturbations'

############################################################
# Batch 2 (confocal)        
############################################################
class EmbeddingsB2DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch2"]]
        

        self.SPLIT_DATA = True #True        
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CELL_LINES = ['WT', 'FUS', 'TDP43']
        self.CONDITIONS = ['unstressed']
        # self.MARKERS = ['G3BP1']
        # self.MARKERS = ['NONO', 'G3BP1', 'TOMM20', 'PURA', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        
class EmbeddingsB25DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["batch2_5_spd_format"]]
        
        self.SPLIT_DATA = False #True        
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CELL_LINES = ['WT', 'FUS', 'TDP43']
        # self.CONDITIONS = ['unstressed']
        # self.MARKERS = ['G3BP1']
        # self.MARKERS = ['NONO', 'G3BP1', 'TOMM20', 'PURA', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
                
class EmbeddingsB2B25DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["spd2/SpinningDisk/batch2","batch2_5_spd_format"]]
        
        self.ADD_BATCH_TO_LABEL = True
        self.SPLIT_DATA = False        
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CELL_LINES = ['WT', 'FUS', 'TDP43']
        self.CONDITIONS = ['unstressed']
        # self.MARKERS = ['G3BP1']
        # self.MARKERS = ['NONO', 'G3BP1', 'TOMM20', 'PURA', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        