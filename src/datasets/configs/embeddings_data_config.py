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
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
class EmbeddingsB9DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False  
        self.EXPERIMENT_TYPE = 'neurons'
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
        self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CONDITIONS = ['Untreated']
        #self.MARKERS = ['G3BP1']
        
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        #self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class EmbeddingsB6DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True

        self.CELL_LINES = ['WT']#, 'FUSHeterozygous', 'FUSRevertant']
        self.MARKERS = ["G3BP1"] #['FUS']
        self.REPS = ['rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        #self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class EmbeddingsB5DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch5_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.REPS = ['rep1', 'rep2']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        #self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class EmbeddingsB4DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch4_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        #self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class EmbeddingsB3DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch3_16bit_no_downsample"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        #self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class EmbeddingsALLDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}_16bit_no_downsample" for i in [3,4,5,6,7,8,9]]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['FUSHomozygous', 'TDP43', 'TBK1', 'WT', 'SCNA', 'FUSRevertant','OPTN', 'FUSHeterozygous']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

############################################################
# deltaNLS
############################################################        
class EmbeddingsDeltaNLSDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", 'deltaNLS', f) for f in 
                        ["batch2_16bit_no_downsample", "batch3_16bit_no_downsample", "batch4_16bit_no_downsample", "batch5_16bit_no_downsample"]]
                
        self.SPLIT_DATA = False        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
############################################################
# U2OS
############################################################        
class EmbeddingsU2OSDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal","U2OS_spd_format")]
        self.SPLIT_DATA = False        
        self.CELL_LINES = ['U2OS']
        self.EXPERIMENT_TYPE = 'U2OS'
        self.CONDITIONS = ['Untreated','stress']
        self.EMBEDDINGS_LAYER = 'vqindhist1'
        #self.markers = ['G3BP1','DAPI','Phalloidin','DCP1A']
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        #self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

############################################################
# NiemannPick
############################################################        
class EmbeddingsNiemannPickDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", 'NiemannPick', f) for f in 
                        [#"batch1_16bit_no_downsample", 
                         #"batch2_16bit_no_downsample", 
                         "batch3_16bit_no_downsample", 
                         #"batch4_16bit_no_downsample"
                         ]]
                
        self.SPLIT_DATA = False        
        self.EXPERIMENT_TYPE = 'NiemannPick'
        # Your can set self.REPS to a specific rep or leave it None to load the two reps 
        self.REPS = ['rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']
        # You can set this var to True if you want the UMAP to color the reps with different colors
        self.ADD_REP_TO_LABEL = False
        # You can set this var to True if you want the UMAP to color the batches with different colors
        self.ADD_BATCH_TO_LABEL = False
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2'
        
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        #self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################
                
############################################################
# Perturbations 
############################################################
class EmbeddingsPertConfocalDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                                ["Perturbations_spd_format"]]
        
        self.SPLIT_DATA = False        
        self.CELL_LINES = ['WT', 'TDP43']
        #self.CONDITIONS = ['Untreated', 'DMSO1uM', 'Edavarone', 'Pridopine', 'DMSO100uM', 'Riluzole', 'Tubastatin', 'Chloroquine']
        self.CONDITIONS = ['Untreated', 'Pridopine']
        self.MARKERS = ['NCL', 'SQSTM1', 'Calreticulin', 'DAPI', 'PURA', 'NONO']
        # self.REPS = ['rep1', 'rep2']
        
        self.EXPERIMENT_TYPE = 'perturbations_confocal'
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' # 'vqindhist1', 'vqvec2'
        
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        #self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        #self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################
        
class EmbeddingsPertSPDDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["Perturbations_16bit_no_downsample"]]
                

        self.SPLIT_DATA = False        
        self.CELL_LINES = ['WT', 'TDP43']
        self.CONDITIONS = ['Untreated', 'DMSO1uM', 'Edavarone', 'Pridopine', 'DMSO100uM', 'Riluzole', 'Tubastatin', 'Chloroquine']
        self.MARKERS = ['NCL', 'SQSTM1', 'Calreticulin', 'DAPI', 'PURA', 'NONO']
        # self.REPS = ['rep1', 'rep2']
        
        self.EXPERIMENT_TYPE = 'perturbations'
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' # 'vqindhist1', 'vqvec2'
        
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        #self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        #self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

############################################################
# U2OS data
############################################################        
class EmbeddingsU2OSDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                        ["U2OS_spd_format"]]
        
        self.SPLIT_DATA = False        
        self.CELL_LINES = ['U2OS']
        self.EXPERIMENT_TYPE = 'U2OS'
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' # 'vqindhist1'
        
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

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
        
class EmbeddingsOpenCellSubsetDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["OpenCell"]]
        
        self.SPLIT_DATA = True        
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['G3BP1']
        self.EXPERIMENT_TYPE = 'opencell'

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
        
############################################################
# NiemannPick     
############################################################
class EmbeddingsNP14DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "NiemannPick", f) for f in 
                        [f"batch{i}_16bit_no_downsample" for i in [1,4]]]#,4,5,6,9]]]
        
        self.SPLIT_DATA = True
        self.EXPERIMENT_TYPE = 'NiemannPick'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['KO','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

class EmbeddingsNPB1DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "NiemannPick", f) for f in 
                        ["batch1_16bit_no_downsample"]]        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'NiemannPick'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['KO','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

class EmbeddingsNPB2DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "NiemannPick", f) for f in 
                        ["batch2_16bit_no_downsample"]]        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'NiemannPick'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['KO','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

class EmbeddingsNPB3DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "NiemannPick", f) for f in 
                        ["batch3_16bit_no_downsample"]]        
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'NiemannPick'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['KO','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

class EmbeddingsNPB4DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "NiemannPick", f) for f in 
                        ["batch4_16bit_no_downsample"]]        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'NiemannPick'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['KO','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

############################################################
# DeltaNLS     
############################################################
class EmbeddingsdNLSB2DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch2_16bit_no_downsample" ]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['TDP43','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

class EmbeddingsdNLSB3DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch3_16bit_no_downsample" ]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['TDP43','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']
    
    
        # Set a function to map the labels, can be None if not needed.
        # Instructions:
        # - The function must be given as string!
        # - Please start with 'lambda self:' and then put your lambda function
        # - If you need to use a package, use it through import as follows __import__('numpy').array([])
        # - Example: "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        #Important: this version of a function gives labels like "cellline_condition": "lambda self: lambda labels: __import__('numpy').asarray([' '.join(l.split('_')[-3-int(self.ADD_REP_TO_LABEL):-1-int(self.ADD_REP_TO_LABEL)]) for l in labels])"
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([' '.join(l.split('_')[-3-int(self.ADD_REP_TO_LABEL):-1-int(self.ADD_REP_TO_LABEL)]) for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        self.COLORMAP = {"WT Untreated": "#2FA0C1", 'TDP43 dox': "#90278E", "TDP43 Untreated":"#494CB3"}

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

class EmbeddingsdNLSB4DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch4_16bit_no_downsample" ]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['TDP43','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

class EmbeddingsdNLSB5DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch5_16bit_no_downsample" ]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.EMBEDDINGS_LAYER = 'vqvec2'
        self.CELL_LINES = ['TDP43','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']

class EmbeddingsdNLSB25DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch2_16bit_no_downsample","batch5_16bit_no_downsample" ]]
        
        self.SPLIT_DATA = True
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        self.EMBEDDINGS_LAYER = 'vqindhist1'
        self.CELL_LINES = ['TDP43','WT']
        # self.MARKERS = ['TOMM20','mitotracker','GM130'] #['FUS']
        self.REPS = ['rep1', 'rep2'] # Can be : ['rep1', 'rep2'] or ['rep1'] or ['rep2']
