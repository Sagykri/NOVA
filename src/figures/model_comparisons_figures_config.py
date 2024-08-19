import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.common.configs.dataset_config import DatasetConfig

############################################################
# UMAP1
############################################################ 
class NeuronsUMAP1B78FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated - testset of B7-8
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]
        
        # Take only test set of B7+8
        self.SPLIT_DATA = True 
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS = ['rep2','rep1']
        self.CELL_LINES_CONDS = ['WT_Untreated']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_MARKERS
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']

        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

class NeuronsUMAP1B78OpencellFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated - testset of B7-8
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]
        
        # Take only test set of B7+8
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS = ['rep2','rep1']
        self.CELL_LINES_CONDS = ['WT_Untreated']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_MARKERS
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']

        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

############################################################
# UMAP0 - stress
############################################################ 
class NeuronsUMAP0B6Rep1StressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1'] 
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray([' '.join("
            "l.split('_')[-3:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class NeuronsUMAP0B6Rep2StressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray([' '.join("
            "l.split('_')[-3:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )      
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0B6BothRepsStressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray([' '.join("
            "l.split('_')[-3:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = None #self.UMAP_MAPPINGS_CONDITION

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0B9Rep1StressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1'] 
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray([' '.join("
            "l.split('_')[-3:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class NeuronsUMAP0B9Rep2StressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray([' '.join("
            "l.split('_')[-3:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )      
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0B9BothRepsStressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray([' '.join("
            "l.split('_')[-3:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = None #self.UMAP_MAPPINGS_CONDITION

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0B69BothRepsStressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6","batch9"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = True
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray([' '.join("
            "l.split('_')[-3:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = None #self.UMAP_MAPPINGS_CONDITION

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

############################################################
# UMAP0 - ALS lines
############################################################ 
class NeuronsUMAP0B6Rep1ALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        # self.CELL_LINES = ['WT']
        self.REPS = ['rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class NeuronsUMAP0B6Rep2ALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        # self.CELL_LINES = ['WT']
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0B6BothRepsALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        # self.CELL_LINES = ['WT']
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = None #self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0B9Rep1ALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        # self.CELL_LINES = ['WT']
        self.REPS = ['rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class NeuronsUMAP0B9Rep2ALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        # self.CELL_LINES = ['WT']
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0B9BothRepsALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        # self.CELL_LINES = ['WT']
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = None #self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0B69BothRepsALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6","batch9"]]
        
        # self.CELL_LINES = ['WT']
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = True
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = None #self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

############################################################
# UMAP0 - dNLS
############################################################ 
class EmbeddingsdNLSB2Rep1DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch2"]]
        
        self.REPS = ['rep1'] 
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class EmbeddingsdNLSB2Rep2DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch2"]]
        
        self.REPS = ['rep2'] 
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class EmbeddingsdNLSB3Rep1DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.REPS = ['rep1'] 
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class EmbeddingsdNLSB3Rep2DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.REPS = ['rep2'] 
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class EmbeddingsdNLSB4Rep1DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.REPS = ['rep1'] 
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class EmbeddingsdNLSB4Rep2DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.REPS = ['rep2'] 
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class EmbeddingsdNLSB5Rep1DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk",'deltaNLS', f) for f in 
                        ["batch5"]]
        
        self.REPS = ['rep1'] 
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

class EmbeddingsdNLSB5Rep2DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch5"]]
        
        self.REPS = ['rep2'] 
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray(['_'.join("
            "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################     

############################################################
# UMAP0 - U2OS
############################################################ 
class EmbeddingsU2OSRep1FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                        ["U2OS_spd_format"]]
        
        self.SPLIT_DATA = False        
        self.CELL_LINES = ['U2OS']
        self.EXPERIMENT_TYPE = 'U2OS'
        self.MARKERS = ['G3BP1', 'DCP1A', 'Phalloidin', 'DAPI']
        self.REPS = ['rep1']
        
        self.MAP_LABELS_FUNCTION = (
            "lambda self: lambda labels: "
            "__import__('numpy').asarray([' '.join("
            "l.split('_')[-3:-2+self.ADD_BATCH_TO_LABEL] + "
            "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
            ") for l in labels])"
        )      
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION
        
        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

############################################################
# UMAP2
############################################################ 
class NeuronsUMAP2B6Rep2FUSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant"]
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B6Rep2FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        # self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant"]
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B6Rep2FUSLinesNOFUSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant"]
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1',"FUS"]
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B6Rep2NOFUSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1',"FUS"]
        
        self.EXPERIMENT_TYPE = 'neurons'
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B6Rep2NOFUSLinesFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT', "TBK1", "OPTN","TDP43"]
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class dNLSUMAP2B3BothRepsFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.REPS = ['rep2','rep1'] 
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B6BothRepsFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        # self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant"]
        self.REPS = ['rep2','rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B6BothRepsNOFUSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        # self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant"]
        self.REPS = ['rep2','rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1','FUS']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B6BothRepsNOFUSNOSCNAFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant",'OPTN','TBK1','TDP43']
        self.REPS = ['rep2','rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1','FUS']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B6BothRepsNOSCNAFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant",'OPTN','TBK1','TDP43']
        self.REPS = ['rep2','rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B9BothRepsFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        # self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant"]
        self.REPS = ['rep2','rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B9BothRepsNOFUSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        # self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant"]
        self.REPS = ['rep2','rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1','FUS']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B9BothRepsNOFUSNOSCNAFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant",'OPTN','TBK1','TDP43']
        self.REPS = ['rep2','rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1','FUS']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

class NeuronsUMAP2B9BothRepsNOSCNAFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.CELL_LINES = ['WT', "FUSHomozygous", "FUSHeterozygous", "FUSRevertant",'OPTN','TBK1','TDP43']
        self.REPS = ['rep2','rep1'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.MAP_LABELS_FUNCTION = None
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

# ######### TESTING NEW STUFF ##########
# class NeuronsUMAP0B6Rep1ALS____FigureConfig(DatasetConfig):
#     def __init__(self):
#         super().__init__()
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
#                         ["batch6"]]
        
#         self.CELL_LINES = ['WT','TBK1','FUSHomozygous']
#         self.REPS = ['rep1'] 
#         self.CONDITIONS = ['Untreated']
#         self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
#         
#         self.EXPERIMENT_TYPE = 'neurons'
#         self.SPLIT_DATA = False
#         self.ADD_REP_TO_LABEL = False
#         self.ADD_BATCH_TO_LABEL = False
        
#         self.MAP_LABELS_FUNCTION = (
#             "lambda self: lambda labels: "
#             "__import__('numpy').asarray(['_'.join("
#             "l.split('_')[-4:-2+self.ADD_BATCH_TO_LABEL] + "
#             "([l.split('_')[-1]] if self.ADD_REP_TO_LABEL else [])"
#             ") for l in labels])"
#         )
#         self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
#         # self.CELL_LINE_COND_HIGH = ['WT_Untreated','TBK1_Untreated', 'FUSHomozygous_Untreated']
#         self.MARKERS = ['GM130']
#         # Set the size of the dots
#         self.SIZE = 30
#         # Set the alpha of the dots (0=max opacity, 1=no opacity)
#         self.ALPHA = 0.7
#         #######################################     