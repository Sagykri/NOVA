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
        
        """UMAP1 of WT untreated
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
        self.ORDERED_MARKER_NAMES = ["DAPI", 'TDP43', 'PEX14', 'NONO', 'ANXA11', 'FUS', 'Phalloidin', 
                            'PURA', 'mitotracker', 'TOMM20', 'NCL', 'Calreticulin', 'CLTC', 'KIF5A', 'SCNA', 'SQSTM1', 'PML',
                            'DCP1A', 'PSD95', 'LAMP1', 'GM130', 'NEMO', 'CD41', 'G3BP1']
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

class NeuronsUMAP1B78OpencellFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]
        
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS = ['rep2','rep1']
        self.CELL_LINES_CONDS = ['WT_Untreated']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_MARKERS
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        self.ORDERED_MARKER_NAMES = ["DAPI", 'TDP43', 'PEX14', 'NONO', 'ANXA11', 'FUS', 'Phalloidin', 
                            'PURA', 'mitotracker', 'TOMM20', 'NCL', 'Calreticulin', 'CLTC', 'KIF5A', 'SCNA', 'SQSTM1', 'PML',
                            'DCP1A', 'PSD95', 'LAMP1', 'GM130', 'NEMO', 'CD41', 'G3BP1']
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
# opera 18 days (REIMAGED)
class NeuronsUMAP1B12Opera18daysREIMAGEDFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.REPS = ['rep2','rep1']
        self.CELL_LINES_CONDS = ['WT_Untreated']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # self.ORDERED_MARKER_NAMES = ["DAPI", 'TDP43', 'PEX14', 'NONO', 'ANXA11', 'FUS', 'Phalloidin', 
        #     'PURA', 'mitotracker', 'TOMM20', 'NCL', 'Calreticulin', 'CLTC', 'KIF5A', 'SCNA', 'SQSTM1', 'PML',
        #     'DCP1A', 'PSD95', 'LAMP1', 'GM130', 'NEMO', 'CD41', 'G3BP1'] #+ ['AGO2', 'HNRNPA1', 'PSPC1', 'Tubulin', 'VDAC1']
        self.MARKERS = ["DAPI", 'TDP43', 'PEX14', 'NONO', 'ANXA11', 'FUS', 'Phalloidin', 
            'PURA', 'mitotracker', 'TOMM20', 'NCL', 'Calreticulin', 'CLTC', 'KIF5A', 'SCNA', 'SQSTM1', 'PML',
            'DCP1A', 'PSD95', 'LAMP1', 'GM130', 'NEMO', 'CD41', 'G3BP1']
        
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
        

## Only WT Untreated

class NeuronsUMAP0B6BothRepsOnlyWTUntreatedFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['Untreated']
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
        self.UMAP_MAPPINGS = None

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################  
        
class NeuronsUMAP0B9BothRepsOnlyWTUntreatedFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['Untreated']
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
        self.UMAP_MAPPINGS = None

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################  
        

## Opera 18 days (REIMAGED)

class NeuronsUMAP0Opera18daysREIMAGEDStressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons_d18'
        
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
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0Opera18daysREIMAGEDB1BothRepsStressFigureConfig(NeuronsUMAP0Opera18daysREIMAGEDStressFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True

        
class NeuronsUMAP0Opera18daysREIMAGEDB2BothRepsStressFigureConfig(NeuronsUMAP0Opera18daysREIMAGEDStressFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True

        
class NeuronsUMAP0Opera18daysREIMAGEDB1Rep1StressFigureConfig(NeuronsUMAP0Opera18daysREIMAGEDStressFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        
        self.REPS = ['rep1'] 
        self.ADD_BATCH_TO_LABEL = False

        
class NeuronsUMAP0Opera18daysREIMAGEDB2Rep1StressFigureConfig(NeuronsUMAP0Opera18daysREIMAGEDStressFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        
        self.REPS = ['rep1'] 
        self.ADD_BATCH_TO_LABEL = False

        
class NeuronsUMAP0Opera18daysREIMAGEDB1Rep2StressFigureConfig(NeuronsUMAP0Opera18daysREIMAGEDStressFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        
        self.REPS = ['rep2'] 
        self.ADD_BATCH_TO_LABEL = False

        
class NeuronsUMAP0Opera18daysREIMAGEDB2Rep2StressFigureConfig(NeuronsUMAP0Opera18daysREIMAGEDStressFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        
        self.REPS = ['rep2'] 
        self.ADD_BATCH_TO_LABEL = False

        
class NeuronsUMAP0Opera18daysREIMAGEDBothBatchesBothRepsStressFigureConfig(NeuronsUMAP0Opera18daysREIMAGEDStressFigureConfig):
    def __init__(self):
        super().__init__()
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True



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
        
class NeuronsUMAP0B6BothRepsOnlyFUSLinesOnlyFUSMarkerFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['FUS']
        
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
        
# ALS - pairs

class NeuronsUMAP0B6Rep2ALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B6Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B6Rep2ALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B6Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B6Rep2ALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B6Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B6Rep2ALS_WT_OPTN_FigureConfig(NeuronsUMAP0B6Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B6Rep2ALS_WT_TBK1_FigureConfig(NeuronsUMAP0B6Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B6Rep2ALS_WT_TDP43_FigureConfig(NeuronsUMAP0B6Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
#
        
class NeuronsUMAP0B6Rep1ALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B6Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B6Rep1ALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B6Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B6Rep1ALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B6Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B6Rep1ALS_WT_OPTN_FigureConfig(NeuronsUMAP0B6Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B6Rep1ALS_WT_TBK1_FigureConfig(NeuronsUMAP0B6Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B6Rep1ALS_WT_TDP43_FigureConfig(NeuronsUMAP0B6Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
        
#
class NeuronsUMAP0B9Rep2ALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B9Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B9Rep2ALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B9Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B9Rep2ALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B9Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B9Rep2ALS_WT_OPTN_FigureConfig(NeuronsUMAP0B9Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B9Rep2ALS_WT_TBK1_FigureConfig(NeuronsUMAP0B9Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B9Rep2ALS_WT_TDP43_FigureConfig(NeuronsUMAP0B9Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
        
class NeuronsUMAP0B9Rep1ALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B9Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B9Rep1ALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B9Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B9Rep1ALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B9Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B9Rep1ALS_WT_OPTN_FigureConfig(NeuronsUMAP0B9Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B9Rep1ALS_WT_TBK1_FigureConfig(NeuronsUMAP0B9Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B9Rep1ALS_WT_TDP43_FigureConfig(NeuronsUMAP0B9Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
 

        
class NeuronsUMAP0B7Rep1ALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7"]]
        
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

class NeuronsUMAP0B7Rep2ALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7"]]
        
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
        
class NeuronsUMAP0B7Rep2ALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B7Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B7Rep2ALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B7Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B7Rep2ALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B7Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B7Rep2ALS_WT_OPTN_FigureConfig(NeuronsUMAP0B7Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B7Rep2ALS_WT_TBK1_FigureConfig(NeuronsUMAP0B7Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B7Rep2ALS_WT_TDP43_FigureConfig(NeuronsUMAP0B7Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
#

class NeuronsUMAP0B7Rep1ALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B7Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B7Rep1ALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B7Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B7Rep1ALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B7Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B7Rep1ALS_WT_OPTN_FigureConfig(NeuronsUMAP0B7Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B7Rep1ALS_WT_TBK1_FigureConfig(NeuronsUMAP0B7Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B7Rep1ALS_WT_TDP43_FigureConfig(NeuronsUMAP0B7Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']

class NeuronsUMAP0B8Rep2ALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]
        
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
        
class NeuronsUMAP0B8Rep1ALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch8"]]
        
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
#
        
class NeuronsUMAP0B8Rep2ALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B8Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B8Rep2ALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B8Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B8Rep2ALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B8Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B8Rep2ALS_WT_OPTN_FigureConfig(NeuronsUMAP0B8Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B8Rep2ALS_WT_TBK1_FigureConfig(NeuronsUMAP0B8Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B8Rep2ALS_WT_TDP43_FigureConfig(NeuronsUMAP0B8Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
#

class NeuronsUMAP0B8Rep1ALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B8Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B8Rep1ALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B8Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B8Rep1ALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B8Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B8Rep1ALS_WT_OPTN_FigureConfig(NeuronsUMAP0B8Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B8Rep1ALS_WT_TBK1_FigureConfig(NeuronsUMAP0B8Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B8Rep1ALS_WT_TDP43_FigureConfig(NeuronsUMAP0B8Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
## 78 testset

class NeuronsUMAP0B7Rep2TestsetALSFigureConfig(NeuronsUMAP0B7Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.SPLIT_DATA = True
        
class NeuronsUMAP0B7Rep1TestsetALSFigureConfig(NeuronsUMAP0B7Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.SPLIT_DATA = True
        
class NeuronsUMAP0B8Rep1TestsetALSFigureConfig(NeuronsUMAP0B8Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.SPLIT_DATA = True
        
class NeuronsUMAP0B8Rep2TestsetALSFigureConfig(NeuronsUMAP0B8Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.SPLIT_DATA = True
        

# b8 testset

class NeuronsUMAP0B8Rep2TestsetALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B8Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B8Rep2TestsetALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B8Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B8Rep2TestsetALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B8Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B8Rep2TestsetALS_WT_OPTN_FigureConfig(NeuronsUMAP0B8Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B8Rep2TestsetALS_WT_TBK1_FigureConfig(NeuronsUMAP0B8Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B8Rep2TestsetALS_WT_TDP43_FigureConfig(NeuronsUMAP0B8Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
#

class NeuronsUMAP0B8Rep1TestsetALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B8Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B8Rep1TestsetALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B8Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B8Rep1TestsetALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B8Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B8Rep1TestsetALS_WT_OPTN_FigureConfig(NeuronsUMAP0B8Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B8Rep1TestsetALS_WT_TBK1_FigureConfig(NeuronsUMAP0B8Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B8Rep1TestsetALS_WT_TDP43_FigureConfig(NeuronsUMAP0B8Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
# b7 testset

class NeuronsUMAP0B7Rep2TestsetALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B7Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B7Rep2TestsetALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B7Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B7Rep2TestsetALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B7Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B7Rep2TestsetALS_WT_OPTN_FigureConfig(NeuronsUMAP0B7Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B7Rep2TestsetALS_WT_TBK1_FigureConfig(NeuronsUMAP0B7Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B7Rep2TestsetALS_WT_TDP43_FigureConfig(NeuronsUMAP0B7Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
#

class NeuronsUMAP0B7Rep1TestsetALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B7Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        
class NeuronsUMAP0B7Rep1TestsetALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B7Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        
class NeuronsUMAP0B7Rep1TestsetALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B7Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        
class NeuronsUMAP0B7Rep1TestsetALS_WT_OPTN_FigureConfig(NeuronsUMAP0B7Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        
class NeuronsUMAP0B7Rep1TestsetALS_WT_TBK1_FigureConfig(NeuronsUMAP0B7Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        
class NeuronsUMAP0B7Rep1TestsetALS_WT_TDP43_FigureConfig(NeuronsUMAP0B7Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        
        
#        
# TBK1 as baseline
class NeuronsUMAP0B6Rep1ALS_TBK1_OPTN_FigureConfig(NeuronsUMAP0B6Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'OPTN']

class NeuronsUMAP0B6Rep2ALS_TBK1_OPTN_FigureConfig(NeuronsUMAP0B6Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'OPTN']
        
class NeuronsUMAP0B6Rep1ALS_TBK1_FUSHet_FigureConfig(NeuronsUMAP0B6Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'FUSHeterozygous']

class NeuronsUMAP0B6Rep2ALS_TBK1_FUSHet_FigureConfig(NeuronsUMAP0B6Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'FUSHeterozygous']
        
class NeuronsUMAP0B6Rep1ALS_TBK1_TDP43_FigureConfig(NeuronsUMAP0B6Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'TDP43']

class NeuronsUMAP0B6Rep2ALS_TBK1_TDP43_FigureConfig(NeuronsUMAP0B6Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'TDP43']
        
#

class NeuronsUMAP0B9Rep1ALS_TBK1_OPTN_FigureConfig(NeuronsUMAP0B9Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'OPTN']

class NeuronsUMAP0B9Rep2ALS_TBK1_OPTN_FigureConfig(NeuronsUMAP0B9Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'OPTN']
        
class NeuronsUMAP0B9Rep1ALS_TBK1_FUSHet_FigureConfig(NeuronsUMAP0B9Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'FUSHeterozygous']

class NeuronsUMAP0B9Rep2ALS_TBK1_FUSHet_FigureConfig(NeuronsUMAP0B9Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'FUSHeterozygous']
        
class NeuronsUMAP0B9Rep1ALS_TBK1_TDP43_FigureConfig(NeuronsUMAP0B9Rep1ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'TDP43']

class NeuronsUMAP0B9Rep2ALS_TBK1_TDP43_FigureConfig(NeuronsUMAP0B9Rep2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'TDP43']
        
#

class NeuronsUMAP0B7Rep1ALS_TBK1_OPTN_FigureConfig(NeuronsUMAP0B7Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'OPTN']

class NeuronsUMAP0B7Rep2ALS_TBK1_OPTN_FigureConfig(NeuronsUMAP0B7Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'OPTN']
        
class NeuronsUMAP0B7Rep1ALS_TBK1_FUSHet_FigureConfig(NeuronsUMAP0B7Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'FUSHeterozygous']

class NeuronsUMAP0B7Rep2ALS_TBK1_FUSHet_FigureConfig(NeuronsUMAP0B7Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'FUSHeterozygous']
        
class NeuronsUMAP0B7Rep1ALS_TBK1_TDP43_FigureConfig(NeuronsUMAP0B7Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'TDP43']

class NeuronsUMAP0B7Rep2ALS_TBK1_TDP43_FigureConfig(NeuronsUMAP0B7Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'TDP43']
        
#

class NeuronsUMAP0B8Rep1ALS_TBK1_OPTN_FigureConfig(NeuronsUMAP0B8Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'OPTN']

class NeuronsUMAP0B8Rep2ALS_TBK1_OPTN_FigureConfig(NeuronsUMAP0B8Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'OPTN']
        
class NeuronsUMAP0B8Rep1ALS_TBK1_FUSHet_FigureConfig(NeuronsUMAP0B8Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'FUSHeterozygous']

class NeuronsUMAP0B8Rep2ALS_TBK1_FUSHet_FigureConfig(NeuronsUMAP0B8Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'FUSHeterozygous']
        
class NeuronsUMAP0B8Rep1ALS_TBK1_TDP43_FigureConfig(NeuronsUMAP0B8Rep1TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'TDP43']

class NeuronsUMAP0B8Rep2ALS_TBK1_TDP43_FigureConfig(NeuronsUMAP0B8Rep2TestsetALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['TBK1', 'TDP43']        

##
        
## Opera 18 days (REIMAGED)

class NeuronsUMAP0Bpera18daysREIMAGEDALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.CELL_LINES = ['WT', "FUSRevertant", "FUSHomozygous", "FUSHeterozygous", "TBK1", "OPTN","TDP43"]
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons_d18'
        
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
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0Bpera18daysREIMAGEDB1BothRepsALSFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True
        
class NeuronsUMAP0Bpera18daysREIMAGEDB2BothRepsALSFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True
        
class NeuronsUMAP0Bpera18daysREIMAGEDB1Rep1ALSFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.REPS = ['rep1'] 
        self.ADD_BATCH_TO_LABEL = False

        
class NeuronsUMAP0Bpera18daysREIMAGEDB1Rep2ALSFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.REPS = ['rep2'] 
        self.ADD_BATCH_TO_LABEL = False

        
class NeuronsUMAP0Bpera18daysREIMAGEDB2Rep1ALSFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.REPS = ['rep1'] 
        self.ADD_BATCH_TO_LABEL = False
        
class NeuronsUMAP0Bpera18daysREIMAGEDB2Rep2ALSFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.REPS = ['rep2'] 
        self.ADD_BATCH_TO_LABEL = False
        
class NeuronsUMAP0Bpera18daysREIMAGEDBothBatchesBothRepsALSFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True

# with SNCA
class NeuronsUMAP0Bpera18daysREIMAGEDALSWithSNCAFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.CELL_LINES = ['WT', "FUSRevertant", "FUSHomozygous", "FUSHeterozygous", "TBK1", "OPTN","TDP43", "SNCA"]
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons_d18'
        
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
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        ####################################### 

class NeuronsUMAP0Bpera18daysREIMAGEDB1BothRepsALSWithSNCAFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True
        
class NeuronsUMAP0Bpera18daysREIMAGEDB2BothRepsALSWithSNCAFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True
        
                
class NeuronsUMAP0Bpera18daysREIMAGEDB2BothRepsJoinedALSWithSNCAFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.UMAP_MAPPINGS = None
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
        
        
class NeuronsUMAP0Bpera18daysREIMAGEDB1BothRepsJoinedALSWithSNCAFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.UMAP_MAPPINGS = None
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
        
class NeuronsUMAP0Bpera18daysREIMAGEDB1Rep1ALSWithSNCAFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.REPS = ['rep1'] 
        self.ADD_BATCH_TO_LABEL = False

        
class NeuronsUMAP0Bpera18daysREIMAGEDB1Rep2ALSWithSNCAFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.REPS = ['rep2'] 
        self.ADD_BATCH_TO_LABEL = False

        
class NeuronsUMAP0Bpera18daysREIMAGEDB2Rep1ALSWithSNCAFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.REPS = ['rep1'] 
        self.ADD_BATCH_TO_LABEL = False
        
class NeuronsUMAP0Bpera18daysREIMAGEDB2Rep2ALSWithSNCAFigureConfig(NeuronsUMAP0Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.REPS = ['rep2'] 
        self.ADD_BATCH_TO_LABEL = False
        
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
        
class EmbeddingsdNLSB3Rep1OnlyTDP43LineOnlyDCP1ADatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.REPS = ['rep1'] 
        self.MARKERS = ['DCP1A']
        self.CELL_LINES = ['TDP43']
        
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

class EmbeddingsdNLSB3Rep2OnlyTDP43LineOnlyDCP1ADatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.REPS = ['rep2'] 
        self.MARKERS = ['DCP1A']
        self.CELL_LINES = ['TDP43']
        
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
        
class EmbeddingsdNLSB3BothRepsOnlyWTOnlyDCP1ADatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['DCP1A']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
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
        self.UMAP_MAPPINGS = None

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################  
       
class EmbeddingsdNLSB3BothRepsOnlyTDP43UntreatedDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.CELL_LINES = ['TDP43']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['Untreated']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
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
        self.UMAP_MAPPINGS = None

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################  
        
class EmbeddingsdNLSB3BothRepsOnlyTDP43DoxDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.CELL_LINES = ['TDP43']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['dox']
        
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
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
        self.UMAP_MAPPINGS = None

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################  
        
class EmbeddingsdNLSB3BothRepsJoinedDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.REPS = ['rep1', 'rep2'] 
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
        
class EmbeddingsdNLSB4Rep1OnlyTDP43LineOnlyDCP1ADatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.REPS = ['rep1'] 
        self.MARKERS = ['DCP1A']
        self.CELL_LINES = ['TDP43']
        
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

class EmbeddingsdNLSB4Rep2OnlyTDP43LineOnlyDCP1ADatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.REPS = ['rep2'] 
        self.MARKERS = ['DCP1A']
        self.CELL_LINES = ['TDP43']
        
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
        
class EmbeddingsdNLSB4BothRepsOnlyWTOnlyDCP1ADatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['DCP1A']
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
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
        self.UMAP_MAPPINGS = None

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################  

class EmbeddingsdNLSB4BothRepsOnlyTDP43UntreatedDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.CELL_LINES = ['TDP43']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['Untreated']
        
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
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
        self.UMAP_MAPPINGS = None

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################  
        
class EmbeddingsdNLSB4BothRepsOnlyTDP43DoxDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.CELL_LINES = ['TDP43']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['dox']
        
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
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
        self.UMAP_MAPPINGS = None

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
        
class EmbeddingsdNLSB5BothRepsOnlyTDP43UntreatedDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch5"]]
        
        self.CELL_LINES = ['TDP43']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['Untreated']
        
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
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
        self.UMAP_MAPPINGS = None

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################  
        
class EmbeddingsdNLSB2BothRepsOnlyTDP43UntreatedDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch2"]]
        
        self.CELL_LINES = ['TDP43']
        self.REPS = ['rep1', 'rep2'] 
        self.CONDITIONS = ['Untreated']
        
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
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
        self.UMAP_MAPPINGS = None

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
        
class dNLSUMAP2B3BothRepsWithoutWTFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.CELL_LINES = ["TDP43"]
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
        
## Opera 18 days (REIMAGED)

class NeuronsUMAP2Bpera18daysREIMAGEDALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.CELL_LINES = ['WT', "FUSRevertant", "FUSHomozygous", "FUSHeterozygous", "TBK1", "OPTN","TDP43"]
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons_d18'
        
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

class NeuronsUMAP2Bpera18daysREIMAGEDB1BothRepsALSFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True
        
class NeuronsUMAP2Bpera18daysREIMAGEDB2BothRepsALSFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True
        
class NeuronsUMAP2Bpera18daysREIMAGEDB1Rep1ALSFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.REPS = ['rep1'] 
        
class NeuronsUMAP2Bpera18daysREIMAGEDB1Rep2ALSFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.REPS = ['rep2'] 
        
class NeuronsUMAP2Bpera18daysREIMAGEDB2Rep1ALSFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.REPS = ['rep1'] 
        
class NeuronsUMAP2Bpera18daysREIMAGEDB2Rep2ALSFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.REPS = ['rep2'] 


# with SNCA
class NeuronsUMAP2Bpera18daysREIMAGEDALSWithSNCAFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.CELL_LINES = ['WT', "FUSRevertant", "FUSHomozygous", "FUSHeterozygous", "TBK1", "OPTN","TDP43", "SNCA"]
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons_d18'
        
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

class NeuronsUMAP2Bpera18daysREIMAGEDB1BothRepsALSWithSNCAFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True
        
class NeuronsUMAP2Bpera18daysREIMAGEDB2BothRepsALSWithSNCAFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.UMAP_MAPPINGS = None
        self.ADD_REP_TO_LABEL = True
        
class NeuronsUMAP2Bpera18daysREIMAGEDB1Rep1ALSWithSNCAFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.REPS = ['rep1'] 
        
class NeuronsUMAP2Bpera18daysREIMAGEDB1Rep2ALSWithSNCAFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        self.REPS = ['rep2'] 
        
class NeuronsUMAP2Bpera18daysREIMAGEDB2Rep1ALSWithSNCAFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.REPS = ['rep1'] 
        
class NeuronsUMAP2Bpera18daysREIMAGEDB2Rep2ALSWithSNCAFigureConfig(NeuronsUMAP2Bpera18daysREIMAGEDALSWithSNCAFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        self.REPS = ['rep2'] 

# ALS - pairs

class NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        
        # self.CELL_LINES = ['WT', "FUSRevertant", "FUSHomozygous", "FUSHeterozygous", "TBK1", "OPTN","TDP43"]
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons_d18'
        
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
        
class NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        
        # self.CELL_LINES = ['WT', "FUSRevertant", "FUSHomozygous", "FUSHeterozygous", "TBK1", "OPTN","TDP43"]
        self.REPS = ['rep1','rep2'] 
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.EXPERIMENT_TYPE = 'neurons_d18'
        
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

class NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_OPTN_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_TBK1_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_TDP43_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        self.REPS = ['rep2'] 
        
#
        
class NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_OPTN_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_TBK1_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_TDP43_FigureConfig(NeuronsUMAP0B1Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        self.REPS = ['rep1'] 
        
# b2

class NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_OPTN_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_TBK1_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        self.REPS = ['rep2'] 
        
class NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_TDP43_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        self.REPS = ['rep2'] 
        
#
        
class NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_FUSHet_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHeterozygous']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_FUSHom_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSHomozygous']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_FUSRev_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'FUSRevertant']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_OPTN_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'OPTN']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_TBK1_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TBK1']
        self.REPS = ['rep1'] 
        
class NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_TDP43_FigureConfig(NeuronsUMAP0B2Opera18daysREIMAGEDALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT', 'TDP43']
        self.REPS = ['rep1'] 






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
