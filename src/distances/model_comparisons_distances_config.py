import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


from src.common.configs.dataset_config import DatasetConfig

class NeuronsDistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated - testset of B7-8
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons'    
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']

class Neurons69Full78TestDistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated - testset of B7-8
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons'    
        self.TRAIN_BATCHES = ['batch7','batch8']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"

        
class NeuronsTest78TBK1DistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated - testset of B7-8
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons'   
        self.CELL_LINES = ['FUSHomozygous','FUSHeterozygous','FUSRevertant', 'OPTN','TBK1','TDP43']
        self.CONDITIONS = ['Untreated']
        self.TRAIN_BATCHES = ['batch7','batch8']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        self.BASELINE_CELL_LINE_CONDITION = "TBK1_Untreated"
        
class NeuronsTest78FUSRevertantDistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated - testset of B7-8
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons'   
        self.CELL_LINES = ['FUSHomozygous','FUSHeterozygous','FUSRevertant', 'OPTN','TBK1','TDP43', 'WT']
        self.CONDITIONS = ['Untreated']
        self.TRAIN_BATCHES = ['batch7','batch8']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        self.BASELINE_CELL_LINE_CONDITION = "FUSRevertant_Untreated"

class dNLSDistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(2,6)]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class dNLSTest25DistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(2,6)]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        self.TRAIN_BATCHES = ['batch2','batch5']

class dNLS345DistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(3,6)]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "TDP43_Untreated"
        
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
class dNLS34DistanceConfig(DatasetConfig): # used for the model trained on dNLS
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(3,5)]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "TDP43_Untreated"
        self.MARKERS = list(self.UMAP_MAPPINGS_MARKERS.keys())
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class EmbeddingsDay18DistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.MARKERS_TO_EXCLUDE = None
        
        
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"

        

class NeuronsDistanceB78TestConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch7", "batch8"]]
        
        self.SPLIT_DATA = True 
        self.EXPERIMENT_TYPE = 'neurons'    
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1'] # The embeddings are already calculated so it doesn't need to be the exact same markers as in the trianing_config
        
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"

class NeuronsDistanceB78FullConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch7", "batch8"]]
        
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons'    
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
