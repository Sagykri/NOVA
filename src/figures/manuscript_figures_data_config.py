import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig

############################################################
# Figure1
############################################################ 
class NeuronsUMAP1B6FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False

############################################################
# Figure1 - supp
############################################################
class NeuronsUMAP1B9FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False












class NeuronsUMAP0StressB6FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SETS = ['testset'] # for pretrain model, there is only testset. For finetuned, we want to take only the test?? #TODO decide
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        # Take only test set of B7+8  
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.MARKERS = ['G3BP1','PML','ANXA11'] #TODO: remove
