

# fig1
# c: umap1 b78 of pretrained model (test set to have same numbers?)
# supp: umap1 of b78 in cytoself or cellprofiler

# fig2:
# b: umap0 of stress (batch6?)
# c: stress distances
# supp: umap1 of b78 wt untreated 

# fig3:dnls
# umap0 of dcp1a in all cell lines (batch3?)
# distances between -dox and +dox

# fig5:als
# umap2 of all cell lines
# distances bubble plot + interseting umap0: meaning, make all umap0 of wt_untreated and other line (couples)

import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig

############################################################
# Figure1
############################################################ 
class NeuronsUMAP1B78OpencellFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]
        
        self.SETS = ['trainset','valset','testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False

class NeuronsUMAP1B78FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        # Take only test set of B7+8  
        self.SETS = ['testset']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False