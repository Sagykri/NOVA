import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig 
from src.common.lib.umap_plotting import MapLabelsFunction

class UMAP1Config(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.ORDERED_MARKER_NAMES = ["DAPI", 'TDP43', 'PEX14', 'NONO', 'ANXA11', 'FUS', 'Phalloidin', 
                    'PURA', 'mitotracker', 'TOMM20', 'NCL', 'Calreticulin', 'CLTC', 'KIF5A', 'SCNA', 'SQSTM1', 'PML',
                    'DCP1A', 'PSD95', 'LAMP1', 'GM130', 'NEMO', 'CD41', 'G3BP1']
    
        # Set the size of the dots
        self.SIZE = 0.3
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MARKERS
        # Colors 
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_MARKERS

class UMAP0Stress(DatasetConfig):
    def __init__(self):
        super().__init__()
               
        self.MAP_LABELS_FUNCTION = "conditions"
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_CONDITION

class UMAP0ALS(DatasetConfig):
    def __init__(self):
        super().__init__()
               
        self.MAP_LABELS_FUNCTION =  "cell_lines"
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS

class UMAP0dNLS(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  "cell_lines_conditions"
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_DOX

class UMAP2Stress(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  "multiplex_conditions"
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_CONDITION

class UMAP2ALS(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  "multiplex_cell_lines"
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS

class UMAP2dNLS(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION = "multiplex_cell_lines_conditions"
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_DOX

class UMAP0NoMapping(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS = None

class UMAP0RepsAsLabels(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  "reps"
        self.COLOR_MAPPINGS = None
