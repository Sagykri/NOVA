import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.plot_config import PlotConfig 
from src.common.lib.umap_plotting import MapLabelsFunction

class UMAP1PlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.ORDERED_MARKER_NAMES = ["DAPI", 'TDP43', 'PEX14', 'NONO', 'ANXA11', 'FUS', 'Phalloidin', 
                    'PURA', 'mitotracker', 'TOMM20', 'NCL', 'Calreticulin', 'CLTC', 'KIF5A', 'SCNA', 'SQSTM1', 'PML',
                    'DCP1A', 'PSD95', 'LAMP1', 'GM130', 'NEMO', 'CD41', 'G3BP1']
    
        # Set the size of the dots
        self.SIZE = 0.2
        self.ALPHA = 1
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MARKERS.name
        # Colors 
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_MARKERS
        # umap type
        self.UMAP_TYPE = 1

class UMAP0StressPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()
               
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_CONDITION
        # umap type
        self.UMAP_TYPE = 0
class UMAP0ALSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()
               
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS
        # umap type
        self.UMAP_TYPE = 0
class UMAP0dNLSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES_CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_DOX
        # umap type
        self.UMAP_TYPE = 0
class UMAP2StressPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MULTIPLEX_CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_CONDITION
        # umap type
        self.UMAP_TYPE = 2
class UMAP2ALSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  "multiplex_cell_lines"
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS
        # umap type
        self.UMAP_TYPE = 2
class UMAP2dNLSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION = "multiplex_cell_lines_conditions"
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_DOX
        # umap type
        self.UMAP_TYPE = 2
class UMAP0NoMappingPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS = None
        # umap type
        self.UMAP_TYPE = 0
class UMAP0RepsAsLabelsPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  "reps"
        self.COLOR_MAPPINGS = None
        # umap type
        self.UMAP_TYPE = 0

class DistancesdNLSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_DOX
