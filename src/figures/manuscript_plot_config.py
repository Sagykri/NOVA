import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.plot_config import PlotConfig 
from src.common.lib.umap_plotting import MapLabelsFunction

class UMAP1PlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()
        
        self.ORDERED_MARKER_NAMES = ["DAPI", 'NCL','FUS', 'TDP43','NONO', 'ANXA11', 'GM130', 'LAMP1', 
                                     'Calreticulin','PEX14', 'DCP1A', 'CD41', 'SQSTM1','PML',
                                     'SCNA','SNCA', 'NEMO',
                                     'PSD95','KIF5A',  'CLTC', 'TOMM20','mitotracker', 'PURA', 'G3BP1', 'Phalloidin']
    
        # Set the size of the dots
        self.SIZE = 1
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

        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MULTIPLEX_CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_CONDITION
        # umap type
        self.UMAP_TYPE = 2
class UMAP2ALSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  MapLabelsFunction.MULTIPLEX_CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS
        # umap type
        self.UMAP_TYPE = 2
class UMAP2dNLSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MULTIPLEX_CELL_LINES_CONDITIONS.name
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

        self.MAP_LABELS_FUNCTION =  MapLabelsFunction.REPS.name
        self.COLOR_MAPPINGS = None
        # umap type
        self.UMAP_TYPE = 0

class DistancesNeuronsStressPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_CONDITION_AND_ALS
        self.SHOW_BASELINE = True

class DistancesNeuronsStressNoBaselinePlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_CONDITION_AND_ALS
        self.SHOW_BASELINE = False

class DistancesNeuronsALSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_CONDITION_AND_ALS

class DistancesdNLSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_DOX
        self.UPPER_GRAPH_YLIM = (0.55, 1)
        self.LOWER_GRAPH_YLIM = (-0.01, 0.25)
        self.SHOW_BASELINE = True

class DistancesdNLSNoBaselinePlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_DOX
        self.UPPER_GRAPH_YLIM = (0.55, 1)
        self.LOWER_GRAPH_YLIM = (-0.01, 0.25)
        self.SHOW_BASELINE = False

class UMAP1dNLSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.ORDERED_MARKER_NAMES = ["DAPI", 'TDP43B', 'PEX14', 'NONO', 'ANXA11', 'FUS', 'Phalloidin', 
                    'PURA', 'mitotracker', 'TOMM20', 'NCL', 'Calreticulin', 'CLTC', 'KIF5A', 'SCNA', 'SQSTM1', 'PML',
                    'DCP1A', 'PSD95', 'LAMP1', 'GM130', 'NEMO', 'CD41', 'G3BP1']
    
        # Set the size of the dots
        self.SIZE = 1
        self.ALPHA = 1
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MARKERS.name
        # Colors 
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_MARKERS
        # umap type
        self.UMAP_TYPE = 1