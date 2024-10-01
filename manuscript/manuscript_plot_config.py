import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from manuscript.plot_config import PlotConfig
from src.datasets.label_utils import MapLabelsFunction

class UMAP1PlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()
        
        self.ORDERED_MARKER_NAMES = ["DAPI", 'NCL','FUS', 'TDP43','NONO', 'ANXA11', 'GM130', 'LAMP1', 
                                     'Calreticulin','PEX14', 'DCP1A', 'CD41', 'SQSTM1','PML',
                                     'SCNA','SNCA', 'FMRP','NEMO',
                                     'PSD95','KIF5A',  'CLTC', 'TOMM20','mitotracker', 'PURA', 'G3BP1', 
                                     'Phalloidin']
    
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
class UMAP0AlyssaCoyneColorByGroupPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()
               
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALYSSA
        # umap type
        self.UMAP_TYPE = 0

class UMAP0AlyssaCoyneColorByPatientPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()
               
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.REMOVE_MARKER.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALYSSA
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

class UMAP2AlyssaCoyneColorByGroupPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MULTIPLEX_CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALYSSA
        # umap type
        self.UMAP_TYPE = 2

class UMAP2AlyssaCoyneColorByPatientPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  None
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALYSSA
        # umap type
        self.UMAP_TYPE = 2

class UMAP2AlyssaCoyneColorByPatientColorControlsPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  None
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALYSSA
        # umap type
        self.UMAP_TYPE = 2
        self.TO_COLOR = ['Controls_rep1','Controls_rep2', 'Controls_rep3',
                         'Controls_rep4', 'Controls_rep5','Controls_rep6']

class UMAP2AlyssaCoyneColorByPatientColorC9PlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  None
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALYSSA
        # umap type
        self.UMAP_TYPE = 2
        self.TO_COLOR = ['c9orf72ALSPatients_rep1','c9orf72ALSPatients_rep2', 'c9orf72ALSPatients_rep3']        

class UMAP2AlyssaCoyneColorByPatientColorsALSPositivePlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  None
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALYSSA
        # umap type
        self.UMAP_TYPE = 2
        self.TO_COLOR = [f'sALSPositiveCytoTDP43_rep{i}' for i in range(1,11)]
        
class UMAP2AlyssaCoyneColorByPatientColorsALSNegativePlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  None
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALYSSA
        # umap type
        self.UMAP_TYPE = 2
        self.TO_COLOR = ['sALSNegativeCytoTDP43_rep1','sALSNegativeCytoTDP43_rep2']
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
        self.ORDERED_CELL_LINES_NAMES = ['FUSRevertant_Untreated','FUSHomozygous_Untreated','FUSHeterozygous_Untreated',
                                         'TDP43_Untreated','TBK1_Untreated','OPTN_Untreated']

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

class DistancesAlyssaCoynePlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_ALYSSA
