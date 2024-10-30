import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from manuscript.plot_config import PlotConfig
from src.datasets.label_utils import MapLabelsFunction

class UMAP1PlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()
        
        self.ORDERED_MARKER_NAMES = ["DAPI", 'FUS','PEX14','ANXA11','NONO',
                                     'GM130','TDP43','Calreticulin','Phalloidin',
                                     'TOMM20','mitotracker', 'CLTC','PURA',
                                    'SCNA','SNCA', 'KIF5A','SQSTM1',
                                    'CD41','G3BP1', 'FMRP', 'NCL', 
                                    'LAMP1', 'PML', 'DCP1A','PSD95', 
                                      'NEMO']
    
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

class UMAP0StressDAPIPlotConfig(UMAP0StressPlotConfig):
    def __init__(self):
        super().__init__()
               
        self.MIX_GROUPS = True

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
        self.ORDERED_CELL_LINES = ['FUSHomozygous_Untreated',
                                         'TDP43_Untreated','TBK1_Untreated','OPTN_Untreated']
        self.ORDERED_MARKERS = ['GM130','Phalloidin','G3BP1','CLTC','mitotracker','NCL','KIF5A', 'TOMM20', 'PURA',
         'NONO','SCNA','ANXA11','FUS','TDP43','PEX14', 'DAPI','PSD95','PML',
        'CD41','NEMO','Calreticulin','LAMP1','SQSTM1','DCP1A','FMRP']

class DistancesNeuronsFUSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_CONDITION_AND_ALS
        self.ORDERED_CELL_LINES = ['FUSHomozygous_Untreated','FUSHeterozygous_Untreated',
                                         'FUSRevertant_Untreated']
        self.ORDERED_MARKERS = ['GM130','Phalloidin','G3BP1','CLTC','mitotracker','NCL','KIF5A', 'TOMM20', 'PURA',
         'NONO','SCNA','ANXA11','FUS','TDP43','PEX14', 'DAPI','PSD95','PML',
        'CD41','NEMO','Calreticulin','LAMP1','SQSTM1','DCP1A','FMRP']

class DistancesNeuronsFUSD18PlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_CONDITION_AND_ALS
        self.ORDERED_CELL_LINES = ['FUSHomozygous_Untreated','FUSHeterozygous_Untreated',
                                         'FUSRevertant_Untreated']
        self.ORDERED_MARKERS = ['GM130','Phalloidin','G3BP1','CLTC','mitotracker','NCL','KIF5A', 'TOMM20', 'PURA',
         'NONO','SNCA','ANXA11','FUS','TDP43','PEX14', 'DAPI','PSD95','PML',
        'CD41','NEMO','Calreticulin','LAMP1','SQSTM1','DCP1A','FMRP']
class DistancesdNLSPlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_DOX
        self.YAXIS_CUT_RANGES = {'upper_graph':(0.55, 1), 'lower_graph':(-0.01, 0.25)}
        self.SHOW_BASELINE = True

class DistancesdNLSNoBaselinePlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_DOX
        self.YAXIS_CUT_RANGES = {'upper_graph':(0.55, 1), 'lower_graph':(-0.01, 0.25)}
        self.SHOW_BASELINE = False

class DistancesAlyssaCoynePlotConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_ALYSSA
