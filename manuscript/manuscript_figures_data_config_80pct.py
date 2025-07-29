import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.figures_config import FigureConfig
from manuscript.plot_config import PlotConfig
from src.datasets.label_utils import MapLabelsFunction
############################################################
# Figure 1
############################################################ 
class NeuronsUMAP1B9FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8", f) for f in 
                        ["batch9"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False
        
############################################################
# Figure 1 - supp
############################################################
class NeuronsUMAP1B6FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch680pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False

class NeuronsUMAP1B69FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch680pct", "batch980pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class NeuronsUMAP1B6WithoutDAPIFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch680pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False

class NeuronsUMAP1B9WithoutDapiFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch980pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False
############################################################
# Figure 2 
############################################################
class NeuronsUMAP0StressB9WithoutDAPIFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch980pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0StressB6WithoutDAPIFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch680pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0StressB4WithoutDAPIFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch480pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0StressB5WithoutDAPIFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch580pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0StressB9DAPIFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch980pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = None
        self.MARKERS = ['DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsDistancesStressFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in range(6,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['WT_stress']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class NeuronsDistancesStressWith45FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in range(4,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['WT_stress']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

############################################################
# Figure 2 - supp
############################################################
class NeuronsUMAP0StressB6FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch680pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class U2OSUMAP0StressDatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                        ["U2OS_spd_format"]]
        
        self.CELL_LINES = ['U2OS']
        self.EXPERIMENT_TYPE = 'U2OS'
        self.MARKERS = ['G3BP1', 'DCP1A', 'Phalloidin', 'DAPI']
        self.SHOW_ARI = True

############################################################
# Figure 3
############################################################
class dNLSUMAP0B3DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        # self.CELL_LINES = ['TDP43']
        self.MARKERS = ['TDP43']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP0B3TDP43DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.CELL_LINES = ['TDP43']
        self.MARKERS = ['TDP43','DCP1A']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP0B4TDP43DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.CELL_LINES = ['TDP43']
        # self.MARKERS = ['TDP43']#,'DCP1A']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP0B5TDP43DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch5"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.CELL_LINES = ['TDP43']
        self.MARKERS = ['TDP43','DCP1A']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        
class dNLSUMAP0B3DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch3"]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.CELL_LINES = ['TDP43']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=True

class dNLSUMAP0B4DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch4"]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.CELL_LINES = ['TDP43']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=True

class dNLSUMAP0B5DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch5"]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.CELL_LINES = ['TDP43']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=True

class dNLSDistancesFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(3,6)]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "TDP43_Untreated"
        self.CELL_LINES_CONDITIONS = ['TDP43_dox']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

############################################################
# Figure 3 - supp
############################################################

############################################################
# Figure 5
############################################################

class NeuronsUMAP2ALSFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
       
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2ALSB9FigureConfig(NeuronsUMAP2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch980pct"]

class NeuronsDistancesALSFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in range(6,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['FUSHomozygous_Untreated','FUSHeterozygous_Untreated',
                                      'TBK1_Untreated','OPTN_Untreated','TDP43_Untreated']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class NeuronsUMAP0ALSFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
       
        self.EXPERIMENT_TYPE = 'neurons80pct'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0ALSFigureConfig_FUS(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = None
        self.CELL_LINES = ['WT','FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.MARKERS = ['FUS']

        self.CONDITIONS = ['Untreated']
        
        self.EXPERIMENT_TYPE = 'neurons'

        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0ALSFigureConfig_FUS_allB(NeuronsUMAP0ALSFigureConfig_FUS):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch480pct", "batch580pct", "batch680pct", "batch980pct"]

class NeuronsUMAP0ALSFigureConfig_FUS_B4(NeuronsUMAP0ALSFigureConfig_FUS):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch480pct"]

class NeuronsUMAP0ALSFigureConfig_FUS_B5(NeuronsUMAP0ALSFigureConfig_FUS):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch580pct"]

class NeuronsUMAP0ALSFigureConfig_FUS_B6(NeuronsUMAP0ALSFigureConfig_FUS):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch680pct"]

class NeuronsUMAP0ALSFigureConfig_FUS_B9(NeuronsUMAP0ALSFigureConfig_FUS):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch980pct"]


class NeuronsUMAP0ALSB9FUSFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch980pct"]
        self.CELL_LINES = ['WT','FUSHomozygous']
        self.MARKERS = ['FUS']



class NeuronsUMAP0ALSB9DCP1AFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch980pct"]
        self.CELL_LINES = ['WT','TBK1','TDP43','FUSHomozygous']
        self.MARKERS = ['DCP1A']


class NeuronsUMAP0ALSB9ANXA11FigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch980pct"]
        self.CELL_LINES = ['WT','OPTN','TBK1']
        self.MARKERS = ['ANXA11']

class NeuronsUMAP0ALSB9CLTCFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch980pct"]
        self.CELL_LINES = ['WT','OPTN','TBK1']
        self.MARKERS = ['CLTC']

class NeuronsUMAP0ALSB9SQSTM1FigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch980pct"]
        self.CELL_LINES = ['WT','OPTN']
        self.MARKERS = ['SQSTM1']
class AlyssaCoyneDistancesFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_7tiles80pct'
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        self.BASELINE_CELL_LINE_CONDITION = "Controls_Untreated"
        self.CELL_LINES_CONDITIONS = ['sALSPositiveCytoTDP43_Untreated','sALSNegativeCytoTDP43_Untreated','c9orf72ALSPatients_Untreated']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        self.REPS = [f'rep{i}' for i in range(1,11)]

class AlyssaCoyneUMAP0FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]
      
        self.EXPERIMENT_TYPE = 'AlyssaCoyne'    
        self.CELL_LINES = ['Controls','sALSPositiveCytoTDP43', 
                           'sALSNegativeCytoTDP43','c9orf72ALSPatients']
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name
        self.ADD_REP_TO_LABEL=True
        self.ADD_BATCH_TO_LABEL=False
        self.ADD_CONDITION_TO_LABEL = False

class AlyssaCoyneUMAP2FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne'    
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.MULTIPLEX_CELL_LINES.name
        self.ADD_REP_TO_LABEL=True
        self.ADD_BATCH_TO_LABEL = False
        self.ADD_CONDITION_TO_LABEL = False


############################################################
# Figure 5 - supp
############################################################
class NeuronsUMAP2StressB6FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch680pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP2B3FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'    
        self.CELL_LINES = ['TDP43']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP2B4FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'    
        self.CELL_LINES = ['TDP43']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP2B5FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch5"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'    
        self.CELL_LINES = ['TDP43']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2StressB9FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch980pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2ALSB6FigureConfig(NeuronsUMAP2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch680pct"]


      
class NeuronsUMAP2ALSB6_without_fus_marker_FigureConfig(NeuronsUMAP2ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS_TO_EXCLUDE = ['TIA1','FUS']

class NeuronsUMAP2ALSB6ALSLines_wo_fusFigureConfig(NeuronsUMAP2ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS_TO_EXCLUDE = ['TIA1','FUS']
        self.CELL_LINES = ['WT','TDP43','FUSHomozygous', 
                           'TBK1','SCNA','OPTN']
        
class NeuronsUMAP2ALSB9ALSLines_wo_fusFigureConfig(NeuronsUMAP2ALSB9FigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS_TO_EXCLUDE = ['TIA1','FUS']
        self.CELL_LINES = ['WT','TDP43','FUSHomozygous',
                           'TBK1','SCNA','OPTN']
class NeuronsUMAP2ALSB6ALSLinesWOSNCAFigureConfig(NeuronsUMAP2ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','FUSHomozygous',
                           'TBK1','OPTN']
        
class NeuronsUMAP2ALSB6_without_fushomo_FigureConfig(NeuronsUMAP2ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','SCNA', 'FUSHeterozygous',
                           'TBK1','FUSRevertant','OPTN']
        
class NeuronsUMAP2ALSB6_without_fushetero_FigureConfig(NeuronsUMAP2ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','SCNA', 'FUSHomozygous',
                           'TBK1','FUSRevertant','OPTN']
        
class NeuronsUMAP2ALSB6ALSLinesFigureConfig(NeuronsUMAP2ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','SCNA', 'FUSHomozygous',
                           'TBK1','OPTN']
        
class NeuronsUMAP2ALSB69ALSLinesFigureConfig(NeuronsUMAP2ALSB6ALSLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch680pct", "batch980pct"]
        self.ADD_BATCH_TO_LABEL = False
        
class NeuronsUMAP2ALSB4ALSLinesFigureConfig(NeuronsUMAP2ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch480pct"]
        self.CELL_LINES = ['WT','TDP43','SCNA', 'FUSHomozygous',
                           'TBK1','OPTN']
        
class NeuronsUMAP2ALSB5ALSLinesFigureConfig(NeuronsUMAP2ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch580pct"]
        self.CELL_LINES = ['WT','TDP43','SCNA', 'FUSHomozygous',
                           'TBK1','OPTN']

class NeuronsUMAP2ALSB9_without_fus_marker_FigureConfig(NeuronsUMAP2ALSB9FigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS_TO_EXCLUDE = ['TIA1','FUS']

class NeuronsUMAP2ALSB9ALSLinesWOSNCAFigureConfig(NeuronsUMAP2ALSB9FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','FUSHomozygous',
                           'TBK1','OPTN']
        
class NeuronsUMAP2ALSB9_without_fushomo_FigureConfig(NeuronsUMAP2ALSB9FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','SCNA', 'FUSHeterozygous',
                           'TBK1','OPTN']
        
class NeuronsUMAP2ALSB9_without_fushetero_FigureConfig(NeuronsUMAP2ALSB9FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','SCNA', 'FUSHomozygous',
                           'TBK1','FUSRevertant','OPTN']
        
class NeuronsUMAP2ALSB9ALSLinesFigureConfig(NeuronsUMAP2ALSB9FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','SCNA', 'FUSHomozygous',
                           'TBK1','OPTN']
            
class NeuronsUMAP2ALSD18B1FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.CONDITIONS = ['Untreated']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0StressD18B1FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0StressD18B2FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch2"]
        
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0ALSD18B1FigureConfig_FUS(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['FUS']

        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0ALSD18B2FigureConfig_FUS(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch2"]
        
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['FUS']

        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2ALSB9FUSFigureConfig(NeuronsUMAP2ALSB9FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch980pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
class NeuronsUMAP2ALSD18B2FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch2"]
        
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.CONDITIONS = ['Untreated']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0ALSB9FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch980pct"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0ALS_FUSHomozygous_B9FigureConfig(NeuronsUMAP0ALSB9FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','FUSHomozygous']

class NeuronsUMAP0ALS_FUSHeterozygous_B9FigureConfig(NeuronsUMAP0ALSB9FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','FUSHeterozygous']

class NeuronsUMAP0ALS_FUSRevertant_B9FigureConfig(NeuronsUMAP0ALSB9FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','FUSRevertant']

class NeuronsUMAP0ALS_TBK1_B9FigureConfig(NeuronsUMAP0ALSB9FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','TBK1']

class NeuronsUMAP0ALS_OPTN_B9FigureConfig(NeuronsUMAP0ALSB9FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','OPTN']

class NeuronsUMAP0ALS_TDP43_B9FigureConfig(NeuronsUMAP0ALSB9FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','TDP43']


class NeuronsUMAP0ALSB6FUSFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch680pct"]
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS = ['FUS']
class NeuronsUMAP0ALSB6DCP1AFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch680pct"]
        self.CELL_LINES = ['WT','TBK1','TDP43','FUSHomozygous']
        self.MARKERS = ['DCP1A']
class NeuronsUMAP0ALSB6ANXA11FigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch680pct"]
        self.CELL_LINES = ['WT','OPTN','TBK1']
        self.MARKERS = ['ANXA11']

class NeuronsUMAP0ALSB6TDP43FigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch680pct"]
        self.CELL_LINES = ['WT','OPTN','TBK1','FUSHomozygous','TDP43']
        self.MARKERS = ['TDP43']
class NeuronsUMAP0ALSB6CLTCFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch680pct"]
        self.CELL_LINES = ['WT','OPTN','TBK1']
        self.MARKERS = ['CLTC']
class NeuronsUMAP0ALSB6FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch680pct"]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0ALS_FUSHomozygous_B6FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','FUSHomozygous']

class NeuronsUMAP0ALS_FUSHeterozygous_B6FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','FUSHeterozygous']

class NeuronsUMAP0ALS_FUSRevertant_B6FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','FUSRevertant']

class NeuronsUMAP0ALS_TBK1_B6FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','TBK1']

class NeuronsUMAP0ALS_OPTN_B6FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','OPTN']

class NeuronsUMAP0ALS_TDP43_B6FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT','TDP43']


class NeuronsDistancesALSWith45FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in range(4,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['FUSHomozygous_Untreated',
                                      'TBK1_Untreated','OPTN_Untreated','TDP43_Untreated']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

############################################################
# experimental
############################################################

class NeuronsDistancesALSD18FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(1,3)]]
        
        self.EXPERIMENT_TYPE = 'neurons_d1880pct'
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['FUSHomozygous_Untreated','FUSHeterozygous_Untreated','FUSRevertant_Untreated']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class NeuronsDistancesALSFUSFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in range(4,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['FUSHomozygous_Untreated','FUSHeterozygous_Untreated','FUSRevertant_Untreated']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class NeuronsUMAP1D18B1FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = None#['TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False

class NeuronsUMAP1D18B2FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch2"]
        
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = None#['TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False
        
class NeuronsUMAP0ALS_FUSHomozygous_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch680pct','batch980pct']
        self.CELL_LINES = ['WT','FUSHomozygous']
        self.ADD_BATCH_TO_LABEL = False
class NeuronsUMAP0ALS_FUSHeterozygous_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch680pct','batch980pct']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','FUSHeterozygous']

class NeuronsUMAP0ALS_FUSRevertant_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch680pct','batch980pct']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','FUSRevertant']

class NeuronsUMAP0ALS_TBK1_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch680pct','batch980pct']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','TBK1']

class NeuronsUMAP0ALS_OPTN_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch680pct','batch980pct']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','OPTN']

class NeuronsUMAP0ALS_TDP43_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch680pct','batch980pct']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','TDP43']

################
## New Alyssa
################

# UMAP 1

class newAlyssaFigureConfig_UMAP1_B1(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.CELL_LINES = None
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = None
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newAlyssaFigureConfig_UMAP1_B1_C9_CS2YNL(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['C9-CS2YNL']

class newAlyssaFigureConfig_UMAP1_B1_C9_CS7VCZ(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['C9-CS7VCZ']

class newAlyssaFigureConfig_UMAP1_B1_C9_CS8RFT(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['C9-CS8RFT']

class newAlyssaFigureConfig_UMAP1_B1_Ctrl_EDi022(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['Ctrl-EDi022']

class newAlyssaFigureConfig_UMAP1_B1_Ctrl_EDi029(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['Ctrl-EDi029']

class newAlyssaFigureConfig_UMAP1_B1_Ctrl_EDi037(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['Ctrl-EDi037']

class newAlyssaFigureConfig_UMAP1_B1_SALSNegative_CS0ANK(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSNegative-CS0ANK']

class newAlyssaFigureConfig_UMAP1_B1_SALSNegative_CS0JPP(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSNegative-CS0JPP']

class newAlyssaFigureConfig_UMAP1_B1_SALSNegative_CS6ZU8(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSNegative-CS6ZU8']

class newAlyssaFigureConfig_UMAP1_B1_SALSPositive_CS2FN3(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSPositive-CS2FN3']

class newAlyssaFigureConfig_UMAP1_B1_SALSPositive_CS4ZCD(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSPositive-CS4ZCD']

class newAlyssaFigureConfig_UMAP1_B1_SALSPositive_CS7TN6(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSPositive-CS7TN6']


class newAlyssaFigureConfig_UMAP1_B1_C9(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['C9-CS2YNL', 'C9-CS7VCZ', 'C9-CS8RFT']

class newAlyssaFigureConfig_UMAP1_B1_Ctrl(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['Ctrl-EDi022', 'Ctrl-EDi029', 'Ctrl-EDi037']

class newAlyssaFigureConfig_UMAP1_B1_SALSNegative(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSNegative-CS0ANK', 'SALSNegative-CS0JPP', 'SALSNegative-CS6ZU8']

class newAlyssaFigureConfig_UMAP1_B1_SALSPositive(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSPositive-CS2FN3', 'SALSPositive-CS4ZCD', 'SALSPositive-CS7TN6']


# UMAP2
class newAlyssaFigureConfig_UMAP2_B1(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']
        
        self.MARKERS_TO_EXCLUDE = None

        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = None
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

        self.REMOVE_PATIENT_ID_FROM_CELL_LINE = True

#############
## New iNDI
############

# UMAP1

class newNeuronsD8FigureConfig_UMAP1(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP1_B1(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP1_B2(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_UMAP1_B3(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP1_B7(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP1_B8(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP1_B9(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP1_B10(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch10']

# UMAP0

class newNeuronsD8FigureConfig_UMAP0(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CELL_LINES = ['WT']
        
        self.MARKERS_TO_EXCLUDE = ['DAPI']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP0_B1(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP0_B2(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_UMAP0_B3(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP0_B7(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP0_B8(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP0_B9(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP0_B10(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch10']


# UMAP0 ALS
class newNeuronsD8FigureConfig_ALS_UMAP0(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = None
        self.MARKERS = ['DAPI']
        # self.MARKERS_TO_EXCLUDE = ['DAPI']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_ALS_UMAP0_B1(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_ALS_UMAP0_B2(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_ALS_UMAP0_B3(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_ALS_UMAP0_B7(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_ALS_UMAP0_B8(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_ALS_UMAP0_B9(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_ALS_UMAP0_B10(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch10']

# UMAP0 FUS lines
class newNeuronsD8FigureConfig_FUSLines_UMAP0(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = None
        self.MARKERS_TO_EXCLUDE = ['DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B1(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B2(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B3(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B7(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B8(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B9(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B10(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch10']

# UMAP0 FUS
class newNeuronsD8FigureConfig_FUS_UMAP0(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['FUS']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_FUS_UMAP0_B1(newNeuronsD8FigureConfig_FUS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_FUS_UMAP0_B2(newNeuronsD8FigureConfig_FUS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_FUS_UMAP0_B3(newNeuronsD8FigureConfig_FUS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_FUS_UMAP0_B7(newNeuronsD8FigureConfig_FUS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_FUS_UMAP0_B8(newNeuronsD8FigureConfig_FUS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_FUS_UMAP0_B9(newNeuronsD8FigureConfig_FUS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_FUS_UMAP0_B10(newNeuronsD8FigureConfig_FUS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch10']

# UMAP0 DCP1A
class newNeuronsD8FigureConfig_DCP1A_UMAP0(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CELL_LINES = None
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['DCP1A']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_DCP1A_UMAP0_B1(newNeuronsD8FigureConfig_DCP1A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_DCP1A_UMAP0_B2(newNeuronsD8FigureConfig_DCP1A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_DCP1A_UMAP0_B3(newNeuronsD8FigureConfig_DCP1A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_DCP1A_UMAP0_B7(newNeuronsD8FigureConfig_DCP1A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_DCP1A_UMAP0_B8(newNeuronsD8FigureConfig_DCP1A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_DCP1A_UMAP0_B9(newNeuronsD8FigureConfig_DCP1A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_DCP1A_UMAP0_B10(newNeuronsD8FigureConfig_DCP1A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch10']


# UMAP0 LSM14A
class newNeuronsD8FigureConfig_LSM14A_UMAP0(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CELL_LINES = None
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['LSM14A']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_LSM14A_UMAP0_B1(newNeuronsD8FigureConfig_LSM14A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_LSM14A_UMAP0_B2(newNeuronsD8FigureConfig_LSM14A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_LSM14A_UMAP0_B3(newNeuronsD8FigureConfig_LSM14A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_LSM14A_UMAP0_B7(newNeuronsD8FigureConfig_LSM14A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_LSM14A_UMAP0_B8(newNeuronsD8FigureConfig_LSM14A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_LSM14A_UMAP0_B9(newNeuronsD8FigureConfig_LSM14A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_LSM14A_UMAP0_B10(newNeuronsD8FigureConfig_LSM14A_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch10']


# UMAP2
class newNeuronsD8FigureConfig_UMAP2(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.MARKERS_TO_EXCLUDE = ['CD41', 'LAMP1']

        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = None
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP2_B1(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP2_B2(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_UMAP2_B3(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP2_B7(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP2_B8(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP2_B9(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP2_B10(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch10']

##########
# New dNLS
###########

class newDNLSUntreatedUMAP1DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch4', 'batch5', 'batch6']
        
        self.MARKERS_TO_EXCLUDE = []
        self.EXPERIMENT_TYPE = 'dNLS'
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['TIA1']

        self.SHOW_ARI = None#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.MARKERS.name

class newDNLSUntreatedUMAP1DatasetConfig_B1(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class newDNLSUntreatedUMAP1DatasetConfig_B2(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]

class newDNLSUntreatedUMAP1DatasetConfig_B3(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]

class newDNLSUntreatedUMAP1DatasetConfig_B4(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch4"]

class newDNLSUntreatedUMAP1DatasetConfig_B5(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch5"]

class newDNLSUntreatedUMAP1DatasetConfig_B6(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch6"]

##

class newDNLSDoxUMAP1DatasetConfig(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.CONDITIONS = ['DOX']

class newDNLSDoxUMAP1DatasetConfig_B1(newDNLSDoxUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class newDNLSDoxUMAP1DatasetConfig_B2(newDNLSDoxUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]

class newDNLSDoxUMAP1DatasetConfig_B3(newDNLSDoxUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]

class newDNLSDoxUMAP1DatasetConfig_B4(newDNLSDoxUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch4"]

class newDNLSDoxUMAP1DatasetConfig_B5(newDNLSDoxUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch5"]

class newDNLSDoxUMAP1DatasetConfig_B6(newDNLSDoxUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch6"]

## UMAP 0

class newDNLSUMAP0DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = None
        
        self.MARKERS_TO_EXCLUDE = []
        self.EXPERIMENT_TYPE = 'dNLS'
        self.CELL_LINES = ['dNLS']
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL=False #True
        self.ADD_BATCH_TO_LABEL = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES_CONDITIONS.name
        
class newDNLSUMAP0B1DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]

class newDNLSUMAP0B2DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch2"]

class newDNLSUMAP0B3DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch3"]

class newDNLSUMAP0B4DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch4"]

class newDNLSUMAP0B4DatasetConfig_TDP43(newDNLSUMAP0B4DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['TDP43']
        self.ADD_REP_TO_LABEL = False
        self.ARI_LABELS_FUNC = None

class newDNLSUMAP0B5DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch5"]


class newDNLSUMAP0B5DatasetConfig_TDP43(newDNLSUMAP0B5DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['TDP43']
        self.ADD_REP_TO_LABEL = False
        self.ARI_LABELS_FUNC = None

class newDNLSUMAP0B6DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch6"]

class newDNLSUMAP0DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = []

        
class newDNLSUMAP0B1DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]

class newDNLSUMAP0B2DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch2"]

class newDNLSUMAP0B3DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch3"]

class newDNLSUMAP0B4DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch4"]

class newDNLSUMAP0B5DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch5"]

class newDNLSUMAP0B6DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch6"]

# UMAP2

class newDNLSD8FigureConfig_UMAP2(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch4', 'batch5', 'batch6']
        
        self.EXPERIMENT_TYPE = 'dNLS'  
        self.CELL_LINES = ['dNLS']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newDNLSD8FigureConfig_UMAP2_B1(newDNLSD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch1']

class newDNLSD8FigureConfig_UMAP2_B2(newDNLSD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch2']

class newDNLSD8FigureConfig_UMAP2_B3(newDNLSD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch3']

class newDNLSD8FigureConfig_UMAP2_B4(newDNLSD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch4']

class newDNLSD8FigureConfig_UMAP2_B5(newDNLSD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch5']

class newDNLSD8FigureConfig_UMAP2_B6(newDNLSD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ['batch6']

##

class NeuronsConfig_Positive_stress(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [f"batch480pct", f"batch580pct", f"batch680pct", f"batch980pct"]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.ADD_BATCH_TO_LABEL = False
        self.ADD_REP_TO_LABEL = False        
        self.CELL_LINES = ['WT']
        self.MARKERS = None
        self.SHOW_ARI = True

class NeuronsConfig_Positive_stress_G3BP1(NeuronsConfig_Positive_stress):
    def __init__(self):
        super().__init__()

        self.MARKERS = ["G3BP1"]

class NeuronsConfig_Positive_stress_FMRP(NeuronsConfig_Positive_stress):
    def __init__(self):
        super().__init__()

        self.MARKERS = ["FMRP"]

class NeuronsConfig_Positive_stress_PURA(NeuronsConfig_Positive_stress):
    def __init__(self):
        super().__init__()

        self.MARKERS = ["PURA"]


class dNLSConfig_Positive_TDP43(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch3", "batch4", "batch5"]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.CELL_LINES = ['TDP43']
        self.MARKERS = ['TDP43']
        self.ADD_BATCH_TO_LABEL = False
        self.ADD_REP_TO_LABEL=False
        self.SHOW_ARI = True

class new_dNLSConfig_Positive_TDP43(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [f"batch{i}" for i in range(1,7)]
        
        self.EXPERIMENT_TYPE = 'deltaNLS_new'
        self.CELL_LINES = ['dNLS']
        self.MARKERS = ['TDP43']
        self.ADD_BATCH_TO_LABEL = False
        self.ADD_REP_TO_LABEL=False
        self.SHOW_ARI = True