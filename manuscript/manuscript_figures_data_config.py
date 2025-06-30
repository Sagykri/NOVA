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
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
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
                        ["batch6"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False

class NeuronsUMAP1B6WithoutDAPIFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
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
                        ["batch9"]]
        
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
                        ["batch9"]]
        
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
                        ["batch9"]]
        
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
                        [f"batch{i}" for i in range(6,10)]]
        
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
                        [f"batch{i}" for i in range(4,10)]]
        
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
                        ["batch6"]]
        
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
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        # self.CELL_LINES = ['TDP43']
        self.MARKERS = ['TDP43B']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP0B3TDP43DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.CELL_LINES = ['TDP43']
        self.MARKERS = ['TDP43B','DCP1A']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP0B4TDP43DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.CELL_LINES = ['TDP43']
        self.MARKERS = ['TDP43B','DCP1A']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP0B5TDP43DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch5"]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.CELL_LINES = ['TDP43']
        self.MARKERS = ['TDP43B','DCP1A']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        
class dNLSDistancesFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(3,6)]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "TDP43_Untreated"
        self.CELL_LINES_CONDITIONS = ['TDP43_dox']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class dNLSEffectsFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(3,6)]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']

        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class dNLSNewEffectsFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [f"batch{i}" for i in range(1,7)]
        
        self.EXPERIMENT_TYPE = 'deltaNLS_new'
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())

class NeuronsEffectsFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in [4,5,6,9]]]
        
        self.EXPERIMENT_TYPE = 'neurons'
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
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
        self.INPUT_FOLDERS = ["batch9"]

class NeuronsDistancesALSFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
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
       
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False


class NeuronsUMAP0ALSB9FUSFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch9"]
        self.CELL_LINES = ['WT','FUSHomozygous']
        self.MARKERS = ['FUS']

class NeuronsUMAP0ALSB9DCP1AFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch9"]
        self.CELL_LINES = ['WT','TBK1','TDP43','FUSHomozygous']
        self.MARKERS = ['DCP1A']


class NeuronsUMAP0ALSB9ANXA11FigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch9"]
        self.CELL_LINES = ['WT','OPTN','TBK1']
        self.MARKERS = ['ANXA11']

class NeuronsUMAP0ALSB9CLTCFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch9"]
        self.CELL_LINES = ['WT','OPTN','TBK1']
        self.MARKERS = ['CLTC']

class NeuronsUMAP0ALSB9SQSTM1FigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch9"]
        self.CELL_LINES = ['WT','OPTN']
        self.MARKERS = ['SQSTM1']
class AlyssaCoyneDistancesFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_7tiles'
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
      
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_7tiles'    
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
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_7tiles'    
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
                        ["batch6"]]
        
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
        
        self.EXPERIMENT_TYPE = 'deltaNLS'    
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
        
        self.EXPERIMENT_TYPE = 'deltaNLS'    
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
        
        self.EXPERIMENT_TYPE = 'deltaNLS'    
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
                        ["batch9"]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2ALSB6FigureConfig(NeuronsUMAP2ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch6"]


      
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

class CellPaintingNeuronsUMAP2ALSB9ALSLinesFigureConfig(NeuronsUMAP2ALSB9ALSLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS = ['DAPI','Calreticulin','NCL','mitotracker','Phalloidin','GM130']
        self.MARKERS_TO_EXCLUDE = ['TDP43','NONO','ANXA11','LAMP1','FUS','PEX14',
                                   'DCP1A','CD41','SQSTM1','PML','SCNA','SNCA','NEMO',
                                   'PSD95','KIF5A','CLTC','TOMM20','PURA','G3BP1','FMRP']
class NeuronsUMAP2ALSD18B1FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2ALSB9FUSFigureConfig(NeuronsUMAP2ALSB9FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
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
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP0ALSB9FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
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

        self.INPUT_FOLDERS = ["batch6"]
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS = ['FUS']
class NeuronsUMAP0ALSB6DCP1AFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch6"]
        self.CELL_LINES = ['WT','TBK1','TDP43','FUSHomozygous']
        self.MARKERS = ['DCP1A']
class NeuronsUMAP0ALSB6ANXA11FigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch6"]
        self.CELL_LINES = ['WT','OPTN','TBK1']
        self.MARKERS = ['ANXA11']

class NeuronsUMAP0ALSB6TDP43FigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch6"]
        self.CELL_LINES = ['WT','OPTN','TBK1','FUSHomozygous','TDP43']
        self.MARKERS = ['TDP43']
class NeuronsUMAP0ALSB6CLTCFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch6"]
        self.CELL_LINES = ['WT','OPTN','TBK1']
        self.MARKERS = ['CLTC']
class NeuronsUMAP0ALSB6FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = ["batch6"]
        
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
                        [f"batch{i}" for i in range(4,10)]]
        
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
        
        self.EXPERIMENT_TYPE = 'neurons_d18'
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['FUSHomozygous_Untreated','FUSHeterozygous_Untreated','FUSRevertant_Untreated']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class NeuronsDistancesALSFUSFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(4,10)]]
        
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
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL = False
        
class NeuronsUMAP0ALS_FUSHomozygous_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch6','batch9']
        self.CELL_LINES = ['WT','FUSHomozygous']
        self.ADD_BATCH_TO_LABEL = False
class NeuronsUMAP0ALS_FUSHeterozygous_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch6','batch9']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','FUSHeterozygous']

class NeuronsUMAP0ALS_FUSRevertant_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch6','batch9']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','FUSRevertant']

class NeuronsUMAP0ALS_TBK1_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch6','batch9']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','TBK1']

class NeuronsUMAP0ALS_OPTN_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch6','batch9']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','OPTN']

class NeuronsUMAP0ALS_TDP43_B69FigureConfig(NeuronsUMAP0ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch6','batch9']
        self.ADD_BATCH_TO_LABEL = False

        self.CELL_LINES = ['WT','TDP43']




class NeuronsUMAP0ALSFUSFigureConfig(NeuronsUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch9"]
        self.CELL_LINES = ['WT','FUSHomozygous']
        self.ADD_BATCH_TO_LABEL = False
        self.ADD_REP_TO_LABEL = False
        self.SHOW_ARI = False