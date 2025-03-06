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



############################################################
# NIH
############################################################
class NeuronsUMAP1B9NIH8DaysFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "NIH", f) for f in 
                        ["batch1","batch2","batch3",]]
        
        self.EXPERIMENT_TYPE = 'NIH_d8'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        self.MARKERS_TO_EXCLUDE = ['TUJ1', 'TIA1', 'DAPI']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False
        self.ADD_REP_TO_LABEL = False

class NIH8DaysDistancesStressFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        """Boxplot of WT stress vs untreated
        """
        self.INPUT_FOLDERS =  [os.path.join(self.PROCESSED_FOLDER_ROOT, "NIH", f) for f in 
                        ["batch2", "batch3"]]
        
        self.EXPERIMENT_TYPE = 'NIH_d8'
        self.MARKERS_TO_EXCLUDE = []
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['WT_stress']
        self.CELL_LINES = ['WT']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class NeuronsUMAP0StressNIH8DaysFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - WT untreated vs stress
        """
        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "NIH", f) for f in 
                        ["batch3",]]
        
        self.EXPERIMENT_TYPE = 'NIH_d8'    
        
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = []
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False
        self.ADD_REP_TO_LABEL=False

class NIH8DaysDistancesALSFigureConfig(FigureConfig):
    def __init__(self):
        """Bubbleplot of WT vs other cell lines
        """
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "NIH", f) for f in 
                        ["batch1", "batch2","batch3"]]
        
        self.EXPERIMENT_TYPE = 'NIH_d8'    
        self.MARKERS_TO_EXCLUDE = []
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.CELL_LINES_CONDITIONS = ['FUSHomozygous_Untreated','FUSHeterozygous_Untreated','FUSRevertant_Untreated']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class NIH8DaysUMAP2FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP2 multiplex of WT untreated vs stress
        """
        # Batches used for model development   
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "NIH", f) for f in 
                        ["batch3",]]
        self.EXPERIMENT_TYPE = 'NIH_d8'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['TUJ1', 'TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NIH8DaysUMAP2ALSD18B2FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP2 multiplex of WT untreated vs cell lines
        """
        # Batches used for model development   
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "NIH", f) for f in 
                        ["batch3",]]
        self.EXPERIMENT_TYPE = 'NIH_d8'   
        self.MARKERS_TO_EXCLUDE = ['FUS'] 
        self.CONDITIONS = ['Untreated']
        # self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NIH8DaysUMAP0ALSFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        self.EXPERIMENT_TYPE = 'NIH_d8'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NIH8DaysUMAP0ALSB9FUSFigureConfig(NIH8DaysUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS = ['FUS']

class NIH8DaysUMAP0ALSB9DCP1AFigureConfig(NIH8DaysUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS = ['DCP1A']

class NIH8DaysUMAP0ALSB9ANXA11FigureConfig(NIH8DaysUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS = ['ANAX11']

class NIH8DaysUMAP0ALSB9CLTCFigureConfig(NIH8DaysUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS = ['CLTC']

class NIH8DaysUMAP0ALSB9SQSTM1FigureConfig(NIH8DaysUMAP0ALSFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS = ['SQSTM1']

############################################################
# funova
############################################################
control_cell_lines = ["Control-1001733","Control-1017118","Control-1025045","Control-1048087"]
c9orf72_cell_lines = ["C9orf72-HRE-1008566","C9orf72-HRE-981344"]
tdp43_cell_lines = ["TDP--43-G348V-1057052","TDP--43-N390D-1005373"]
all_cell_lines = control_cell_lines + c9orf72_cell_lines + tdp43_cell_lines
# Marker categories
PROTEOSTASIS_MARKERS = ['Stress-initiation', 'mature-Autophagosome', 'Ubiquitin-levels', 'UPR-IRE1a', 'UPR-ATF4', 'UPR-ATF6', 'impaired-Autophagosome', 'Protein-degradation']
NEURONAL_CELL_DEATH_SENESCENCE_MARKERS = ['Autophagy', 'Parthanatos-late', 'DNA-damage-pH2Ax', 'Parthanatos-early', 'Necrosis', 'Necroptosis-HMGB1', 'DNA-damage-P53BP1', 'Apoptosis', 'Necroptosis-pMLKL']
SYNAPTIC_NEURONAL_FUNCTION_MARKERS = ['Cytoskeleton', 'Neuronal-activity', 'Senescence-signaling']
DNA_RNA_DEFECTS_MARKERS = ['Aberrant-splicing', 'Nuclear-speckles-SC35', 'Splicing-factories', 'Nuclear-speckles-SON']
PATHOLOGICAL_PROTEIN_AGGREGATION_MARKERS = ['TDP-43']

class NeuronsUMAP0StressFunovaFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - WT untreated vs stress
        """        
        self.EXPERIMENT_TYPE = 'funova'    
        self.MARKERS_TO_EXCLUDE = []
        self.SHOW_ARI = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.CONDITIONS.name

class FunovaUMAP0CellLinesFigureConfig(NeuronsUMAP0StressFunovaFigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - Cell lines 
        """        
        # self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
        #                 ["Batch1",]]
        self.ADD_LINE_TO_LABEL = True
        self.MARKERS_TO_EXCLUDE = []
        self.ADD_REP_TO_LABEL=False   
        self.SHOW_ARI = True   
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES_CONDITIONS.name 
        # self.CONDITIONS = ['stress'] 

# class umap0NS(NeuronsUMAP0StressFunovaFigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch4",]]
#         self.MARKERS = ['Nuclear-speckles-SON']
        

# class FunovaUMAP0CellLinesFigureConfigDAPI(FunovaUMAP0CellLinesFigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES = ["Control-1025045", "C9orf72-HRE-1008566"]
#         self.MARKERS = ['DAPI']
#         self.REPS = ['rep1']
#         self.CONDITIONS = ['Untreated']

# class FunovaUMAP0CellLinesFigureConfigDAPIB1(FunovaUMAP0CellLinesFigureConfigDAPI):
#     def __init__(self):
#         super().__init__()
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch1",]]

# class FunovaUMAP0CellLinesFigureConfigDAPIB2(FunovaUMAP0CellLinesFigureConfigDAPI):
#     def __init__(self):
#         super().__init__()
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch2",]]
        
# class FunovaUMAP0CellLinesFigureConfigDAPIB3(FunovaUMAP0CellLinesFigureConfigDAPI):
#     def __init__(self):
#         super().__init__()
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch3",]]
        
# class FunovaUMAP0CellLinesFigureConfigDAPIB4(FunovaUMAP0CellLinesFigureConfigDAPI):
#     def __init__(self):
#         super().__init__()
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch4",]]

# class FunovaUMAP0CellLinesFigureConfigPE(FunovaUMAP0CellLinesFigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES = ["Control-1025045", "TDP--43-G348V-1057052"]
#         self.MARKERS = ['Parthanatos-early']
#         self.REPS = ['rep1']
#         self.CONDITIONS = ['Untreated']
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch4",]]

# class FunovaAllCLs(NeuronsUMAP0StressFunovaFigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch4",]]
#         self.CELL_LINES = [all_cell_lines[0]]
#         self.SHOW_ARI = False
#         self.MARKERS = ['DAPI']

class NeuronsUMAP1B9FunovaFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """        
        self.EXPERIMENT_TYPE = 'funova'  
        self.CONDITIONS = ['stress']
        self.MARKERS_TO_EXCLUDE = ['DAPI']
        self.SHOW_ARI = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.MARKERS.name

class FunovaUMAP2FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP2 multiplex of WT untreated vs stress
        """
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4",]]
        self.EXPERIMENT_TYPE = 'funova'    
        # self.CELL_LINES = []
        self.MARKERS_TO_EXCLUDE = []
        # self.MARKERS = ['DAPI']
        self.SHOW_ARI = False
        self.ADD_REP_TO_LABEL=False
        self.ADD_LINE_TO_LABEL = True
        self.ADD_CONDITION_TO_LABEL = True
        # self.CELL_LINES = ["C9orf72-HRE-1008566"] + control_cell_lines
        # self.CONDITIONS = ['Untreated']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES_CONDITIONS.name

# class FunovaUMAP2FigureConfigAllControls(FigureConfig):
#     def __init__(self):
#         super().__init__()
#         """UMAP2 multiplex of WT untreated vs stress
#         """
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch4",]]
#         self.EXPERIMENT_TYPE = 'funova'    
#         self.CELL_LINES = control_cell_lines
#         self.MARKERS_TO_EXCLUDE = []
#         self.SHOW_ARI = True
#         self.ADD_REP_TO_LABEL=False
#         self.ADD_LINE_TO_LABEL = True

# class FunovaUMAP2FigureConfigAllC9s(FigureConfig):
#     def __init__(self):
#         super().__init__()
#         """UMAP2 multiplex of WT untreated vs stress
#         """
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch4",]]
#         self.EXPERIMENT_TYPE = 'funova'    
#         self.CELL_LINES = c9orf72_cell_lines
#         self.MARKERS_TO_EXCLUDE = []
#         self.SHOW_ARI = True
#         self.ADD_REP_TO_LABEL=False
#         self.ADD_LINE_TO_LABEL = True

# class FunovaUMAP2FigureConfigAllTDPs(FigureConfig):
#     def __init__(self):
#         super().__init__()
#         """UMAP2 multiplex of WT untreated vs stress
#         """
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
#                         ["Batch4",]]
#         self.EXPERIMENT_TYPE = 'funova'    
#         self.CELL_LINES = tdp43_cell_lines
#         self.MARKERS_TO_EXCLUDE = []
#         self.SHOW_ARI = True
#         self.ADD_REP_TO_LABEL=False
#         self.ADD_LINE_TO_LABEL = True

class FunovaUMAP2ALSD18B2FigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP2 multiplex of WT untreated vs cell lines
        """
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4",]]
        self.EXPERIMENT_TYPE = 'funova'   
        self.MARKERS_TO_EXCLUDE = [] 
        self.CONDITIONS = ['Untreated']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class FunovaDistancesALSFigureConfig(FigureConfig):
    def __init__(self):
        """Bubbleplot of WT vs other cell lines
        """
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2","Batch3", "Batch4"]]
        
        self.EXPERIMENT_TYPE = 'funova'    
        self.MARKERS_TO_EXCLUDE = []
        self.BASELINE_CELL_LINE_CONDITION = "Control_Untreated"
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES_CONDITIONS = ["C9orf72-HRE-1008566_Untreated","C9orf72-HRE-981344_Untreated", 
                                      "TDP--43-G348V-1057052_Untreated","TDP--43-N390D-1005373_Untreated"]
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.COMMON_CELL_LINES.name

class FunovaDistancesStressFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        """Boxplot of WT stress vs untreated
        """
        self.INPUT_FOLDERS =  [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2","Batch3", "Batch4"]]
        
        self.EXPERIMENT_TYPE = 'funova'
        self.MARKERS_TO_EXCLUDE = []
        self.MARKERS = ['DAPI']
        self.BASELINE_CELL_LINE_CONDITION = "Control-1025045_Untreated"
        self.CELL_LINES_CONDITIONS = ['C9orf72-HRE-1008566_Untreated']
        # self.CELL_LINES = ['TDP--43']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        # self.ARI_LABELS_FUNC = MapLabelsFunction.COMMON_CELL_LINES.name
        self.CONDITIONS = ['Untreated']

class Funova_Batch1_Config(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.SHOW_ARI = False

class Funova_Batch2_Config(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.SHOW_ARI = True

class Funova_Batch3_Config(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.SHOW_ARI = True

class Funova_Batch4_Config(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.SHOW_ARI = True

class Funova_controls_tdp_untreated(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines + tdp43_cell_lines
        self.CONDITIONS = ['Untreated']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_tdp_stress(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines + tdp43_cell_lines
        self.CONDITIONS = ['stress']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_c9_untreated(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines + c9orf72_cell_lines
        self.CONDITIONS = ['Untreated']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_c9_stress(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines + c9orf72_cell_lines
        self.CONDITIONS = ['stress']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_stress(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CONDITIONS = ['stress']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_untreated(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CONDITIONS = ['Untreated']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_untreated_si(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines 
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['Stress-initiation']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_stress_si(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines
        self.CONDITIONS = ['stress'] 
        self.MARKERS = ['Stress-initiation']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_tdp_untreated_si(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = tdp43_cell_lines 
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['Stress-initiation']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_tdp_stress_si(Funova_Batch4_Config):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = tdp43_cell_lines
        self.CONDITIONS = ['stress'] 
        self.MARKERS = ['Stress-initiation']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_control_untreated_tdp(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['TDP-43']
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1","Batch2","Batch3","Batch4",]]
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name
        
class Funova_control_stress_tdp(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines
        self.CONDITIONS = ['stress']
        self.MARKERS = ['TDP-43']
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1","Batch2","Batch3","Batch4",]]
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name


FigureConfigToUse = NeuronsUMAP0StressFunovaFigureConfig

class Funova_Batch1_Control_1001733_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["Control-1001733"]

class Funova_Batch1_Control_1017118_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["Control-1017118"]

class Funova_Batch1_Control_1025045_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["Control-1025045"]

class Funova_Batch1_Control_1048087_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["Control-1048087"]

class Funova_Batch1_C9orf72_HRE_1008566_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["C9orf72-HRE-1008566"]

class Funova_Batch1_C9orf72_HRE_981344_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["C9orf72-HRE-981344"]

class Funova_Batch1_TDP_43_G348V_1057052_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["TDP--43-G348V-1057052"]

class Funova_Batch1_TDP_43_N390D_1005373_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["TDP--43-N390D-1005373"]

class Funova_Batch2_Control_1001733_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["Control-1001733"]

class Funova_Batch2_Control_1017118_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["Control-1017118"]

class Funova_Batch2_Control_1025045_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["Control-1025045"]

class Funova_Batch2_Control_1048087_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["Control-1048087"]

class Funova_Batch2_C9orf72_HRE_1008566_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["C9orf72-HRE-1008566"]

class Funova_Batch2_C9orf72_HRE_981344_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["C9orf72-HRE-981344"]

class Funova_Batch2_TDP_43_G348V_1057052_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["TDP--43-G348V-1057052"]

class Funova_Batch2_TDP_43_N390D_1005373_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["TDP--43-N390D-1005373"]

class Funova_Batch3_Control_1001733_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["Control-1001733"]

class Funova_Batch3_Control_1017118_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["Control-1017118"]

class Funova_Batch3_Control_1025045_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["Control-1025045"]

class Funova_Batch3_Control_1048087_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["Control-1048087"]

class Funova_Batch3_C9orf72_HRE_1008566_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["C9orf72-HRE-1008566"]

class Funova_Batch3_C9orf72_HRE_981344_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["C9orf72-HRE-981344"]

class Funova_Batch3_TDP_43_G348V_1057052_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["TDP--43-G348V-1057052"]

class Funova_Batch3_TDP_43_N390D_1005373_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["TDP--43-N390D-1005373"]


class Funova_Batch4_Control_1001733_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["Control-1001733"]

class Funova_Batch4_Control_1017118_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["Control-1017118"]

class Funova_Batch4_Control_1025045_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["Control-1025045"]

class Funova_Batch4_Control_1048087_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["Control-1048087"]

class Funova_Batch4_C9orf72_HRE_1008566_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["C9orf72-HRE-1008566"]

class Funova_Batch4_C9orf72_HRE_981344_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["C9orf72-HRE-981344"]

class Funova_Batch4_TDP_43_G348V_1057052_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["TDP--43-G348V-1057052"]

class Funova_Batch4_TDP_43_N390D_1005373_Config(FigureConfigToUse):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["TDP--43-N390D-1005373"]

# class FunovaUMAP2FigureConfig_DAPI(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['DAPI']

# class FunovaUMAP2FigureConfig_Stress_initiation(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Stress-initiation']

# class FunovaUMAP2FigureConfig_mature_Autophagosome(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['mature-Autophagosome']

# class FunovaUMAP2FigureConfig_Cytoskeleton(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Cytoskeleton']

# class FunovaUMAP2FigureConfig_Ubiquitin_levels(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Ubiquitin-levels']

# class FunovaUMAP2FigureConfig_UPR_IRE1a(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['UPR-IRE1a']

# class FunovaUMAP2FigureConfig_UPR_ATF4(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['UPR-ATF4']

# class FunovaUMAP2FigureConfig_UPR_ATF6(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['UPR-ATF6']

# class FunovaUMAP2FigureConfig_impaired_Autophagosome(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['impaired-Autophagosome']

# class FunovaUMAP2FigureConfig_Autophagy(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Autophagy']

# class FunovaUMAP2FigureConfig_Aberrant_splicing(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Aberrant-splicing']

# class FunovaUMAP2FigureConfig_Parthanatos_late(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Parthanatos-late']

# class FunovaUMAP2FigureConfig_Nuclear_speckles_SC35(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Nuclear-speckles-SC35']

# class FunovaUMAP2FigureConfig_Splicing_factories(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Splicing-factories']

# class FunovaUMAP2FigureConfig_TDP_43(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['TDP-43']

# class FunovaUMAP2FigureConfig_Nuclear_speckles_SON(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Nuclear-speckles-SON']

# class FunovaUMAP2FigureConfig_DNA_damage_pH2Ax(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['DNA-damage-pH2Ax']

# class FunovaUMAP2FigureConfig_Parthanatos_early(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Parthanatos-early']

# class FunovaUMAP2FigureConfig_Necrosis(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Necrosis']

# class FunovaUMAP2FigureConfig_Necroptosis_HMGB1(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Necroptosis-HMGB1']

# class FunovaUMAP2FigureConfig_Neuronal_activity(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Neuronal-activity']

# class FunovaUMAP2FigureConfig_DNA_damage_P53BP1(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['DNA-damage-P53BP1']

# class FunovaUMAP2FigureConfig_Apoptosis(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Apoptosis']

# class FunovaUMAP2FigureConfig_Necroptosis_pMLKL(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Necroptosis-pMLKL']

# class FunovaUMAP2FigureConfig_Protein_degradation(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Protein-degradation']

# class FunovaUMAP2FigureConfig_Senescence_signaling(FunovaUMAP2FigureConfig):
#     def __init__(self):
#         super().__init__()
#         self.MARKERS = ['Senescence-signaling']

class FunovaUMAP2FigureConfigPROTEOSTASIS_MARKERS(FunovaUMAP2FigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS = PROTEOSTASIS_MARKERS

class FunovaUMAP2FigureConfigNEURONAL_CELL_DEATH_SENESCENCE_MARKERS(FunovaUMAP2FigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS = NEURONAL_CELL_DEATH_SENESCENCE_MARKERS

class FunovaUMAP2FigureConfigSYNAPTIC_NEURONAL_FUNCTION_MARKERS(FunovaUMAP2FigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS = SYNAPTIC_NEURONAL_FUNCTION_MARKERS

class FunovaUMAP2FigureConfigDNA_RNA_DEFECTS_MARKERS(FunovaUMAP2FigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS = DNA_RNA_DEFECTS_MARKERS