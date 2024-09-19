import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.plot_config import PlotConfig

############################################################
# Figure 1
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
# Figure 1 - supp
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


############################################################
# Figure 2 
############################################################
class NeuronsUMAP0StressB6FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsDistancesStressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['WT_stress']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

############################################################
# Figure 2 - supp
############################################################
class NeuronsUMAP0StressB9FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class U2OSUMAP0StressDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                        ["U2OS_spd_format"]]
        
        self.SETS = ['testset']
        self.CELL_LINES = ['U2OS']
        self.EXPERIMENT_TYPE = 'U2OS'
        self.MARKERS = ['G3BP1', 'DCP1A', 'Phalloidin', 'DAPI']
        self.SHOW_ARI = True


############################################################
# Figure 3
############################################################
class dNLSUMAP0B3DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS = ['TDP43B','DCP1A']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
class dNLSUMAP0B4DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS = ['TDP43B','DCP1A']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
class dNLSUMAP0B5DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch5"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS = ['TDP43B','DCP1A']
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
class dNLSDistancesFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(3,6)]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
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
class NeuronsUMAP2StressB6FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP2B3FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'deltaNLS'    
        self.CELL_LINES = ['TDP43']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP2B4FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'deltaNLS'    
        self.CELL_LINES = ['TDP43']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class dNLSUMAP2B5FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch5"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'deltaNLS'    
        self.CELL_LINES = ['TDP43']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2ALSB6FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsDistancesALSFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.CELL_LINES_CONDITIONS = ['FUSHomozygous_Untreated','FUSHeterozygous_Untreated','FUSRevertant_Untreated',
                                      'TBK1_Untreated','OPTN_Untreated','TDP43_Untreated']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True


class NeuronsUMAP0ALSB6FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
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

############################################################
# Figure 5 - supp
############################################################
class NeuronsUMAP2StressB9FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CELL_LINES = ['WT']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2ALSB9FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2ALSB6_without_fus_marker_FigureConfig(NeuronsUMAP2ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1','FUS']

class NeuronsUMAP2ALSB6_without_SCNA_line_FigureConfig(NeuronsUMAP2ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','FUSHomozygous', 'FUSHeterozygous',
                           'TBK1','FUSRevertant','OPTN']
        
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
        
class NeuronsUMAP2ALSB6_without_fusrev_FigureConfig(NeuronsUMAP2ALSB6FigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT','TDP43','SCNA', 'FUSHomozygous',
                           'TBK1','FUSHeterozygous','OPTN']
        
class NeuronsUMAP2ALSD18B1FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False

class NeuronsUMAP2ALSD18B2FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False



############################################################
# experimental
############################################################
        
class dNLSUMAP1B3TDP43_UntreatedFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'deltaNLS'    
        self.CELL_LINES = ['TDP43']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())

        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False

class dNLSUMAP1B3TDP43_DOXFigureConfig(dNLSUMAP1B3TDP43_UntreatedFigureConfig):
    def __init__(self):
        super().__init__()
        self.CONDITIONS = ['dox']

class dNLSUMAP1B3WTFigureConfig(dNLSUMAP1B3TDP43_UntreatedFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT']

class NeuronsUMAP1D18B1FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False