import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import colorcet as cc
import seaborn as sns

from src.common.configs.dataset_config import DatasetConfig

class ExampleFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS = ['G3BP1', 'DAPI', 'Phalloidin', 'DCP1A']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
                                 'OPTN_Untreated', 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
                                 'FUSRevertant_Untreated', 'TBK1_Untreated']
        # Otherwise:
        self.CELL_LINES = ['WT', 'TDP43']
        self.CONDITIONS = ['Untreated']
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Output folder:
        self.FIGURE_OUTPUT_FOLDER = os.path.join(self.OUTPUTS_FOLDER, 'figures', 'manuscript', 'fig2', 'panelB')

        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7


############################################################
# Figure 1 - U2OS data - Stress
############################################################        

#PanelA
class U2OSUMAP0StressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        """
        Figure 1A (G3BP1) and Supplementary Figure 2 ('DAPI', 'Phalloidin', 'DCP1A')
        UMAP0 stress, all markers        
        """        
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                        ["U2OS_spd_format"]]
        
        self.SPLIT_DATA = False        
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'U2OS'
        self.CELL_LINES = ['U2OS']
        self.MARKERS = ['G3BP1']#, 'DAPI', 'Phalloidin', 'DCP1A']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' 
        
        # Set a function to map the labels, can be None if not needed.
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION


        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

# nancy to check if in vqindhist1 we use embeddings    
class U2OSFeatureSpectraStressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        """
        Figure 1A (G3BP1) and Supplementary Figure 2 ('DAPI', 'Phalloidin', 'DCP1A')
        Feature Spectra stress, all markers        

        """        
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                        ["U2OS_spd_format"]]
        
        self.SPLIT_DATA = False        
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'U2OS'
        self.CELL_LINES = ['U2OS']
        self.MARKERS = ['G3BP1', 'DAPI', 'Phalloidin', 'DCP1A']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist1' 
        
        # Set a function to map the labels, can be None if not needed.
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}

        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

############################################################
# Figure 1 - Neurons - Batch6 - rep2 - Stress
############################################################     
#PanelC   
class NeuronsUMAP0B6StressFigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        """
        Figure 1E: WTG3BP1 UMAP0 visualization of  using latent feature representation extracted with Cytoself.
        Figure 1F: G3BP1 UMAP0 visualization of G3BP1 using latent feature representation extracted with Neuroself.
        """
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CELL_LINES = ['WT']
        self.REPS = ['rep2'] 
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS = ['G3BP1']
        
        self.EXPERIMENT_TYPE = 'neurons'
        
        self.SPLIT_DATA = False
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
        
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' 
        
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}
        
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION


        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################        

############################################################
# Figure 2 Neurons - WT - Organellomics
############################################################        

class NeuronsUMAP1B78FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """Figure 2B: UMAP1 of WT untreated - testset of B7-8
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]
        
        # Take only test set of B7+8
        self.SPLIT_DATA = True 
        self.EXPERIMENT_TYPE = 'neurons'    
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist1' 

        # UMAP1 vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_MARKERS
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']

        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

class NeuronsUMAP2B78FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """Figure ?: UMAP2 - batch 6 both reps
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # UMAP1 vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
                                 'OPTN_Untreated', 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
                                 'FUSRevertant_Untreated', 'TBK1_Untreated']
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']

        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

class NeuronsUMAP2B9BothRepsB78FigureConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        
        """Figure ?: UMAP2 - batch 9 both reps
        """

        # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False 
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # UMAP1 vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
                                 'OPTN_Untreated', 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
                                 'FUSRevertant_Untreated', 'TBK1_Untreated']
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']

        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

############################################################
# Figure 3
############################################################   

# Panel A
class Fig3AConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' 
                

        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated', 'stress']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[1] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION
        
        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
############################################################
# Figure 4
############################################################  

# Panel C
class Fig4CConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'deltaNLS'    
        self.REPS       = ['rep1', 'rep2']
        # self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        self.MARKERS = ['TDP43B']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' 
                

        # When using vqindhist:
        # self.CELL_LINES_CONDS = ['TDP43_Untreated', 'TDP43_dox']
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

############################################################
# Figure 5
############################################################     

# Panel C
class Fig5CB6Config(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
                                 'OPTN_Untreated', 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
                                 'FUSRevertant_Untreated', 'TBK1_Untreated']
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

# Panel D - UMAP0
class Fig5DB6Config(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS = ['FUS']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' 
                
        
        self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
class Fig5DB9Config(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS = ['FUS']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' 
                

        self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
# Panel E
class Fig5EB6Config(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
                                 'OPTN_Untreated', 'FUSHomozygous_Untreated',
                                 'TBK1_Untreated']
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
# Panel F
class Fig5FB6Config(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'FUSRevertant_Untreated',
                                 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
                                 ]
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
        
class Fig5FB6SCNAConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
                                 'OPTN_Untreated', 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
                                 'FUSRevertant_Untreated', 'TBK1_Untreated',
                                 'SCNA_Untreated']
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
class Fig5FB9SCNAConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
                            'OPTN_Untreated', 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
                            'FUSRevertant_Untreated', 'TBK1_Untreated',
                            'SCNA_Untreated']

        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

############################################################
# Sup Figure 2
############################################################    
class FigSup2AConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        """
        cytoself
        Supplementary Figure 2A ('DAPI', 'Phalloidin', 'DCP1A')
        UMAP0 stress,        
        """        
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                        ["U2OS_spd_format"]]
        
        self.SPLIT_DATA = False        
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'U2OS'
        self.CELL_LINES = ['U2OS']
        self.MARKERS = ['DAPI', 'Phalloidin', 'DCP1A']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' 
        
        # Set a function to map the labels, can be None if not needed.
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[-2-int(self.ADD_REP_TO_LABEL)] for l in labels])"

        # Set the colormap, for example: {"Untreated": "#52C5D5", 'stress': "#F7810F"} 
        self.COLORMAP = {"Untreated": "#52C5D5", 'stress': "#F7810F"}
        
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        #######################################

############################################################
# Figure Sup 4
############################################################     

class FigSup4AConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        # self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        self.MARKERS = ['PURA', 'G3BP1', 'PML']
        
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' 
                

        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated', 'stress']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[1] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION
        
        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

############################################################
# Figure Sup 5
############################################################     

# Panel A
# class FigSup5AConfig(DatasetConfig):
#     def __init__(self):
#         super().__init__()
#          # Batches used for model development
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
#                         ["batch9"]]
        
#         self.SPLIT_DATA = False 
#         self.ADD_REP_TO_LABEL = False
#         self.ADD_BATCH_TO_LABEL = False
        
#         self.EXPERIMENT_TYPE = 'neurons'    
#         self.REPS       = ['rep1', 'rep2']
#         self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        
#         # Local/Global embeddings
#         self.EMBEDDINGS_LAYER = 'vqvec2' 
                

#         self.CELL_LINES = ['WT']
#         self.CONDITIONS = ['Untreated', 'stress']
        
#         # How labels are shown in legend
#         self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[1] for l in labels])"
        
#         # Colors 
#         self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION
        
#         # Set the size of the dots
#         self.SIZE = 30
#         # Set the alpha of the dots (0=max opacity, 1=no opacity)
#         self.ALPHA = 0.7
        


class FigSup5AConfig(DatasetConfig):
    """UMAP0, deltaNLS

    Args:
        DatasetConfig (_type_): _description_
    """
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'deltaNLS'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS = ['DCP1A', 'TDP43B']
        
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqvec2' 
        self.CELL_LINES = ['TDP43']

        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[1] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX
        
        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

        
############################################################
# Figure Sup 6
############################################################   

# PanelA
# class FigSup6AConfig(DatasetConfig):
#     def __init__(self):
#         super().__init__()
#          # Batches used for model development
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
#                         ["batch9"]]
        
#         self.SPLIT_DATA = False 
#         self.ADD_REP_TO_LABEL = False
#         self.ADD_BATCH_TO_LABEL = False
        
#         self.EXPERIMENT_TYPE = 'neurons'    
#         self.REPS       = ['rep1', 'rep2']
#         self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
#         # Local/Global embeddings
#         self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

#         # When using vqindhist:
#         self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
#                                  'OPTN_Untreated', 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
#                                  'FUSRevertant_Untreated', 'TBK1_Untreated']
        
#         # How labels are shown in legend
#         # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
#         # Colors 
#         self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
#         # Set the size of the dots
#         self.SIZE = 0.3
#         # Set the alpha of the dots (0=max opacity, 1=no opacity)
#         self.ALPHA = 0.7

class FigSup6AConfig(DatasetConfig):
    """SM

    Args:
        DatasetConfig (_type_): _description_
    """
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
                                 'OPTN_Untreated', 'FUSHomozygous_Untreated', 'TBK1_Untreated']
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

# PanelB
class FigSup6BB6Config(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1', 'FUS']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'FUSRevertant_Untreated',
                                 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
                                 ]
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
class FigSup6BB9Config(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1', 'FUS']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'FUSRevertant_Untreated',
                                 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
                                 ]
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
class FigSup6CConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['WT_Untreated', 'WT_stress']
        
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_CONDITION
        
        # Set the size of the dots
        self.SIZE = 0.3
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
# PanelD
class FigSup6DConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
         # Batches used for model development
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.SPLIT_DATA = False 
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False
        
        self.EXPERIMENT_TYPE = 'deltaNLS'    
        self.REPS       = ['rep1', 'rep2']
        self.MARKERS = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','ANXA11','Calreticulin','mitotracker',
                        'KIF5A','TDP43B','TDP43N','CLTC','DCP1A','TOMM20','FUS','SCNA','LAMP1','PML',
                        'PURA','CD41','Phalloidin', 'PEX14']
        
        # Local/Global embeddings
        self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

        # When using vqindhist:
        self.CELL_LINES_CONDS = ['TDP43_Untreated', 'TDP43_dox']
        
        # How labels are shown in legend
        # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
        # Colors 
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_DOX
        
        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
# ############################################################
# # Figure Sup 7
# ############################################################     

# # Panel B
# class FigSup7BB6Config(DatasetConfig):
#     def __init__(self):
#         super().__init__()
#          # Batches used for model development
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
#                         ["batch6"]]
        
#         self.SPLIT_DATA = False 
#         self.ADD_REP_TO_LABEL = False
#         self.ADD_BATCH_TO_LABEL = False
        
#         self.EXPERIMENT_TYPE = 'neurons'    
#         self.REPS       = ['rep1', 'rep2']
#         self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1', 'FUS']
        
#         # Local/Global embeddings
#         self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

#         # When using vqindhist:
#         self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
#                             'OPTN_Untreated', 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
#                             'FUSRevertant_Untreated', 'TBK1_Untreated']

        
#         # How labels are shown in legend
#         # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
#         # Colors 
#         self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
#         # Set the size of the dots
#         self.SIZE = 0.3
#         # Set the alpha of the dots (0=max opacity, 1=no opacity)
#         self.ALPHA = 0.7
        
# class FigSup7BB9Config(DatasetConfig):
#     def __init__(self):
#         super().__init__()
#          # Batches used for model development
#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
#                         ["batch9"]]
        
#         self.SPLIT_DATA = False 
#         self.ADD_REP_TO_LABEL = False
#         self.ADD_BATCH_TO_LABEL = False
        
#         self.EXPERIMENT_TYPE = 'neurons'    
#         self.REPS       = ['rep1', 'rep2']
#         self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1', 'FUS']
        
#         # Local/Global embeddings
#         self.EMBEDDINGS_LAYER = 'vqindhist2' 
                

#         # When using vqindhist:
#         self.CELL_LINES_CONDS = ['WT_Untreated', 'TDP43_Untreated',
#                             'OPTN_Untreated', 'FUSHomozygous_Untreated', 'FUSHeterozygous_Untreated',
#                             'FUSRevertant_Untreated', 'TBK1_Untreated']

        
#         # How labels are shown in legend
#         # self.MAP_LABELS_FUNCTION = "lambda self: lambda labels: __import__('numpy').asarray([l.split('_')[0] for l in labels])"
        
#         # Colors 
#         self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
#         # Set the size of the dots
#         self.SIZE = 0.3
#         # Set the alpha of the dots (0=max opacity, 1=no opacity)
#         self.ALPHA = 0.7
        

