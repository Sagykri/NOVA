import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.figures_config import FigureConfig
from manuscript.plot_config import PlotConfig
from src.datasets.label_utils import MapLabelsFunction

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

## Distances ##
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


class FunovaDistancesALSFigureConfigStress(FigureConfig):
    def __init__(self):
        """Bubbleplot of WT vs other cell lines - stress
        """
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2","Batch3", "Batch4"]]
        
        self.EXPERIMENT_TYPE = 'funova'    
        self.MARKERS_TO_EXCLUDE = []
        self.BASELINE_CELL_LINE_CONDITION = "Control_stress"
        self.CONDITIONS = ['stress']
        self.CELL_LINES_CONDITIONS = ["C9orf72-HRE-1008566_stress","C9orf72-HRE-981344_stress", 
                                      "TDP--43-G348V-1057052_stress","TDP--43-N390D-1005373_stress"]
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.COMMON_CELL_LINES.name

class FunovaDistancesALSFigureConfigAll(FigureConfig):
    def __init__(self):
        """Bubbleplot of WT vs other cell lines
        """
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2","Batch3", "Batch4"]]
        
        self.EXPERIMENT_TYPE = 'funova'    
        self.MARKERS_TO_EXCLUDE = []
        self.BASELINE_CELL_LINE_CONDITION = "Control_Untreated"
        self.CELL_LINES_CONDITIONS = ["Control_stress", "C9orf72-HRE-1008566_Untreated","C9orf72-HRE-1008566_stress","C9orf72-HRE-981344_Untreated", "C9orf72-HRE-981344_stress",
                                      "TDP--43-G348V-1057052_Untreated","TDP--43-G348V-1057052_stress","TDP--43-N390D-1005373_Untreated","TDP--43-N390D-1005373_stress"]
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.COMMON_CELL_LINES.name

class FunovaDistancesALSFigureConfigNoBatch1(FunovaDistancesALSFigureConfigAll):
    def __init__(self):
        """Bubbleplot of WT vs other cell lines
        """
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2","Batch3", "Batch4"]]
        self.BASELINE_CELL_LINE_CONDITION = "Control-1025045_Untreated"
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class FunovaDistancesALSFigureConfigFinetuned(FunovaDistancesALSFigureConfigAll):
    def __init__(self):
        """Bubbleplot of WT vs other cell lines
        """
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2","Batch3", "Batch4"]]

class FunovaDistancesALSFigureConfigAllBut1001733(FigureConfig):
    def __init__(self):
        """Bubbleplot of WT vs other cell lines wo 1001733
        """
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2","Batch3", "Batch4"]]
        
        self.EXPERIMENT_TYPE = 'funova'    
        self.MARKERS_TO_EXCLUDE = []
        self.BASELINE_CELL_LINE_CONDITION = "Control_Untreated"
        self.CELL_LINES = ["Control-1017118", "Control-1025045", "Control-1048087",
                           'C9orf72-HRE-1008566', 'C9orf72-HRE-981344', 'TDP--43-G348V-1057052', 
                           'TDP--43-N390D-1005373']
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
        self.BASELINE_CELL_LINE_CONDITION = "Control_Untreated"
        self.CELL_LINES_CONDITIONS = ['Control_stress']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.COMMON_CELL_LINES.name

class FunovaDistancesStressFigureConfigTDP43(FigureConfig):
    def __init__(self):
        super().__init__()
        """Boxplot of TDP43 stress vs untreated
        """
        self.INPUT_FOLDERS =  [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2","Batch3", "Batch4"]]
        
        self.EXPERIMENT_TYPE = 'funova'
        self.MARKERS_TO_EXCLUDE = []
        self.BASELINE_CELL_LINE_CONDITION = "TDP--43_Untreated"
        self.CELL_LINES_CONDITIONS = ['TDP--43_stress']
        self.CELL_LINES = ['TDP--43']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.COMMON_CELL_LINES.name

class FunovaDistancesStressFigureConfigC9(FigureConfig):
    def __init__(self):
        super().__init__()
        """Boxplot of C9 stress vs untreated
        """
        self.INPUT_FOLDERS =  [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2","Batch3", "Batch4"]]
        
        self.EXPERIMENT_TYPE = 'funova'
        self.MARKERS_TO_EXCLUDE = []
        self.BASELINE_CELL_LINE_CONDITION = "C9orf72-HRE_Untreated"
        self.CELL_LINES_CONDITIONS = ['C9orf72-HRE_stress']
        self.CELL_LINES = ['C9orf72-HRE']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.COMMON_CELL_LINES.name

## Batches ##
class Funova_Batch1_Config(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]

class Funova_Batch2_Config(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]

class Funova_Batch3_Config(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]

class Funova_Batch4_Config(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]

class NeuronsUMAP0StressFunovaFigureConfigMinMax(FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - WT untreated vs stress
        """        
        self.PROCESSED_FOLDER_ROOT = '/home/labs/hornsteinlab/Collaboration/FUNOVA/input/images/processed_minmax/'
        self.EXPERIMENT_TYPE = 'funova_minmax'    
        self.MARKERS_TO_EXCLUDE = []
        self.SHOW_ARI = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.CONDITIONS.name
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]

class NeuronsUMAP0StressFunovaFigureConfigBatches234(FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - WT untreated vs stress
        """        
        self.EXPERIMENT_TYPE = 'funova'    
        self.MARKERS_TO_EXCLUDE = []
        self.SHOW_ARI = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.CONDITIONS.name
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2","Batch3", "Batch4"]]

#### UMAP0 by batch #####

config_to_use = Funova_Batch3_Config ## Change for each batch

class NeuronsUMAP0StressFunovaFigureConfig(config_to_use):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - WT untreated vs stress
        """        
        self.EXPERIMENT_TYPE = 'funova'    
        self.MARKERS_TO_EXCLUDE = []
        self.SHOW_ARI = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.CONDITIONS.name

class FunovaUMAP0CellLinesConditionsFigureConfig(NeuronsUMAP0StressFunovaFigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - Cell lines 
        """        
        self.ADD_LINE_TO_LABEL = True
        self.ADD_REP_TO_LABEL=False   
        # self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES_CONDITIONS.name
        self.SHOW_ARI = False

class FunovaUMAP0CellLinesFigureConfig(NeuronsUMAP0StressFunovaFigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - Cell lines 
        """        
        self.ADD_LINE_TO_LABEL = True
        self.ADD_REP_TO_LABEL=False   
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class FunovaUMAP0CellLinesStressFigureConfig(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - Cell lines + stress
        """        
        self.CONDITIONS = ['stress'] 

class FunovaUMAP0CellLinesUntreatedFigureConfig(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP0 of single markers - Cell lines - stress
        """        
        self.CONDITIONS = ['Untreated'] 

class Funova_controls_tdp_untreated(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines + tdp43_cell_lines
        self.CONDITIONS = ['Untreated']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_tdp_stress(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines + tdp43_cell_lines
        self.CONDITIONS = ['stress']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_c9_untreated(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines + c9orf72_cell_lines
        self.CONDITIONS = ['Untreated']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_c9_stress(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines + c9orf72_cell_lines
        self.CONDITIONS = ['stress']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_stress(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CONDITIONS = ['stress']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_untreated(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CONDITIONS = ['Untreated']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_untreated_si(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines 
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['Stress-initiation']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_controls_stress_si(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines
        self.CONDITIONS = ['stress'] 
        self.MARKERS = ['Stress-initiation']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_tdp_untreated_si(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = tdp43_cell_lines 
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['Stress-initiation']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_tdp_stress_si(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = tdp43_cell_lines
        self.CONDITIONS = ['stress'] 
        self.MARKERS = ['Stress-initiation']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_c9_66_DAPI(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ["C9orf72-HRE-981344"] + control_cell_lines
        self.CONDITIONS = ['Untreated'] 
        self.MARKERS = ['DAPI']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name
        self.COMMON_BASELINE = "Control"

class Funova_c9_66_DAPI0(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ["C9orf72-HRE-1008566"] + [control_cell_lines[0]]
        self.CONDITIONS = ['Untreated'] 
        self.MARKERS = ['DAPI']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_c9_66_DAPI1(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ["C9orf72-HRE-1008566"] + [control_cell_lines[1]]
        self.CONDITIONS = ['Untreated'] 
        self.MARKERS = ['DAPI']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_c9_66_DAPI2(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ["C9orf72-HRE-1008566"] + [control_cell_lines[2]]
        self.CONDITIONS = ['Untreated'] 
        self.MARKERS = ['DAPI']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class Funova_c9_66_DAPI3(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ["C9orf72-HRE-1008566"] + [control_cell_lines[3]]
        self.CONDITIONS = ['Untreated'] 
        self.MARKERS = ['DAPI']
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name                


## UMAP 0 all batches

class Funova_control_untreated_tdp(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['TDP-43']
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1","Batch2","Batch3","Batch4",]]
        
class Funova_control_stress_tdp(FunovaUMAP0CellLinesFigureConfig):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = control_cell_lines
        self.CONDITIONS = ['stress']
        self.MARKERS = ['TDP-43']
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1","Batch2","Batch3","Batch4",]]


## UMAP1 ###        
class UMAP1_FunovaFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """        
        self.EXPERIMENT_TYPE = 'funova'  
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = []
        self.SHOW_ARI = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.MARKERS.name

class UMAP1_FunovaFigureConfig_No_DAPI(UMAP1_FunovaFigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """        
        self.MARKERS_TO_EXCLUDE = ['DAPI']

class UMAP1_FunovaFigureConfig_stress_No_DAPI(UMAP1_FunovaFigureConfig_No_DAPI):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """        
        self.CONDITIONS = ['stress']

class UMAP1_FunovaFigureConfigConditions(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """        
        self.EXPERIMENT_TYPE = 'funova'  
        self.CONDITIONS = ['stress']
        self.MARKERS_TO_EXCLUDE = ['DAPI']
        self.SHOW_ARI = False
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["Control-1001733"]
        


## UMAP2 ##
config_to_use2 = Funova_Batch2_Config ## Change for each batch

class FunovaUMAP2FigureConfig(config_to_use2):
    def __init__(self):
        super().__init__()
        """UMAP2 multiplex of WT untreated vs stress
        """
        self.EXPERIMENT_TYPE = 'funova'    
        # self.CELL_LINES = []
        self.MARKERS_TO_EXCLUDE = []
        self.SHOW_ARI = True
        self.ADD_REP_TO_LABEL=False
        self.ADD_LINE_TO_LABEL = True
        self.ADD_CONDITION_TO_LABEL = True
        self.ARI_LABELS_FUNC = MapLabelsFunction.CONDITIONS.name

class FunovaUMAP2FigureConfig_CellLines(FunovaUMAP2FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP2 multiplex of WT untreated vs stress - by cell lines
        """
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES.name

class FunovaUMAP2FigureConfig_CellLinesConditions(FunovaUMAP2FigureConfig):
    def __init__(self):
        super().__init__()
        """UMAP2 multiplex of WT untreated vs stress - by cell lines and conditions
        """
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES_CONDITIONS.name

class FunovaUMAP2FigureConfig_CellLines_untreated(FunovaUMAP2FigureConfig_CellLines):
    def __init__(self):
        super().__init__()
        """UMAP2 multiplex of WT untreated - by cell lines
        """
        self.CONDITIONS = ['Untreated']    

class FunovaUMAP2FigureConfig_CellLines_stress(FunovaUMAP2FigureConfig_CellLines):
    def __init__(self):
        super().__init__()
        """UMAP2 multiplex of WT stress - by cell lines
        """
        self.CONDITIONS = ['stress']  

class FunovaUMAP2FigureConfigPROTEOSTASIS_MARKERS(FunovaUMAP2FigureConfig_CellLinesConditions):
    def __init__(self):
        super().__init__()
        self.MARKERS = PROTEOSTASIS_MARKERS

class FunovaUMAP2FigureConfigNEURONAL_CELL_DEATH_SENESCENCE_MARKERS(FunovaUMAP2FigureConfig_CellLinesConditions):
    def __init__(self):
        super().__init__()
        self.MARKERS = NEURONAL_CELL_DEATH_SENESCENCE_MARKERS

class FunovaUMAP2FigureConfigSYNAPTIC_NEURONAL_FUNCTION_MARKERS(FunovaUMAP2FigureConfig_CellLinesConditions):
    def __init__(self):
        super().__init__()
        self.MARKERS = SYNAPTIC_NEURONAL_FUNCTION_MARKERS

class FunovaUMAP2FigureConfigDNA_RNA_DEFECTS_MARKERS(FunovaUMAP2FigureConfig_CellLinesConditions):
    def __init__(self):
        super().__init__()
        self.MARKERS = DNA_RNA_DEFECTS_MARKERS

class FunovaUMAP2FigureConfigPROTEOSTASIS_MARKERS_untreated(FunovaUMAP2FigureConfig_CellLines_untreated):
    def __init__(self):
        super().__init__()
        self.MARKERS = PROTEOSTASIS_MARKERS

class FunovaUMAP2FigureConfigNEURONAL_CELL_DEATH_SENESCENCE_MARKERS_untreated(FunovaUMAP2FigureConfig_CellLines_untreated):
    def __init__(self):
        super().__init__()
        self.MARKERS = NEURONAL_CELL_DEATH_SENESCENCE_MARKERS

class FunovaUMAP2FigureConfigSYNAPTIC_NEURONAL_FUNCTION_MARKERS_untreated(FunovaUMAP2FigureConfig_CellLines_untreated):
    def __init__(self):
        super().__init__()
        self.MARKERS = SYNAPTIC_NEURONAL_FUNCTION_MARKERS

class FunovaUMAP2FigureConfigDNA_RNA_DEFECTS_MARKERS_untreated(FunovaUMAP2FigureConfig_CellLines_untreated):
    def __init__(self):
        super().__init__()
        self.MARKERS = DNA_RNA_DEFECTS_MARKERS

class FunovaUMAP2FigureConfigPROTEOSTASIS_MARKERS_stress(FunovaUMAP2FigureConfig_CellLines_stress):
    def __init__(self):
        super().__init__()
        self.MARKERS = PROTEOSTASIS_MARKERS

class FunovaUMAP2FigureConfigNEURONAL_CELL_DEATH_SENESCENCE_MARKERS_stress(FunovaUMAP2FigureConfig_CellLines_stress):
    def __init__(self):
        super().__init__()
        self.MARKERS = NEURONAL_CELL_DEATH_SENESCENCE_MARKERS

class FunovaUMAP2FigureConfigSYNAPTIC_NEURONAL_FUNCTION_MARKERS_stress(FunovaUMAP2FigureConfig_CellLines_stress):
    def __init__(self):
        super().__init__()
        self.MARKERS = SYNAPTIC_NEURONAL_FUNCTION_MARKERS

class FunovaUMAP2FigureConfigDNA_RNA_DEFECTS_MARKERS_stress(FunovaUMAP2FigureConfig_CellLines_stress):
    def __init__(self):
        super().__init__()
        self.MARKERS = DNA_RNA_DEFECTS_MARKERS

## All batches and cell line combinations:
## Choose the base config from: NeuronsUMAP0StressFunovaFigureConfig, UMAP1_FunovaFigureConfig, UMAP1_FunovaFigureConfig_No_DAPI, UMAP1_FunovaFigureConfig_stress_No_DAPI
config_to_use_cell_lines = UMAP1_FunovaFigureConfig_No_DAPI 
class Funova_Batch1_Control_1001733_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["Control-1001733"]

class Funova_Batch1_Control_1017118_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["Control-1017118"]

class Funova_Batch1_Control_1025045_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["Control-1025045"]

class Funova_Batch1_Control_1048087_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["Control-1048087"]

class Funova_Batch1_C9orf72_HRE_1008566_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["C9orf72-HRE-1008566"]

class Funova_Batch1_C9orf72_HRE_981344_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["C9orf72-HRE-981344"]

class Funova_Batch1_TDP_43_G348V_1057052_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["TDP--43-G348V-1057052"]

class Funova_Batch1_TDP_43_N390D_1005373_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch1")]
        self.CELL_LINES = ["TDP--43-N390D-1005373"]

class Funova_Batch2_Control_1001733_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["Control-1001733"]

class Funova_Batch2_Control_1017118_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["Control-1017118"]

class Funova_Batch2_Control_1025045_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["Control-1025045"]

class Funova_Batch2_Control_1048087_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["Control-1048087"]

class Funova_Batch2_C9orf72_HRE_1008566_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["C9orf72-HRE-1008566"]

class Funova_Batch2_C9orf72_HRE_981344_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["C9orf72-HRE-981344"]

class Funova_Batch2_TDP_43_G348V_1057052_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["TDP--43-G348V-1057052"]

class Funova_Batch2_TDP_43_N390D_1005373_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch2")]
        self.CELL_LINES = ["TDP--43-N390D-1005373"]

class Funova_Batch3_Control_1001733_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["Control-1001733"]

class Funova_Batch3_Control_1017118_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["Control-1017118"]

class Funova_Batch3_Control_1025045_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["Control-1025045"]

class Funova_Batch3_Control_1048087_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["Control-1048087"]

class Funova_Batch3_C9orf72_HRE_1008566_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["C9orf72-HRE-1008566"]

class Funova_Batch3_C9orf72_HRE_981344_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["C9orf72-HRE-981344"]

class Funova_Batch3_TDP_43_G348V_1057052_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["TDP--43-G348V-1057052"]

class Funova_Batch3_TDP_43_N390D_1005373_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch3")]
        self.CELL_LINES = ["TDP--43-N390D-1005373"]

class Funova_Batch4_Control_1001733_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["Control-1001733"]

class Funova_Batch4_Control_1017118_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["Control-1017118"]

class Funova_Batch4_Control_1025045_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["Control-1025045"]

class Funova_Batch4_Control_1048087_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["Control-1048087"]

class Funova_Batch4_C9orf72_HRE_1008566_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["C9orf72-HRE-1008566"]

class Funova_Batch4_C9orf72_HRE_981344_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["C9orf72-HRE-981344"]

class Funova_Batch4_TDP_43_G348V_1057052_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["TDP--43-G348V-1057052"]

class Funova_Batch4_TDP_43_N390D_1005373_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Batch4")]
        self.CELL_LINES = ["TDP--43-N390D-1005373"]

class Funova_Control_1001733_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2", "Batch3", "Batch4"]]
        self.CELL_LINES = ["Control-1001733"]

class Funova_Control_1017118_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2", "Batch3", "Batch4"]]
        self.CELL_LINES = ["Control-1017118"]

class Funova_Control_1025045_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2", "Batch3", "Batch4"]]
        self.CELL_LINES = ["Control-1025045"]

class Funova_Control_1048087_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2", "Batch3", "Batch4"]]
        self.CELL_LINES = ["Control-1048087"]

class Funova_C9orf72_HRE_1008566_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2", "Batch3", "Batch4"]]
        self.CELL_LINES = ["C9orf72-HRE-1008566"]

class Funova_C9orf72_HRE_981344_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2", "Batch3", "Batch4"]]
        self.CELL_LINES = ["C9orf72-HRE-981344"]

class Funova_TDP_43_G348V_1057052_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2", "Batch3", "Batch4"]]
        self.CELL_LINES = ["TDP--43-G348V-1057052"]

class Funova_TDP_43_N390D_1005373_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch2", "Batch3", "Batch4"]]
        self.CELL_LINES = ["TDP--43-N390D-1005373"]

class Funova_Batch3_Batch4_controls_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch3", "Batch4"]]
        self.CELL_LINES = control_cell_lines

class Funova_Batch4_all_cell_lines_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4"]]
        self.CELL_LINES = all_cell_lines

class Funova_Batch4_all_cell_lines_Stress_Config(UMAP1_FunovaFigureConfig_stress_No_DAPI):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4"]]
        self.CELL_LINES = all_cell_lines

class Funova_Batch4_controls_c91_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4"]]
        self.CELL_LINES = control_cell_lines + [c9orf72_cell_lines[0]]

class Funova_Batch4_controls_c92_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4"]]
        self.CELL_LINES = control_cell_lines + [c9orf72_cell_lines[1]]

class Funova_Batch4_controls_TDP1_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4"]]
        self.CELL_LINES = control_cell_lines + [tdp43_cell_lines[0]]

class Funova_Batch4_controls_TDP2_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4"]]
        self.CELL_LINES = control_cell_lines + [tdp43_cell_lines[1]]

class Funova_Batch4_c9s_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4"]]
        self.CELL_LINES = c9orf72_cell_lines

class Funova_Batch4_TDPs_Config(config_to_use_cell_lines):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch4"]]
        self.CELL_LINES = tdp43_cell_lines