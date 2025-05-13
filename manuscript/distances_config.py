import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.distances.distances_config import DistanceConfig
from manuscript.plot_config import PlotConfig

class NeuronsDistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True        
        self.MARKERS_TO_EXCLUDE = [ 'TIA1']

class NeuronsDistanceWithBioReplicateConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(4,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True        
        self.MARKERS_TO_EXCLUDE = ['TIA1']

class NeuronsTBK1DistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'   
        self.CELL_LINES = ['FUSHomozygous','FUSHeterozygous','FUSRevertant', 'OPTN','TBK1','TDP43']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True   
        self.BASELINE_CELL_LINE_CONDITION = "TBK1_Untreated"
        
class dNLS345DistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(3,6)]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "TDP43_Untreated"
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True


class Day18DistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True         
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"

class AlyssaCoyneDistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_7tiles'    
        self.BASELINE_CELL_LINE_CONDITION = "Controls_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.RANDOM_SPLIT_BASELINE = False
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        self.REPS = [f'rep{i}' for i in range(1,11)]   

class NIH8DaysDistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1", "batch2", "batch3"]
        
        self.EXPERIMENT_TYPE = 'NIH_d8'   
        self.MARKERS_TO_EXCLUDE = [] 
        self.CELL_LINES = ['WT']
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class NIH8DaysTBK1DistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1", "batch2", "batch3"]
        
        self.EXPERIMENT_TYPE = 'NIH_d8'   
        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = []
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True   
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"

class funovaTBK1DistanceConfig(DistanceConfig):
    ## Bubble plot - Compare to one cell line (patient)
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["Batch1", "Batch2", "Batch3", "Batch4"]
        
        self.EXPERIMENT_TYPE = 'funova'   
        self.CELL_LINES = ["Control-1001733", "Control-1017118", "Control-1025045", "Control-1048087",
                           'C9orf72-HRE-1008566', 'C9orf72-HRE-981344', 'TDP--43-G348V-1057052', 'TDP--43-N390D-1005373']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = []
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True   
        self.BASELINE_CELL_LINE_CONDITION =  "Control-1025045_Untreated"
        self.COMMON_BASELINE = "Control-1025045"

class funovaTBK1DistanceConfigControl(DistanceConfig):
    ## Bubble plot - Compare to all controls
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["Batch1", "Batch2", "Batch3", "Batch4"]
        
        self.EXPERIMENT_TYPE = 'funova'   
        self.CELL_LINES = ["Control-1001733", "Control-1017118", "Control-1025045", "Control-1048087",
                           'C9orf72-HRE-1008566', 'C9orf72-HRE-981344', 'TDP--43-G348V-1057052', 'TDP--43-N390D-1005373']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = []
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True   
        self.BASELINE_CELL_LINE_CONDITION =  "Control_Untreated"
        self.COMMON_BASELINE = "Control"


class funovaTBK1DistanceConfigControlStress(funovaTBK1DistanceConfigControl):
    ## Bubble plot - stress
    def __init__(self):
        super().__init__()
        self.CONDITIONS = ['stress']
        self.BASELINE_CELL_LINE_CONDITION =  "Control_stress"

class funovaTBK1DistanceConfigControlUntreatedStress(funovaTBK1DistanceConfigControl):
    ## Bubble plot untreated + stress
    def __init__(self):
        super().__init__()
        self.CONDITIONS = ['Untreated', 'stress']

class funovaTBK1DistanceConfigControlNoBatch1(funovaTBK1DistanceConfigControl):
    ## Bubble plot - without batch 1 - for finetuned model
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["Batch2", "Batch3", "Batch4"]

class funovaDistanceConfig(DistanceConfig):
    ## Boxplot
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["Batch1", "Batch2", "Batch3", "Batch4"]
        
        self.EXPERIMENT_TYPE = 'funova'   
        self.MARKERS_TO_EXCLUDE = [] 
        self.BASELINE_CELL_LINE_CONDITION = "Control_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.COMMON_BASELINE = "Control"

class funovaDistanceConfigTDP43(DistanceConfig):
    ## Boxplot - TDP43 cell lines
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["Batch1", "Batch2", "Batch3", "Batch4"]
        
        self.EXPERIMENT_TYPE = 'funova'   
        self.MARKERS_TO_EXCLUDE = [] 
        self.CELL_LINES = ['TDP--43']
        self.BASELINE_CELL_LINE_CONDITION = "TDP--43_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.COMMON_BASELINE = "TDP--43"

class funovaDistanceConfigC9(DistanceConfig):
    ## Boxplot - C9 cell lines
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["Batch1", "Batch2", "Batch3", "Batch4"]
        
        self.EXPERIMENT_TYPE = 'funova'   
        self.MARKERS_TO_EXCLUDE = [] 
        self.CELL_LINES = ['C9orf72-HRE']
        self.BASELINE_CELL_LINE_CONDITION = "C9orf72-HRE_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.COMMON_BASELINE = "C9orf72-HRE"

### Finetuned

class funovaTBK1DistanceConfigControlFinetuned(DistanceConfig):
    ## Bubble plot - Batch 1 is out + we compare to each of the controls
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["Batch2", "Batch3", "Batch4"]
        
        self.EXPERIMENT_TYPE = 'funova'   
        self.CELL_LINES = ["Control-1001733", "Control-1017118", "Control-1025045", "Control-1048087",
                           'C9orf72-HRE-1008566', 'C9orf72-HRE-981344', 'TDP--43-G348V-1057052', 'TDP--43-N390D-1005373']
        self.CONDITIONS = ['Untreated', 'stress']
        self.MARKERS_TO_EXCLUDE = []
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True   
        self.BASELINE_CELL_LINE_CONDITION =  "Control_Untreated"
        # self.COMMON_BASELINE = "Control"

class funovaTBK1DistanceConfigControl1(funovaTBK1DistanceConfigControlFinetuned):
    ## Bubble plot
    def __init__(self):
        super().__init__()
        self.BASELINE_CELL_LINE_CONDITION =  "Control-1001733_Untreated"

class funovaTBK1DistanceConfigControl2(funovaTBK1DistanceConfigControlFinetuned):
    ## Bubble plot
    def __init__(self):
        super().__init__()
        self.BASELINE_CELL_LINE_CONDITION =  "Control-1017118_Untreated"

class funovaTBK1DistanceConfigControl3(funovaTBK1DistanceConfigControlFinetuned):
    ## Bubble plot
    def __init__(self):
        super().__init__()
        self.BASELINE_CELL_LINE_CONDITION =  "Control-1025045_Untreated"

class funovaTBK1DistanceConfigControl4(funovaTBK1DistanceConfigControlFinetuned):
    ## Bubble plot
    def __init__(self):
        super().__init__()
        self.BASELINE_CELL_LINE_CONDITION =  "Control-1048087_Untreated"

class funovaTBK1DistanceConfigControlStressFinetuned(funovaTBK1DistanceConfigControlFinetuned):
    ## Bubble plot stress
    def __init__(self):
        super().__init__()
        self.CONDITIONS = ['stress']
        self.BASELINE_CELL_LINE_CONDITION =  "Control_stress"