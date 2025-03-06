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
    ## Bubble plot
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["Batch1", "Batch2", "Batch3", "Batch4"]
        
        self.EXPERIMENT_TYPE = 'funova'   
        self.CELL_LINES = ["Control-1001733", "Control-1017118", "Control-1025045", "Control-1048087",
                           'C9orf72-HRE-1008566', 'C9orf72-HRE-981344', 'TDP--43-G348V-1057052', 'TDP--43-N390D-1005373']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = []
        # self.MARKERS = []
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True   
        self.BASELINE_CELL_LINE_CONDITION =  "Control-1025045_Untreated"
        self.COMMON_BASELINE = "Control-1025045"

class funovaDistanceConfig(DistanceConfig):
    ## Boxplot
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