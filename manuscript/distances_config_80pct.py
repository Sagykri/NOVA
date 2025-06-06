import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.distances.distances_config import DistanceConfig
from manuscript.plot_config import PlotConfig

class NeuronsDistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in range(6,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True        
        self.MARKERS_TO_EXCLUDE = [ 'TIA1']

class NeuronsDistanceWithBioReplicateConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in range(4,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True        
        self.MARKERS_TO_EXCLUDE = ['TIA1']

class NeuronsTBK1DistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in range(6,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'   
        self.CELL_LINES = ['FUSHomozygous','FUSHeterozygous','FUSRevertant', 'OPTN','TBK1','TDP43']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True   
        self.BASELINE_CELL_LINE_CONDITION = "TBK1_Untreated"
        
class DNLS345DistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS80pct", f) for f in 
                        [f"batch{i}" for i in range(3,6)]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "TDP43_Untreated"
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True


class Day18DistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged80pct", f) for f in 
                        ["batch1", "batch2"]]
        
        self.EXPERIMENT_TYPE = 'neurons_d1880pct'    
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True         
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"

class AlyssaCoyneDistanceConfig(DistanceConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_7tiles80pct'    
        self.BASELINE_CELL_LINE_CONDITION = "Controls_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.RANDOM_SPLIT_BASELINE = False
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        self.REPS = [f'rep{i}' for i in range(1,11)]   
