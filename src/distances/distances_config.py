import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.plot_config import PlotConfig

class NeuronsDistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'    
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True        
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']

class NeuronsTBK1DistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons'   
        self.CELL_LINES = ['FUSHomozygous','FUSHeterozygous','FUSRevertant', 'OPTN','TBK1','TDP43']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['FMRP', 'TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True   
        self.BASELINE_CELL_LINE_CONDITION = "TBK1_Untreated"
        
class dNLS345DistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(3,6)]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['FMRP','TIA1']
        self.BASELINE_CELL_LINE_CONDITION = "TDP43_Untreated"
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True


class Day18DistanceConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.SETS = ['testset']
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.MARKERS_TO_EXCLUDE = None
        
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True         
        self.BASELINE_CELL_LINE_CONDITION = "WT_Untreated"