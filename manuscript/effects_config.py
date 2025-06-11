import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.effects.effects_config import EffectConfig
from manuscript.plot_config import PlotConfig

class NeuronsEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}" for i in range(6,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.BASELINE_PERTURB = {'WT_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['FUSHomozygous','FUSHeterozygous', 'OPTN','TBK1','TDP43']],
                                'FUSRevertant_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['FUSHomozygous','FUSHeterozygous', 'OPTN','TBK1','TDP43']]}
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True        
        self.MARKERS_TO_EXCLUDE = [ 'TIA1']

class NeuronsEffectWithBioReplicateConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        [f"batch{i}80pct" for i in range(4,10)]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.BASELINE_PERTURB = {'WT_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['FUSHomozygous','FUSHeterozygous', 'OPTN','TBK1','TDP43']],
                                'FUSRevertant_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['FUSHomozygous','FUSHeterozygous', 'OPTN','TBK1','TDP43']]}
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True        
        self.MARKERS_TO_EXCLUDE = ['TIA1']

class NeuronsTBK1EffectConfig(EffectConfig):
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
        self.BASELINE_PERTURB = {"TBK1_Untreated":[f'{cell_line}_Untreated' for cell_line in 
                                                   ['FUSHomozygous','FUSHeterozygous','FUSRevertant', 'OPTN','TBK1','TDP43']]}
        
class dNLS345EffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk","deltaNLS", f) for f in 
                        [f"batch{i}" for i in range(3,6)]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.BASELINE_PERTURB = {'TDP43_Untreated':['TDP43_dox']}
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.CELL_LINES = ['TDP43']

class Day18EffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1", "batch2"]]
        
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True         
        self.BASELINE_PERTURB = {'WT_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['FUSHomozygous','FUSHeterozygous','FUSRevertant', 'OPTN','TBK1','TDP43']]}

class AlyssaCoyneEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_7tiles'    
        self.BASELINE_PERTURB = {'WT_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['sALSPositiveCytoTDP43', 'sALSNegativeCytoTDP43','c9orf72ALSPatients']]}
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.RANDOM_SPLIT_BASELINE = False
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        self.REPS = [f'rep{i}' for i in range(1,11)]   


class dNLSNewEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "OperadNLS_80pct", f) for f in 
                        [f"batch{i}" for i in range(1,7)]]
        
        self.EXPERIMENT_TYPE = 'deltaNLS_new'
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI'] #TODO: DAPI is taking too long so skipping for now
        self.BASELINE_PERTURB = {'dNLS_Untreated':['dNLS_DOX']}
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.CELL_LINES = ['dNLS']