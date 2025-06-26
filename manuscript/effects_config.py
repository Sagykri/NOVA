import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.effects.effects_config import EffectConfig
from manuscript.plot_config import PlotConfig

class NeuronsEffectWithBioReplicateConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [f"batch{i}80pct" for i in [4,5,6,9]]
        
        self.EXPERIMENT_TYPE = 'neurons'    
        self.BASELINE_PERTURB = {'WT_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['FUSHomozygous','FUSHeterozygous', 'OPTN','TBK1','TDP43','FUSRevertant']] + ['WT_stress'],
                                'FUSRevertant_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['FUSHomozygous','FUSHeterozygous', 'OPTN','TBK1','TDP43','WT']]}       
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']

class dNLS345EffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [f"batch{i}" for i in range(3,6)]
        
        self.EXPERIMENT_TYPE = 'deltaNLS80pct'
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']
        self.BASELINE_PERTURB = {'TDP43_Untreated':['TDP43_dox'], 'WT_Untreated':['TDP43_dox']}
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())

class Day18EffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1", "batch2"]
        
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']       
        self.BASELINE_PERTURB = {'WT_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['FUSHomozygous','FUSHeterozygous','FUSRevertant', 'OPTN','TBK1','TDP43']]}

class AlyssaCoyneEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_7tiles'    
        self.BASELINE_PERTURB = {'WT_Untreated': [f'{cell_line}_Untreated' for cell_line 
                                in ['sALSPositiveCytoTDP43', 'sALSNegativeCytoTDP43','c9orf72ALSPatients']]}
        self.MARKERS_TO_EXCLUDE = ['MERGED']


class dNLSNewEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [f"batch{i}" for i in range(1,7)]
        self.EXPERIMENT_TYPE = 'deltaNLS_new'
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI'] #TODO: DAPI is taking too long so skipping for now
        self.BASELINE_PERTURB = {'dNLS_Untreated':['dNLS_DOX']}#, 'WT_Untreated':['dNLS_DOX']}
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
