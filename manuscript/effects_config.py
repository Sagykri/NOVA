import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.effects.effects_config import EffectConfig
from manuscript.plot_config import PlotConfig

# U2OS
class U2OSEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        self.EXPERIMENT_TYPE = 'U2OS'
        self.BASELINE = 'WT_Untreated'
        self.PERTURBATION = 'WT_stress'

        self.MIN_REQUIRED:int = 0
        self.N_BOOT:int = 1 # No need, only effect size is in use (no variance)

## neuronsDay8_new ##

class NeuronsEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS =  [f"batch{i}" for i in [1,2,3,7,8,9]]
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'  

class NeuronsEffectConfig_with_B10(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS =  [f"batch{i}" for i in [1,2,3,7,8,9,10]]
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    

class NeuronsEffectWTBaselineConfig(NeuronsEffectConfig):
    def __init__(self):
        super().__init__()
        self.BASELINE = 'WT_Untreated'

class NeuronsEffectWTBaselineFUSHomoConfig(NeuronsEffectWTBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'FUSHomozygous_Untreated'

class NeuronsEffectWTBaselineFUSHeteroConfig(NeuronsEffectWTBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'FUSHeterozygous_Untreated'

class NeuronsEffectWTBaselineFUSRevConfig(NeuronsEffectWTBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'FUSRevertant_Untreated'

class NeuronsEffectWTBaselineTDP43Config(NeuronsEffectWTBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'TDP43_Untreated'

class NeuronsEffectWTBaselineOPTNConfig(NeuronsEffectWTBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'OPTN_Untreated'

class NeuronsEffectWTBaselineTBK1Config(NeuronsEffectWTBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'TBK1_Untreated'

class NeuronsEffectWTBaselineSNCAConfig(NeuronsEffectWTBaselineConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS =  [f"batch{i}" for i in [8,9]]
        self.PERTURBATION = 'SNCA_Untreated'

class NeuronsEffectWTBaselineStressConfig(NeuronsEffectWTBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'WT_stress'

class NeuronsEffectFUSRevBaselineConfig(NeuronsEffectConfig):
    def __init__(self):
        super().__init__()
        self.BASELINE = 'FUSRevertant_Untreated'

class NeuronsEffectFUSRevBaselineFUSHomoConfig(NeuronsEffectFUSRevBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'FUSHomozygous_Untreated'

class NeuronsEffectFUSRevBaselineFUSHeteroConfig(NeuronsEffectFUSRevBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'FUSHeterozygous_Untreated'

class NeuronsEffectFUSRevBaselineWTConfig(NeuronsEffectFUSRevBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'WT_Untreated'

class NeuronsEffectFUSRevBaselineTDP43Config(NeuronsEffectFUSRevBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'TDP43_Untreated'

class NeuronsEffectFUSRevBaselineOPTNConfig(NeuronsEffectFUSRevBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'OPTN_Untreated'

class NeuronsEffectFUSRevBaselineTBK1Config(NeuronsEffectFUSRevBaselineConfig):
    def __init__(self):
        super().__init__()
        self.PERTURBATION = 'TBK1_Untreated'

class NeuronsEffectFUSRevBaselineSNCAConfig(NeuronsEffectFUSRevBaselineConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS =  [f"batch{i}" for i in [8,9]]
        self.PERTURBATION = 'SNCA_Untreated'

# Multiplex #
class NeuronsMultiplexEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS =  [f"batch{i}" for i in [1,2,3,7,8,9]]
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        
        self.BASELINE_PERTURB = {'WT_Untreated':['FUSHomozygous_Untreated',
                                                 'FUSHeterozygous_Untreated',
                                                 'FUSRevertant_Untreated',
                                                 'TDP43_Untreated',
                                                 'OPTN_Untreated',
                                                 'TBK1_Untreated']}
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant',
                           'TDP43','OPTN','TBK1']
        self.CONDITIONS = ['Untreated']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = False
        self.MIN_REQUIRED = 200 

class NeuronsMultiplexEffectConfig_FUSLines(NeuronsMultiplexEffectConfig):
    def __init__(self):
        super().__init__()
        
        self.BASELINE_PERTURB = {'WT_Untreated':['FUSHomozygous_Untreated',
                                                 'FUSHeterozygous_Untreated',
                                                 'FUSRevertant_Untreated']}
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant']


class NeuronsMultiplexEffectConfig_WithSNCA(NeuronsMultiplexEffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS =  [f"batch{i}" for i in [8,9]]
        
        self.BASELINE_PERTURB = {'WT_Untreated':['FUSHomozygous_Untreated',
                                                 'FUSHeterozygous_Untreated',
                                                 'FUSRevertant_Untreated',
                                                 'TDP43_Untreated',
                                                 'OPTN_Untreated',
                                                 'TBK1_Untreated',
                                                 'SNCA_Untreated']}
        self.CELL_LINES = ['WT','FUSHomozygous','FUSHeterozygous','FUSRevertant',
                           'TDP43','OPTN','TBK1', 'SNCA']

### Neurons Day 18 ###
class Day18EffectConfig(EffectConfig): ## need to be defined with baseline and perturbation if wanting to use this data
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1", "batch2"]
        
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.MARKERS_TO_EXCLUDE = ['TIA1','DAPI']        


### Alyssa Coyne (pilot) ###

class AlyssaCoyneEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne'    
        self.BASELINE_PERTURB = {'Controls_Untreated':[f'{cell_line}_Untreated'
                                                       for cell_line in 
                                                       ['sALSPositiveCytoTDP43','sALSNegativeCytoTDP43','c9orf72ALSPatients']]}
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        self.MIN_REQUIRED = 200 
        self.N_BOOT = 100 

### Alyssa Coyne (new) ###

class AlyssaCoyneNEWEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.BASELINE_PERTURB = {'Ctrl-EDi022_Untreated':
                                 ['C9-CS2YNL_Untreated','SALSPositive-CS2FN3_Untreated','SALSNegative-CS0ANK_Untreated'],
                                 'Ctrl-EDi029_Untreated':
                                 ['C9-CS7VCZ_Untreated','SALSPositive-CS4ZCD_Untreated','SALSNegative-CS0JPP_Untreated'],
                                 'Ctrl-EDi037_Untreated':
                                 ['C9-CS8RFT_Untreated','SALSPositive-CS7TN6_Untreated','SALSNegative-CS6ZU8_Untreated']}
        self.MIN_REQUIRED = 40 
        self.N_BOOT = 100 

class AlyssaCoyneNEWEffectConfig_Ctrl_C9(AlyssaCoyneNEWEffectConfig):
    def __init__(self):
        super().__init__()   
        self.BASELINE_PERTURB = {'Ctrl-EDi022_Untreated':['C9-CS2YNL_Untreated'],
                                 'Ctrl-EDi029_Untreated':['C9-CS7VCZ_Untreated'],
                                 'Ctrl-EDi037_Untreated':['C9-CS8RFT_Untreated']}

class AlyssaCoyneNEWMultiplexEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.BASELINE_PERTURB = {'Ctrl-EDi022_Untreated':
                                 ['C9-CS2YNL_Untreated','SALSPositive-CS2FN3_Untreated','SALSNegative-CS0ANK_Untreated'],
                                 'Ctrl-EDi029_Untreated':
                                 ['C9-CS7VCZ_Untreated','SALSPositive-CS4ZCD_Untreated','SALSNegative-CS0JPP_Untreated'],
                                 'Ctrl-EDi037_Untreated':
                                 ['C9-CS8RFT_Untreated','SALSPositive-CS7TN6_Untreated','SALSNegative-CS6ZU8_Untreated']
                                 }

        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = False

        self.MIN_REQUIRED = 40 # To include everyone
        self.N_BOOT = 100 

class AlyssaCoyneNEWMultiplexEffectConfig_Ctrl_C9(AlyssaCoyneNEWMultiplexEffectConfig):
    def __init__(self):
        super().__init__()
   
        self.BASELINE_PERTURB = {'Ctrl-EDi022_Untreated':['C9-CS2YNL_Untreated'],
                                 'Ctrl-EDi029_Untreated':['C9-CS7VCZ_Untreated'],
                                 'Ctrl-EDi037_Untreated':['C9-CS8RFT_Untreated']
                                 }

### dNLS ###

class dNLSEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1", 'batch2', 'batch4', 'batch5', 'batch6']
        self.EXPERIMENT_TYPE = 'dNLS'
        self.BASELINE = 'dNLS_Untreated'
        self.PERTURBATION = 'dNLS_DOX'

        self.MIN_REQUIRED:int = 200
        self.N_BOOT:int = 1000

class dNLSMultiplexEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS =  [f"batch{i}" for i in [1,2,4,5,6]]
        
        self.EXPERIMENT_TYPE = 'dNLS'    
        
        self.BASELINE_PERTURB = {'dNLS_Untreated':['dNLS_DOX']}
        self.CELL_LINES = ['dNLS']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = False
        self.MIN_REQUIRED = 40 # after testing number of multiplexed labels across batches
        self.N_BOOT = 1000

### NIH ###
class NIHEffectConfig_Stress(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS =  [f"batch{i}" for i in [1,2,3]]
        
        self.EXPERIMENT_TYPE = 'NIH'    
        self.BASELINE = 'WT_Untreated'
        self.PERTURBATION = 'WT_stress'

        self.MIN_REQUIRED:int = 450
        self.N_BOOT:int = 1000

class NIHMultiplexEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS =  [f"batch{i}" for i in [1,2,3]]
        
        self.EXPERIMENT_TYPE = 'NIH'    
        
        self.BASELINE_PERTURB = {'WT_Untreated':['WT_stress']}
        self.CELL_LINES = ['WT']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = False
        self.MIN_REQUIRED = 200 # after testing number of multiplexed labels across batches

### iAstrocytes ###
class iAstrocytesEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'iAstrocytes_Tile146'    
        self.BASELINE_PERTURB = {'WT_Untreated':['C9_Untreated']}
        self.MIN_REQUIRED = 200 



