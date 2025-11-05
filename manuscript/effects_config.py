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

        ## Sagy 08.09.25
        self.MIN_REQUIRED:int = 80 # with subsample fraction = 0.8 it enforce at least 33 tiles per condition X marker X batch
        self.SUBSAMPLE_FRACTION = 0.8
        ###
        self.N_BOOT:int = 1000

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
        self.MIN_REQUIRED = 8#40 
        self.N_BOOT = 1000#100  # Must have many boots for trimming

        # Sagy 9.9.25
        self.SUBSAMPLE_FRACTION = 1.0 # use all samples (no subsampling) cause we have few samples
        self.BOOTSTRAP_TRIMMING_ALPHA = 0.01  # trim 1% from each side
        #

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
        self.N_BOOT = 1000  #100

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

        ## Sagy 08.09.25
        # self.MIN_REQUIRED:int = 200 # OLD
        self.MIN_REQUIRED:int = 80 # with subsample fraction = 0.8 it enforce at least 33 tiles per condition X marker X batch
        self.SUBSAMPLE_FRACTION = 0.8 # 
        ###

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

        ## Sagy 08.09.25
        # self.MIN_REQUIRED:int = 450 # OLD
        self.MIN_REQUIRED:int = 80 # with subsample fraction = 0.8 it enforce at least 33 tiles per condition X marker X batch
        self.SUBSAMPLE_FRACTION = 0.8
        ###

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


class AATNOVAEffectConfig(EffectConfig):
    def __init__(self):
        super().__init__()

        self.EXPERIMENT_TYPE = 'AAT_NOVA'

        # The path to the data folders
        self.INPUT_FOLDERS =  ["batch1", "batch2"]
        
        self.BASELINE:str = None # example: WT_Untreated
        self.PERTURBATION:str = None # example: WT_stress
        # Dictionary mapping each baseline to a list of perturbations.
        self.BASELINE_PERTURB:Dict[int:List[int]] = None # Used for Alyssa's data. for example: {'WT_Untreated':['WT_stress']}

        self.MIN_REQUIRED:int = 30 # min required sites!

        self.N_BOOT:int = 200

        self.SUBSAMPLE_FRACTION = 0.8 # fraction of samples to use in each bootstrap iteration (the formula is: max(MIN_REQUIRED, int(n_samples ** SUBSAMPLE_FRACTION)) (i.e. to the power))

        self.BOOTSTRAP_TRIMMING_ALPHA = 0 # fraction of extreme values to trim from the bootstrap distribution for estimating the variance (e.g. 0.01 means trimming 1% from each tail) (Default: 0, no trimming)

        self.MARKERS = ['pTDP-43','TDP-43','FK2','SMI-32','LC3-II','UNC13A']
# "combined-NT"
# KD_list = ["PPP2R1A","HMGCS1","PIK3C3","NDUFAB1","MAPKAP1","NDUFS2","RALA","TLK1","NRIP1","TARDBP","RANBP17","CYLD"]

# ------------- C9: NT vs KD -----------------
class AATNOVAEffectConfigC9_NTvsKD(AATNOVAEffectConfig):
    def __init__(self):
        super().__init__()
        self.BASELINE:str =  "C9_combined-NT" # example: WT_Untreated
        self.PERTURBATION:str = f"C9_{self.PERTUB_NAME}" # example: WT_stress

class AATNOVAEffectConfigC9_NTvsPPP2R1A(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "PPP2R1A"
        super().__init__()

class AATNOVAEffectConfigC9_NTvsHMGCS1(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "HMGCS1"
        super().__init__()

class AATNOVAEffectConfigC9_NTvsPIK3C3(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "PIK3C3"
        super().__init__()
class AATNOVAEffectConfigC9_NTvsNDUFAB1(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "NDUFAB1"
        super().__init__()
class AATNOVAEffectConfigC9_NTvsMAPKAP1(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "MAPKAP1"
        super().__init__()
class AATNOVAEffectConfigC9_NTvsNDUFS2(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "NDUFS2"
        super().__init__()
class AATNOVAEffectConfigC9_NTvsRALA(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "RALA"
        super().__init__()
class AATNOVAEffectConfigC9_NTvsTLK1(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "TLK1"
        super().__init__()
class AATNOVAEffectConfigC9_NTvsNRIP1(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "NRIP1"
        super().__init__()
class AATNOVAEffectConfigC9_NTvsTARDBP(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "TARDBP"
        super().__init__()
class AATNOVAEffectConfigC9_NTvsRANBP17(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "RANBP17"
        super().__init__()
class AATNOVAEffectConfigC9_NTvsCYLD(AATNOVAEffectConfigC9_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "CYLD"
        super().__init__()

# ------------- CTL: NT vs KD -----------------
class AATNOVAEffectConfigCTL_NTvsKD(AATNOVAEffectConfig):
    def __init__(self):
        super().__init__()
        self.BASELINE:str =  "CTL_combined-NT"# example: WT_Untreated
        self.PERTURBATION:str = f"CTL_{self.PERTUB_NAME}" # example: WT_stress

class AATNOVAEffectConfigCTL_NTvsPPP2R1A(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "PPP2R1A"
        super().__init__()

class AATNOVAEffectConfigCTL_NTvsHMGCS1(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "HMGCS1"
        super().__init__()

class AATNOVAEffectConfigCTL_NTvsPIK3C3(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "PIK3C3"
        super().__init__()
class AATNOVAEffectConfigCTL_NTvsNDUFAB1(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "NDUFAB1"
        super().__init__()
class AATNOVAEffectConfigCTL_NTvsMAPKAP1(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "MAPKAP1"
        super().__init__()
class AATNOVAEffectConfigCTL_NTvsNDUFS2(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "NDUFS2"
        super().__init__()
class AATNOVAEffectConfigCTL_NTvsRALA(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "RALA"
        super().__init__()
class AATNOVAEffectConfigCTL_NTvsTLK1(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "TLK1"
        super().__init__()
class AATNOVAEffectConfigCTL_NTvsNRIP1(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "NRIP1"
        super().__init__()
class AATNOVAEffectConfigCTL_NTvsTARDBP(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "TARDBP"
        super().__init__()
class AATNOVAEffectConfigCTL_NTvsRANBP17(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "RANBP17"
        super().__init__()
class AATNOVAEffectConfigCTL_NTvsCYLD(AATNOVAEffectConfigCTL_NTvsKD):
    def __init__(self):
        self.PERTUB_NAME:str = "CYLD"
        super().__init__()

# ------------- KD: CTL vs C9 -----------------
class AATNOVAEffectConfigKD_CTLvsC9(AATNOVAEffectConfig):
    def __init__(self, perturb_name: str):
        """
        Base configuration for CTL vs C9 comparison under a given perturbation.
        Example: perturb_name = 'PPP2R1A' → baseline = CTL_PPP2R1A, perturbation = C9_PPP2R1A
        """
        super().__init__()
        self.PERTUB_NAME = perturb_name
        self.BASELINE = f"CTL_{self.PERTUB_NAME}"
        self.PERTURBATION = f"C9_{self.PERTUB_NAME}"


# ------------- Individual perturbation configs -----------------
class AATNOVAEffectConfigPPP2R1A_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("PPP2R1A")


class AATNOVAEffectConfigHMGCS1_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("HMGCS1")


class AATNOVAEffectConfigPIK3C3_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("PIK3C3")


class AATNOVAEffectConfigNDUFAB1_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("NDUFAB1")


class AATNOVAEffectConfigMAPKAP1_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("MAPKAP1")


class AATNOVAEffectConfigNDUFS2_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("NDUFS2")


class AATNOVAEffectConfigRALA_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("RALA")


class AATNOVAEffectConfigTLK1_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("TLK1")


class AATNOVAEffectConfigNRIP1_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("NRIP1")


class AATNOVAEffectConfigTARDBP_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("TARDBP")


class AATNOVAEffectConfigRANBP17_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("RANBP17")


class AATNOVAEffectConfigCYLD_CTLvsC9(AATNOVAEffectConfigKD_CTLvsC9):
    def __init__(self):
        super().__init__("CYLD")

# ------------- NT: CTL vs C9 -----------------
class AATNOVAEffectConfigNT_CTLvsC9(AATNOVAEffectConfig):
    def __init__(self):
        """
        Base configuration for CTL vs C9 comparison under a given perturbation.
        Example: perturb_name = 'PPP2R1A' → baseline = CTL_PPP2R1A, perturbation = C9_PPP2R1A
        """
        super().__init__()
        self.BASELINE = f"CTL_combined-NT"
        self.PERTURBATION = f"C9_combined-NT"
