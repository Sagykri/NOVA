import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig

# Neurons Day 8
class NeuronsD8PreprocessingConfig_80pct(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'SpinningDisk')
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_spd.csv')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "preprocessing", "ManuscriptFinalData_80pct", "neuronsDay8")
        
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_spd", "SPDPreprocessor")

        self.MARKERS_TO_EXCLUDE = ['TIA1']

class NeuronsD8PreprocessingConfig_80pct_B5(NeuronsD8PreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch5")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch5")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch5")

class NeuronsD8PreprocessingConfig_80pct_B6(NeuronsD8PreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch6")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch6")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch6")
       
class NeuronsD8PreprocessingConfig_80pct_B7(NeuronsD8PreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch7")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch7")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch7")
       
class NeuronsD8PreprocessingConfig_80pct_B8(NeuronsD8PreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch8")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch8")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch8")
       
class NeuronsD8PreprocessingConfig_80pct_B9(NeuronsD8PreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch9")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch9")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch9")


######
## Neurons Day 18
####

class NeuronsD18PreprocessingConfig_80pct(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'Opera18DaysReimaged_sorted')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay18")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "preprocessing", "ManuscriptFinalData_80pct", "neuronsDay18")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'marker_focus_boundries_neuronsDay18_240825.csv')

        self.VARIANCE_THRESHOLD_NUCLEI:float = 0.02 # after testing on both batches, FUS lines and WT
        self.MIN_ALIVE_NUCLEI_AREA: int = -1
        self.MARKERS_TO_EXCLUDE = ['CD41']

class NeuronsD18PreprocessingConfig_80pct_B1(NeuronsD18PreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch1")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch1")
        
        
class NeuronsD18PreprocessingConfig_80pct_B2(NeuronsD18PreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch2")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch2")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch2")


####
## Alyssa
####

class AlyssaPreprocessingConfig_B1(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, 'AlyssaCoyne', 'MOmaps_iPSC_patients_TDP43_PB_CoyneLab', 'batch1')]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "AlyssaCoyne",'batch1')]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "preprocessing", "ManuscriptFinalData_80pct", "AlyssaCoyne", "batch1")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_spd", "SPDPreprocessor")
        self.RESCALE_INTENSITY = {
          'LOWER_BOUND': 0,
          'UPPER_BOUND': 100,
        }
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  None
        self.TILE_INTERMEDIATE_SHAPE = (146,146)
        self.EXPECTED_IMAGE_SHAPE = (1022,1022)

        
        self.MIN_ALIVE_NUCLEI_AREA: int = -1

        self.MARKERS_TO_EXCLUDE = ['MERGED']

###      
## new Alyssa
##
class AlyssaPreprocessingConfig_New_B1_woBrenner(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, 'AlyssaCoyne_new_sorted', 'batch1')]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "AlyssaCoyne_new",'batch1')]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "preprocessing", "ManuscriptFinalData_80pct", "AlyssaCoyne_new", "batch1")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_spd", "SPDPreprocessor")
        self.RESCALE_INTENSITY = {
          'LOWER_BOUND': 0,
          'UPPER_BOUND': 100,
        }
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  None
        self.TILE_INTERMEDIATE_SHAPE = (146,146)
        self.EXPECTED_IMAGE_SHAPE = (1022,1022)

        self.MIN_MEDIAN_INTENSITY_NUCLEI_BLOB_THRESHOLD = 0.8
        self.MIN_ALIVE_NUCLEI_AREA = -1  # No minimum alive nuclei area for this dataset

        self.MARKERS_TO_EXCLUDE = ['CD41']

####
# new dNLS
####

class dNLSPreprocessingConfig_80pct(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'OPERA_dNLS_6_batches_NOVA_sorted')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "dNLS")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "preprocessing", "ManuscriptFinalData_80pct", "dNLS")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_operadNLS.csv')

        self.MIN_ALIVE_NUCLEI_AREA = 800

class dNLSPreprocessingConfig_80pct_B1(dNLSPreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch1CLEAN")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch1CLEAN")

class dNLSPreprocessingConfig_80pct_B2(dNLSPreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch2")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch2CLEAN")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch2CLEAN")

class dNLSPreprocessingConfig_80pct_B3(dNLSPreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch3")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch3CLEAN")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch3CLEAN")

class dNLSPreprocessingConfig_80pct_B4(dNLSPreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch4")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch4CLEAN")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch4CLEAN")

class dNLSPreprocessingConfig_80pct_B5(dNLSPreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch5")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch5CLEAN")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch5CLEAN")

class dNLSPreprocessingConfig_80pct_B6(dNLSPreprocessingConfig_80pct):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch6")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch6CLEAN")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch6CLEAN")

####
## new INDIs
####

class NeuronsD8PreprocessingConfig_80pct_New(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'OPERA_indi_sorted')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "preprocessing", "ManuscriptFinalData_80pct", "neuronsDay8_new")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")
        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_newINDI_allBatches.csv')

        self.MIN_ALIVE_NUCLEI_AREA = 800

class NeuronsD8PreprocessingConfig_80pct_New_B1(NeuronsD8PreprocessingConfig_80pct_New):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch1")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch1")

class NeuronsD8PreprocessingConfig_80pct_New_B2(NeuronsD8PreprocessingConfig_80pct_New):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch2")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch2")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch2")

class NeuronsD8PreprocessingConfig_80pct_New_B3(NeuronsD8PreprocessingConfig_80pct_New):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch3")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch3")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch3")


class NeuronsD8PreprocessingConfig_80pct_New_B7(NeuronsD8PreprocessingConfig_80pct_New):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch7")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch7")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch7")

class NeuronsD8PreprocessingConfig_80pct_New_B8(NeuronsD8PreprocessingConfig_80pct_New):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch8")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch8")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch8")

class NeuronsD8PreprocessingConfig_80pct_New_B9(NeuronsD8PreprocessingConfig_80pct_New):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch9")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch9")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch9")

class NeuronsD8PreprocessingConfig_80pct_New_B10(NeuronsD8PreprocessingConfig_80pct_New):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch10")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch10")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch10")
