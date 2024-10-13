import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig

class SPDPreprocessingBaseConfig(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'SpinningDisk')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "preprocessing", "spd")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_spd", "SPDPreprocessor")

class NeuronsPreprocessingConfig(SPDPreprocessingBaseConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_spd.csv')
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "neurons")

class Batch6(NeuronsPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch6")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch6")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch6")
       
class Batch7(NeuronsPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch7")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch7")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch7")
       
class Batch8(NeuronsPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch8")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch8")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch8")
       
class Batch9(NeuronsPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch9")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch9")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch9")
        
###################
#       dNLS      #
###################

class dNLSPreprocessingConfig(SPDPreprocessingBaseConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_dNLS.csv')
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "deltaNLS")


class dNLS_Batch3(dNLSPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "deltaNLS_sort", "batch3")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "deltaNLS","batch3")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch3")

class dNLS_Batch4(dNLSPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "deltaNLS_sort", "batch4")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "deltaNLS","batch4")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch4")
       
class dNLS_Batch5(dNLSPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "deltaNLS_sort", "batch5")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "deltaNLS","batch5")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch5")
        
