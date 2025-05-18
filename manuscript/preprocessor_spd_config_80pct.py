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

class Batch680pct(NeuronsPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch6")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch680pct")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch680pct")
       
class Batch780pct(NeuronsPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch7")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch780pct")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch780pct")
       
class Batch880pct(NeuronsPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch8")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch880pct")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch880pct")
       
class Batch980pct(NeuronsPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch9")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch980pct")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch980pct")

###################
#       dNLS      #
###################

class dNLSPreprocessingConfig(SPDPreprocessingBaseConfig):
    def __init__(self):
        super().__init__()
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_dNLS.csv')
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "deltaNLS")

class dNLS_Batch380pct(dNLSPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "deltaNLS_sort", "batch3")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "deltaNLS","batch380pct")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch380pct")

class dNLS_Batch480pct(dNLSPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "deltaNLS_sort", "batch4")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "deltaNLS","batch480pct")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch480pct")
       
class dNLS_Batch580pct(dNLSPreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "deltaNLS_sort", "batch5")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "deltaNLS","batch580pct")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch580pct")