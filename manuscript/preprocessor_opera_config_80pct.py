import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig

class Opera18DaysReimagedPreprocessingBaseConfig(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'Opera18DaysReimaged_sorted')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "Opera18Days_Reimaged")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_opera18days_reimaged.csv')

class NeuronsD18_Batch1(Opera18DaysReimagedPreprocessingBaseConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch180pct")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch180pct")
        
        
class NeuronsD18_Batch2(Opera18DaysReimagedPreprocessingBaseConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, "batch2")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "batch280pct")]
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "batch280pct")
