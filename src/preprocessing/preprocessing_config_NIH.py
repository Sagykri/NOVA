import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig

class PreprocessingBaseConfigNIH(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'Cory', 'indi-image-pilot-20241128')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "NIH")
        # self.OUTPUTS_FOLDER = os.path.join(self.HOME_FOLDER, "outputs", "NIH")
        self.OUTPUTS_FOLDER = os.path.join("/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/preprocessing/", "NIH")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_nih", "NIHPreprocessor")
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_NIH.csv')
        
        self.CELL_LINES = ['WT', 'FUSRevertant', 'FUSHomozygous', 'FUSHeterozygous']
        self.CONDITIONS = ['Untreated', 'stress']
        self.REPS = [f"rep{i}" for i in range(1,9)]
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, f"batch{i}") for i in range(1,4)]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f"batch{i}") for i in range(1,4)]
