import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig

class PreprocessingBaseConfigFUNOVA(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        # self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'Cory', 'indi-image-pilot-20241128')
        # self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "NIH")
        self.OUTPUTS_FOLDER = os.path.join("/home/labs/hornsteinlab/Collaboration/FUNOVA/outputs/preprocessing/")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")        
        self.CELL_LINES = ["C9orf72-HRE-1008566",
                            "C9orf72-HRE-981344",
                            "Control-1001733",
                            "Control-1017118",
                            "Control-1025045",
                            "Control-1048087",
                            "TDP--43-G348V-1057052",
                            "TDP--43-N390D-1005373"]
        
        self.CONDITIONS = ['Untreated', 'stress']
        self.REPS = [f"rep{i}" for i in range(1,3)]
        # self.CELLPOSE = {
        #     'NUCLEUS_DIAMETER': 60,
        #     'CELLPROB_THRESHOLD': 0,
        #     'FLOW_THRESHOLD': 0.6
        # }

class PreprocessingBaseConfigFUNOVAExp3(PreprocessingBaseConfigFUNOVA):
    def __init__(self):
        super().__init__()
        
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_funova_Exp3_25.2.25.csv')
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, f"Batch{i}") for i in range(1,3)]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f"Batch{i}") for i in range(1,3)]
        self.MARKERS_FOCUS_BOUNDRIES_TILES_PATH = os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_funova_Exp3_tiles.csv')
        

class PreprocessingBaseConfigFUNOVAExp4(PreprocessingBaseConfigFUNOVA):
    def __init__(self):
        super().__init__()
        
        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_funova_Exp4_25.2.25.csv')
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, f"Batch{i}") for i in range(3,5)]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f"Batch{i}") for i in range(3,5)]
        self.MARKERS_FOCUS_BOUNDRIES_TILES_PATH = os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_funova_Exp4_tiles.csv')
