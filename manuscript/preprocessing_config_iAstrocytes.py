import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig

class PreprocessingBaseConfigiAstrocytes(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'John', 'iAstrocytes', 'ordered')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "iAstrocytes")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "iAstrocytes")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")        

        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_iAstrocytes.csv')
        
        self.CELL_LINES = ['WT', 'C9']
        self.CONDITIONS = ['Untreated']
        self.REPS = [f"rep{i}" for i in range(1,4)]
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, f"batch{i}") for i in range(1,2)]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f"batch{i}") for i in range(1,2)]

class PreprocessingBaseConfigiAstrocytesTile146(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'John', 'iAstrocytes', 'ordered')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "iAstrocytes_Tile146")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "iAstrocytes_Tile146")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")        

        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_iAstrocytes.csv')
        
        self.CELL_LINES = ['WT', 'C9']
        self.CONDITIONS = ['Untreated']
        self.REPS = [f"rep{i}" for i in range(1,4)]
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, f"batch{i}") for i in range(1,2)]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f"batch{i}") for i in range(1,2)]
        
        # The expected image shape
        self.EXPECTED_IMAGE_SHAPE = (1022,1022)
        # The tile shape when cropping the image into tiles
        self.TILE_INTERMEDIATE_SHAPE = (146,146)

class PreprocessingBaseConfigiAstrocytesTile146(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'John', 'iAstrocytes', 'ordered')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "iAstrocytes_Tile146")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "iAstrocytes_Tile146")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")        

        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_iAstrocytes.csv')
        
        self.CELL_LINES = ['WT', 'C9']
        self.CONDITIONS = ['Untreated']
        self.REPS = [f"rep{i}" for i in range(1,4)]
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, f"batch{i}") for i in range(1,2)]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f"batch{i}") for i in range(1,2)]
        
        # The expected image shape
        self.EXPECTED_IMAGE_SHAPE = (1022,1022)
        # The tile shape when cropping the image into tiles
        self.TILE_INTERMEDIATE_SHAPE = (146,146)

class PreprocessingBaseConfigiAstrocytesTile160(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT, 'John', 'iAstrocytes', 'ordered')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT, "iAstrocytes_Tile160")
        self.OUTPUTS_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "iAstrocytes_Tile160")
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src", "preprocessing", "preprocessors", "preprocessor_opera", "OperaPreprocessor")        

        self.MARKERS_FOCUS_BOUNDRIES_PATH =  os.path.join(os.getenv("NOVA_HOME"), 'manuscript', 'markers_focus_boundries', 'markers_focus_boundries_iAstrocytes.csv')
        
        self.CELL_LINES = ['WT', 'C9']
        self.CONDITIONS = ['Untreated']
        self.REPS = [f"rep{i}" for i in range(1,4)]
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT, f"batch{i}") for i in range(1,2)]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f"batch{i}") for i in range(1,2)]
        
        # The expected image shape
        self.EXPECTED_IMAGE_SHAPE = (960,960)
        # The tile shape when cropping the image into tiles
        self.TILE_INTERMEDIATE_SHAPE = (160,160)
