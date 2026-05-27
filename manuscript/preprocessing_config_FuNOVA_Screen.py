import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig
from manuscript.FuNOVA_Screen_Conditions_Lists import plate1_conditions, plate2_conditions, plate3_conditions, plate4_conditions
FuNOVA_DATA_DIR = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen"
class PreprocessingBaseConfigFuNOVAScreen(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = FuNOVA_DATA_DIR
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT,'sorted')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT,'FuNOVA_Screen')
        self.OUTPUTS_FOLDER =  os.path.join(os.getenv("NOVA_HOME"),'outputs','preprocessing','FuNOVA_Screen')
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_FOLDER,'logs')
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src","preprocessing","preprocessors","preprocessor_opera","OperaPreprocessor")        

        self.CELL_LINES = ["C9"]
        self.CONDITIONS = plate1_conditions + plate2_conditions + plate3_conditions + plate4_conditions
        self.REPS = ["rep1","rep2"]
        self.PANELS = ["Panel1","Panel2","Panel3","Panel4"]
        self.MARKERS = ['DAPI_new_cy3', 'TDP-43_new_cy3', 'pTDP-43_new_cy3', 'pAMPK_new_cy3', 'Cas3_new_cy3'] # ['DAPI', 'TDP-43', 'p62', 'pTDP-43', 'ATF6', 'pAMPK', 'G3BP1', 'Calreticulin', 'Aggreagtes', 'Cas3', 'pS6'] # DISCARD - 'HDGFL2', 'FK-2'
        self.NUCLEUS_MARKER_NAME = "DAPI_new_cy3"

        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_HOME"),
            'manuscript',
            'markers_focus_boundries',
            'markers_focus_boundries_FuNOVA_Screen.csv'
        )


class PreprocessingBaseConfigFuNOVAScreenBatch1(PreprocessingBaseConfigFuNOVAScreen):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT,"batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT,"batch1")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER,"batch1")

