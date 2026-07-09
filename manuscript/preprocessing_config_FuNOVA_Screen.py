import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig
from manuscript.FuNOVA_Screen_Conditions_Lists_b1 import (
    plate1_conditions as b1_plate1_conditions,
    plate2_conditions as b1_plate2_conditions,
    plate3_conditions as b1_plate3_conditions,
    plate4_conditions as b1_plate4_conditions,
)

from manuscript.FuNOVA_Screen_Conditions_Lists_b2 import (
    plate1_conditions as b2_plate1_conditions,
    plate2_conditions as b2_plate2_conditions,
    plate3_conditions as b2_plate3_conditions,
    plate4_conditions as b2_plate4_conditions,
)

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

        self.CELL_LINES = None # ["C9"]
        self.CONDITIONS = None
    
        self.REPS = None #["rep1","rep2"]
        self.PANELS = None # ["panelA","panelB","panelC","panelD"]
    
        self.NUCLEUS_MARKER_NAME = "DAPI"

        self.NUM_WORKERS = 9


        ########## preprocessing config thresholds ########## 
        self.RESCALE_INTENSITY = { # PER CHANNEL
            'LOWER_BOUND': [0.5 ,0.5], 
            'UPPER_BOUND': [99.2, 95.0] 
        }
        self.NO_RESCALE_FOR_LOW_SIGNAL_MARKERS = ["pS6"] # rescale only pS6 channel with (0,100)

        self.MAX_INTENSITY_THRESHOLD_NUCLEI:float = 0.17

        self.MAX_NUM_NUCLEI_BLOB:int = 11
        self.MIN_ALIVE_NUCLEI_AREA: int = 1100

        self.MAX_ECC:float = 0.92

        self.MIN_SOL:float = 0.88

        self.MAX_BLOB_AREA:int = 6000

        self.MIN_VARIANCE_THRESHOLD_ALIVE_NUCLEI: float = 0.015

        self.MIN_MEDIAN_INTENSITY_THRESHOLD_ALIVE_NUCLEI: float = 0.55 
        self.MAX_VARIANCE_THRESHOLD_ALIVE_NUCLEI: float = 0.028 
        self.MAX_MEDIAN_INTENSITY_THRESHOLD_ALIVE_NUCLEI: float = 0.98 


class PreprocessingBaseConfigFuNOVAScreenBatch1(PreprocessingBaseConfigFuNOVAScreen):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT,"batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT,"batch1")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER,"batch1")

        self.MARKERS = ['DAPI', 'p62','ATF6', 'G3BP1', 'Aggreagtes', 'pS6'] # DISCARD - 'HDGFL2', 'FK-2', 'Calreticulin'; use only new cy3 markers

        self.MARKERS_TO_EXCLUDE = None #['DAPI_new_cy3', 'TDP-43_new_cy3', 'pTDP-43_new_cy3', 'pAMPK_new_cy3', 'Cas3_new_cy3']
        
        self.NUCLEUS_MARKER_NAME = "DAPI"    

        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_HOME"),
            'manuscript',
            'markers_focus_boundries',
            'markers_focus_boundries_FuNOVA_Screen.csv'
        )

class PreprocessingBaseConfigFuNOVAScreenBatch1_NewCy3(PreprocessingBaseConfigFuNOVAScreenBatch1):
    def __init__(self):
        super().__init__()
        
        self.MARKERS = ['DAPI_new_cy3', 'TDP-43_new_cy3', 'pTDP-43_new_cy3', 'pAMPK_new_cy3', 'Cas3_new_cy3'] # use only new cy3 markers

        self.MARKERS_TO_EXCLUDE = None #['DAPI', 'p62','ATF6', 'G3BP1', 'Aggreagtes', 'pS6']

        self.NUCLEUS_MARKER_NAME = "DAPI_new_cy3"



class PreprocessingBaseConfigFuNOVAScreenBatch2(PreprocessingBaseConfigFuNOVAScreen):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT,"batch2")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT,"batch2")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER,"batch2")

        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_HOME"),
            'manuscript',
            'markers_focus_boundries',
            'markers_focus_boundries_FuNOVA_Screen_b2.csv'
        )

        # AFTER FIRST RUN STOPPED BEACUSE OF TIME LIMIT
        self.MARKERS = ['DAPI', 'TDP-43', 'p62', 'pTDP-43', 'ATF6', 'pAMPK', 'G3BP1', 'Aggreagtes', 'Cas3', 'pS6'] # DISCARD - 'HDGFL2', 'FK-2', 'Calreticulin'

        self.MARKERS_TO_EXCLUDE = None

        self.NUCLEUS_MARKER_NAME = "DAPI"