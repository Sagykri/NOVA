import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig

DATA_DIR = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/RANBP17_exp/co-localization_and_KD"
class PreprocessingBaseConfigRANBP17Exp(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = DATA_DIR
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT,'sorted')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT,'RANBP17_exp')
        self.OUTPUTS_FOLDER =  os.path.join(os.getenv("NOVA_HOME"),'outputs','preprocessing','RANBP17_exp')
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_FOLDER,'logs')
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src","preprocessing","preprocessors","preprocessor_opera","OperaPreprocessor")        

        self.CELL_LINES = ["iW11"]
        self.PANELS = ["panelA"]
        self.MARKERS = ['DAPI', 'TDP-43', 'RANBP17']
        self.NUCLEUS_MARKER_NAME = "DAPI"

        


class PreprocessingBaseConfigRANBP17ExpBatch1(PreprocessingBaseConfigRANBP17Exp):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT,"batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT,"batch1")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER,"batch1")

        self.CONDITIONS = ["tardp-kd", "ranbp17-kd", "control-179", "control-180", "both-kd", "untreated"]
        self.REPS = ["rep1","rep2","rep3","rep4"]

        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_HOME"),
            'manuscript',
            'markers_focus_boundries',
            'markers_focus_boundries_RANBP17_exp.csv'
        )


class PreprocessingBaseConfigRANBP17ExpBatch2(PreprocessingBaseConfigRANBP17Exp):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT,"batch2")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT,"batch2")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER,"batch2")

        self.CONDITIONS = ["tardp-kd", "ranbp17-kd", "control-179", "control-180", "both-kd"]
        self.REPS = ["rep1","rep2","rep3","rep4"]
        
        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_HOME"),
            'manuscript',
            'markers_focus_boundries',
            'markers_focus_boundries_RANBP17_exp_batch2.csv'
        )

class PreprocessingBaseConfigRANBP17ExpBatch3(PreprocessingBaseConfigRANBP17Exp):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT,"batch3")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT,"batch3")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER,"batch3")

        self.CONDITIONS = ["tardp-kd", "ranbp17-kd", "control-179", "control-180", "both-kd", "untreated"]
        self.REPS = ["rep1","rep2"]
        
        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_HOME"),
            'manuscript',
            'markers_focus_boundries',
            'markers_focus_boundries_RANBP17_exp_batch3.csv'
        )


