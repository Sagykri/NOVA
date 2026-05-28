import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.figures_config import FigureConfig


RANBP17_exp_MARKERS = ['DAPI', 'TDP-43', 'RANBP17']
RANBP17_exp_MARKERS_TO_EXCLUDE = None
RANBP17_exp_CELL_LINES = ["iW11"]
RANBP17_exp_BATCHES = ['batch1', 'batch2', 'batch3']


RANBP17_exp_ALL_CONDITIONS = ["tardp-kd", "ranbp17-kd", "control-179", "control-180", "both-kd", "untreated"]

RANBP17_exp_ALL_KD_CONDITIONS = ["tardp-kd", "ranbp17-kd", "both-kd"]

RANBP17_exp_ALL_CONTROL_CONDITIONS = ["control-179", "control-180", "untreated"]


class RANBP17_exp_BaseFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.EXPERIMENT_TYPE = 'RANBP17_exp'
        self.CELL_LINES = RANBP17_exp_CELL_LINES
        self.CONDITIONS = None
        self.MARKERS = None
        self.MARKERS_TO_EXCLUDE = RANBP17_exp_MARKERS_TO_EXCLUDE
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True

class RANBP17_exp_Batch1FigureConfig(RANBP17_exp_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch1']

class RANBP17_exp_Batch2FigureConfig(RANBP17_exp_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch2']

class RANBP17_exp_Batch3FigureConfig(RANBP17_exp_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ['batch3']

