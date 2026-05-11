import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.figures_config import FigureConfig
from NOVA.manuscript.FuNOVA_Screen_Conditions_Lists import (
    plate1_conditions, plate2_conditions, plate3_conditions, plate4_conditions,
    plate1_control_conditions, plate2_control_conditions, plate3_control_conditions, plate4_control_conditions,
    plate1_kds, plate2_kds, plate3_kds, plate4_kds,
)

FUNOVA_SCREEN_MARKERS = [
    'DAPI', 'TDP-43', 'p62', 'pTDP-43', 'ATF6', 'pAMPK',
    'G3BP1', 'Calreticulin', 'Aggreagtes', 'Cas3', 'pS6',
]
FUNOVA_SCREEN_MARKERS_TO_EXCLUDE = ['HDGFL2', 'FK-2']
FUNOVA_SCREEN_BATCHES = ['batch1']
FUNOVA_SCREEN_CELL_LINES = ['C9']

FUNOVA_SCREEN_CONDITIONS_PLATES = {
    'plate1': plate1_conditions,
    'plate2': plate2_conditions,
    'plate3': plate3_conditions,
    'plate4': plate4_conditions,
}
FUNOVA_SCREEN_CONTROL_CONDITIONS_PLATES = {
    'plate1': plate1_control_conditions,
    'plate2': plate2_control_conditions,
    'plate3': plate3_control_conditions,
    'plate4': plate4_control_conditions,
}
FUNOVA_SCREEN_KDS_CONDITIONS_PLATES = {
    'plate1': plate1_kds,
    'plate2': plate2_kds,
    'plate3': plate3_kds,
    'plate4': plate4_kds,
}
FUNOVA_SCREEN_ALL_CONDITIONS = (
    plate1_conditions + plate2_conditions + plate3_conditions + plate4_conditions
)


class FuNOVA_Screen_BaseFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.EXPERIMENT_TYPE = 'FuNOVA_Screen'
        self.INPUT_FOLDERS = list(FUNOVA_SCREEN_BATCHES)
        self.CELL_LINES = list(FUNOVA_SCREEN_CELL_LINES)
        self.CONDITIONS = None
        self.MARKERS = None
        self.MARKERS_TO_EXCLUDE = list(FUNOVA_SCREEN_MARKERS_TO_EXCLUDE)
        self.ADD_REP_TO_LABEL = True
        self.ADD_BATCH_TO_LABEL = True
