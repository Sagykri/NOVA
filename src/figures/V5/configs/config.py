import os
from src.common.configs.figure_config import FigureConfig
from models.cytoself.configs.config import CytoselfConfig
from models.imgself.configs.config import ImgselfConfig
from models.neuroself.configs.config import NeuroselfConfig

class FigureV5Config(FigureConfig):
    def __init__(self):
        super().__init__()
        
        self.HOME_SUBFOLDER = os.path.join(self.HOME_FIGURES_FOLDER, "V5")
        
        self.neuroself_config = NeuroselfConfig()
        self.imgself_config = ImgselfConfig()
        self.cytoself_config = CytoselfConfigFig1()
        
        self.neuroself_config.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, "logs")
        self.imgself_config.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, "logs")
        self.cytoself_config.LOGS_FOLDER = os.path.join(self.HOME_SUBFOLDER, "logs")
        
        
        
        self.output_folder = os.path.join(self.HOME_SUBFOLDER, "outputs", "figures")
        
        # Example: self.figures = {fig_id: [panel_id1, panel_id2, ...], fig_id2: [..], ...}
        self.figures = {}


class CytoselfConfigFig1(CytoselfConfig):
    def __init__(self):
        super().__init__()
        
        self.ADD_CONDITION_TO_LABEL = True 
        self.ADD_LINE_TO_LABEL = False