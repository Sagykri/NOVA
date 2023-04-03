
from models.neuroself.configs.config import NeuroselfConfig


class RunConfigExample(NeuroselfConfig):
    
    def __init__(self):
        super(NeuroselfConfig, self).__init__()
        
        self.MARKERS_TO_EXCLUDE = ['DAPI', 'lysotracker', 'Syto12']
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['unstressed', 'stressed']
        self.SPLIT_DATA = True
        self.DATA_SET_TYPE = 'test'
        self.MARKERS_FOR_DOWNSAMPLE = None
        self.TRAIN_PCT = 0.7
        self.SHUFFLE = True
        self.ADD_LINE_TO_LABEL = False
        self.ADD_TYPE_TO_LABEL = False
        self.ADD_CONDITION_TO_LABEL = True 