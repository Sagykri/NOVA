import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig

### Neurons Day 8 NEW ###
class EmbeddingsDay8NewDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = None
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsDay8B1DatasetConfig(EmbeddingsDay8NewDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in 
                        ["batch1"]]
        
class EmbeddingsDay8B2DatasetConfig(EmbeddingsDay8NewDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in 
                        ["batch2"]]

class EmbeddingsDay8B3DatasetConfig(EmbeddingsDay8NewDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in 
                        ["batch3"]]
        
class EmbeddingsDay8B7DatasetConfig(EmbeddingsDay8NewDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in 
                        ["batch7"]]

class EmbeddingsDay8B8DatasetConfig(EmbeddingsDay8NewDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in 
                        ["batch8"]]
        
class EmbeddingsDay8B9DatasetConfig(EmbeddingsDay8NewDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in 
                        ["batch9"]]
        


# AlyssaCoyne (pilot)
class EmbeddingsAlyssaCoyneDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "AlyssaCoyne", f) for f in 
                        ["batch1"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'AlyssaCoyne'    
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

# AlyssaCoyne new
class EmbeddingsAlyssaCoyneNEWDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "AlyssaCoyne_new", f) for f in
                        ["batch1"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

# new dNLS
class EmbeddingsNewdNLSDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = None
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'dNLS'
        self.MARKERS_TO_EXCLUDE = []
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsNewdNLSB1DatasetConfig(EmbeddingsNewdNLSDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "dNLS", f) for f in 
                        ["batch1"]]

class EmbeddingsNewdNLSB2DatasetConfig(EmbeddingsNewdNLSDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "dNLS", f) for f in 
                        ["batch2"]]

class EmbeddingsNewdNLSB3DatasetConfig(EmbeddingsNewdNLSDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "dNLS", f) for f in 
                        ["batch3"]]

class EmbeddingsNewdNLSB4DatasetConfig(EmbeddingsNewdNLSDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "dNLS", f) for f in 
                        ["batch4"]]

class EmbeddingsNewdNLSB5DatasetConfig(EmbeddingsNewdNLSDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "dNLS", f) for f in 
                        ["batch5"]]

class EmbeddingsNewdNLSB6DatasetConfig(EmbeddingsNewdNLSDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "dNLS", f) for f in 
                        ["batch6"]]


### NIH ##

class EmbeddingsNIHDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = None
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'NIH'
        self.MARKERS_TO_EXCLUDE = ['CD41']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsNIHDatasetConfig_B1(EmbeddingsNIHDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "NIH", f) for f in 
                        ["batch1"]]

class EmbeddingsNIHDatasetConfig_B2(EmbeddingsNIHDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "NIH", f) for f in 
                        ["batch2"]]

class EmbeddingsNIHDatasetConfig_B3(EmbeddingsNIHDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "NIH", f) for f in 
                        ["batch3"]]

class EmbeddingsNIHDatasetConfig_WT(EmbeddingsNIHDatasetConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = ['WT']

class EmbeddingsNIHDatasetConfig_WT_B1(EmbeddingsNIHDatasetConfig_WT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "NIH", f) for f in 
                        ["batch1"]]

class EmbeddingsNIHDatasetConfig_WT_B2(EmbeddingsNIHDatasetConfig_WT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "NIH", f) for f in 
                        ["batch2"]]

class EmbeddingsNIHDatasetConfig_WT_B3(EmbeddingsNIHDatasetConfig_WT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "NIH", f) for f in 
                        ["batch3"]]


## NeuronsDay18 ##
class EmbeddingsDay18B1DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay18", f) for f in 
                        ["batch1"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsDay18B2DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay18", f) for f in 
                        ["batch2"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neuronsDay18'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True