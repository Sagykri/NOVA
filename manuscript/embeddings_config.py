import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig


class EmbeddingsOpenCellDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["OpenCell"]]
        self.SPLIT_DATA = True
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.EXPERIMENT_TYPE = 'Opencell'

class EmbeddingsOpenCellFineTuneDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["OpenCell"]]
        self.SPLIT_DATA = False
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.EXPERIMENT_TYPE = 'Opencell'

class EmbeddingsU2OSDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                        ["U2OS_spd_format"]]
        
        self.SPLIT_DATA = False        
        self.CELL_LINES = ['U2OS']
        self.EXPERIMENT_TYPE = 'U2OS'
        self.MARKERS = ['G3BP1', 'DCP1A', 'Phalloidin', 'DAPI']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsB78DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]
        
        self.SPLIT_DATA = True
        self.EXPERIMENT_TYPE = 'neurons'    
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsB6DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'    
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsB9DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'    
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsB3DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch3"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'    
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsB4DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch4"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'    
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsB5DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch5"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'    
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsdNLSB2DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch2"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsdNLSB3DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch3"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsdNLSB4DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch4"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsdNLSB5DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", "deltaNLS", f) for f in 
                        ["batch5"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'deltaNLS'
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
       
class EmbeddingsB78PretrainDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7", "batch8"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons'    
        self.MARKERS_TO_EXCLUDE = ['TIA1']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsDay18B1DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch1"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        
class EmbeddingsDay18B2DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Opera18DaysReimaged", f) for f in 
                        ["batch2"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'neurons_d18'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsAlyssaCoyneDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "AlyssaCoyne", "MOmaps_iPSC_patients_TDP43_PB_CoyneLab_7tiles", f) for f in 
                        ["batch1"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_7tiles'    
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsNIHDay8DatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "NIH", f) for f in 
                        ["batch1", "batch2", "batch3"]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'NIH_d8'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsFUNOVADatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2","Batch3","Batch4", ]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'funova'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsFUNOVADatasetConfigMinMax(EmbeddingsConfig):
    def __init__(self):
        super().__init__()
        self.PROCESSED_FOLDER_ROOT = '/home/labs/hornsteinlab/Collaboration/FUNOVA/input/images/processed_minmax/'
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1", "Batch2","Batch3","Batch4", ]]
        
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'funova_minmax'    
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsFUNOVADatasetConfigFinetuned(EmbeddingsConfig):
    def __init__(self):
        ## Run Batch 1 with split_data = True and rest of the batches with split_data = False
        super().__init__()
        self.PROCESSED_FOLDER_ROOT = '/home/labs/hornsteinlab/Collaboration/FUNOVA/input/images/processed/'
        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["Batch1" ]] #"Batch2","Batch3","Batch4",
        
        self.SPLIT_DATA = True
        self.EXPERIMENT_TYPE = 'funova' 
        self.MARKERS_TO_EXCLUDE = None
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True