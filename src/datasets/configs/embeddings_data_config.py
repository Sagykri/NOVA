import os

import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.dataset_config import DatasetConfig

# TODO: (210823) CLEAN!

class EmbeddingsExampleDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.SPLIT_DATA = False
        self.CELL_LINES = ['WT', 'TDP43', 'FUSHeterozygous', 'FUSRevertant']
        
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        #self.MARKERS = ['TOMM20','mitotracker'] #['FUS']
    
        # Should calculate the embeddings or load them from the files?
        self.CALCULATE_EMBEDDINGS = True
        # EMBEDDINGS_TYPE_TO_LOAD - relevent only when CALCULATE_EMBEDDINGS=False
        # Which type to load: ['trainset', 'valtest', 'testset', 'all']
        self.EMBEDDINGS_TYPE_TO_LOAD = 'testset'

class EmbeddingsB6DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch6"]]
        
        self.CALCULATE_EMBEDDINGS = False
        self.SPLIT_DATA = False
        # self.CELL_LINES = ['WT', 'TDP43', 'FUSHeterozygous', 'FUSRevertant']
        
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        all_markers = ['ANXA11', 'Calreticulin', 'CD41', 'CLTC', 'DAPI', 'DCP1A', 'FMRP', 'FUS', 'G3BP1', 'GM130', 'KIF5A', 'LAMP1',
                        'mitotracker', 'NCL', 'NEMO', 'NONO', 'PEX14', 'Phalloidin', 'PML', 'PSD95', 'PURA', 'SCNA', 'SQSTM1', 'TDP43',
                        'TIA1', 'TOMM20']
        self.MARKERS = all_markers[:20] 
        #self.MARKERS = ['TOMM20','mitotracker'] #['FUS']

class EmbeddingsB78DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7_16bit", "batch8_16bit"]]
        
        self.SPLIT_DATA = True#False        
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['TOMM20', 'mitotracker', 'FUS'] 
        
class EmbeddingsB7NODSDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch7_16bit_no_downsample"]]
        
        self.CALCULATE_EMBEDDINGS = True
        self.SPLIT_DATA = True  
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CELL_LINES = ['WT', 'FUSHeterozygous', 'FUSRevertant', 'TDP43']
        self.CONDITIONS = ['Untreated']
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['TOMM20', 'mitotracker', 'FUS'] 
                

class EmbeddingsB9DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9_16bit"]]
        
        self.CALCULATE_EMBEDDINGS = True
        self.SPLIT_DATA = True        
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CELL_LINES = ['WT', 'FUSHeterozygous', 'FUSRevertant', 'TDP43']
        self.CELL_LINES = ['WT', 'TDP43']#, 'TBK1', 'OPTN', 'FUSRevertant']#, 'FUSHeterozygous', 'FUSRevertant']

        self.CONDITIONS = ['Untreated']
        self.SAMPLE_PCT = 0.3 

        # all_markers = ['ANXA11', 'Calreticulin', 'CD41', 'CLTC', 'DAPI', 'DCP1A', 'FMRP', 'FUS', 'G3BP1', 'GM130', 'KIF5A', 'LAMP1',
        #                 'mitotracker', 'NCL', 'NEMO', 'NONO', 'PEX14', 'Phalloidin', 'PML', 'PSD95', 'PURA', 'SCNA', 'SQSTM1', 'TDP43',
        #                 'TIA1', 'TOMM20']
        self.MARKERS = ["Calreticulin", "NCL", "NONO", "PURA", "SQSTM1"] #all_markers[:20] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        
class EmbeddingsB9NoDownsamplingDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch9_16bit_no_downsample"]]
        
        self.CALCULATE_EMBEDDINGS = False
        self.SPLIT_DATA = False        
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CELL_LINES = ['WT']#, 'FUSHeterozygous', 'FUSRevertant', 'TDP43']
        # self.CONDITIONS = ['Untreated']
        self.MARKERS = ['G3BP1'] #['NONO', 'G3BP1', 'TOMM20', 'mitotracker', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        
class EmbeddingsOpenCellDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["OpenCell"]]
        
        self.SPLIT_DATA = True        
        self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['TOMM20', 'mitotracker', 'FUS'] 
        
        
class EmbeddingsB4DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch4"]]
        
        # self.SPLIT_DATA = False#True    
        # self.CELL_LINES = ['WT', 'TDP43', 'FUSHeterozygous', 'FUSRevertant']
        # self.CONDITIONS = ['Untreated']    
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['G3BP1']#, 'FMRP'] 
        
        self.SPLIT_DATA = False#True       
        # self.ADD_LINE_TO_LABEL = False
        # self.ADD_CONDITION_TO_LABEL = False 
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CELL_LINES = ['WT', 'TDP43', 'TBK1', 'OPTN', 'FUSRevertant']#, 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['FUS']#, 'FMRP'] 
        
        self.SAMPLE_PCT = 0.3 
        
class EmbeddingsB5DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch5"]]
        
        self.SPLIT_DATA = False#True       
        # self.ADD_LINE_TO_LABEL = False
        # self.ADD_CONDITION_TO_LABEL = False 
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CELL_LINES = ['WT', 'TDP43', 'TBK1', 'OPTN', 'FUSRevertant']#, 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['FUS']#, 'FMRP'] 
        
        self.SAMPLE_PCT = 0.3 

        
class EmbeddingsB3DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch3"]]
        
        # self.SPLIT_DATA = False#True        
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['FUS']#, 'FMRP'] 
        
        self.SPLIT_DATA = False#True       
        # self.ADD_LINE_TO_LABEL = False
        # self.ADD_CONDITION_TO_LABEL = False 
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CELL_LINES = ['WT', 'TDP43', 'TBK1', 'OPTN', 'FUSRevertant']#, 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['FUS']#, 'FMRP'] 
        
        self.SAMPLE_PCT = 0.3 
        
class EmbeddingsPertConfDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "Confocal", f) for f in 
                                ["Perturbations_spd_format"]]
        
        self.CALCULATE_EMBEDDINGS = True
        
        self.SPLIT_DATA = False        
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['FUS']#, 'FMRP']
        
        # self.SAMPLE_PCT = 0.1 
        
class EmbeddingsPertSPDDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["Perturbations"]]
                
        self.CALCULATE_EMBEDDINGS = True

        self.SPLIT_DATA = False        
        # self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['FUS']#, 'FMRP'] 
        
        # self.SAMPLE_PCT = 0.1
        
class EmbeddingsB2DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["batch2"]]
        
        self.CALCULATE_EMBEDDINGS = True

        self.SPLIT_DATA = True #True        
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CELL_LINES = ['WT', 'FUS', 'TDP43']
        self.CONDITIONS = ['unstressed']
        # self.MARKERS = ['G3BP1']
        # self.MARKERS = ['NONO', 'G3BP1', 'TOMM20', 'PURA', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        
class EmbeddingsB25DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["batch2_5_spd_format"]]
        
        self.SPLIT_DATA = False #True        
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CELL_LINES = ['WT', 'FUS', 'TDP43']
        # self.CONDITIONS = ['unstressed']
        # self.MARKERS = ['G3BP1']
        # self.MARKERS = ['NONO', 'G3BP1', 'TOMM20', 'PURA', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        
        
class EmbeddingsB2B25DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, f) for f in 
                        ["spd2/SpinningDisk/batch2","batch2_5_spd_format"]]
        
        self.ADD_BATCH_TO_LABEL = True
        self.CALCULATE_EMBEDDINGS = True
        self.SPLIT_DATA = False        
        # self.CELL_LINES = ['WT']#, 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        # self.CELL_LINES = ['WT', 'FUS', 'TDP43']
        self.CONDITIONS = ['unstressed']
        # self.MARKERS = ['G3BP1']
        # self.MARKERS = ['NONO', 'G3BP1', 'TOMM20', 'PURA', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        
        
class EmbeddingsMicrogliaB2DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", 'microglia', f) for f in 
                        ["batch2"]]
        
        self.CALCULATE_EMBEDDINGS = False
        self.SPLIT_DATA = False        
        self.CELL_LINES = ['WT', 'FUSHeterozygous', 'FUSHomozygous', 'FUSRevertant', 'OPTN', 'SCNA', 'TBK1']
        # self.CELL_LINES = ['WT', 'FUSHeterozygous', 'FUSRevertant', 'TDP43']
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['NONO', 'G3BP1', 'TOMM20', 'mitotracker', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        # self.SAMPLE_PCT = 0.1
        
class EmbeddingsMicrogliaB3DatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", 'microglia', f) for f in 
                        ["batch3"]]
        
        self.CALCULATE_EMBEDDINGS = True
        self.SPLIT_DATA = False      
        self.CELL_LINES = ['WT', 'FUSHeterozygous', 'FUSHomozygous', 'FUSRevertant', 'OPTN', 'TDP43', 'TBK1']
        # self.CELL_LINES = ['WT', 'FUSHeterozygous', 'FUSRevertant', 'TDP43']
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['NONO', 'G3BP1', 'TOMM20', 'mitotracker', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        # self.SAMPLE_PCT = 0.1
        
        
class EmbeddingsMicrogliaAllDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "spd2", "SpinningDisk", f) for f in 
                        ["microglia/batch2", 'microglia/batch3', 'microglia/batch4', 'microglia_LPS/batch1', 'microglia_LPS/batch2']]
        
        # self.CALCULATE_EMBEDDINGS = True
        self.SPLIT_DATA = False      
        # self.CELL_LINES = ['WT', 'FUSHeterozygous', 'FUSHomozygous', 'FUSRevertant', 'OPTN', 'TDP43', 'TBK1']
        # self.CELL_LINES = ['WT', 'FUSHeterozygous', 'FUSRevertant', 'TDP43']
        # self.CONDITIONS = ['Untreated']
        # self.MARKERS = ['NONO', 'G3BP1', 'TOMM20', 'mitotracker', 'FUS'] 
        # self.MARKERS_TO_EXCLUDE = ['TDP43','FUS']
        # self.SAMPLE_PCT = 0.1