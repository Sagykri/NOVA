import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME")) 

import datetime
import logging
import random
import numpy as np


from src.common.lib.utils import init_logging


class BaseConfig():
    def __init__(self):
        __now = datetime.datetime.now()
        
        self.__SEED = 1
        np.random.seed(self.__SEED)
        random.seed(self.__SEED)

        self.HOME_FOLDER = os.environ['MOMAPS_HOME']
        self.HOME_DATA_FOLDER = os.environ['MOMAPS_DATA_HOME'] \
                                    if 'MOMAPS_DATA_HOME' in os.environ \
                                    else os.path.join(self.HOME_FOLDER, "input")

        # Data
        self.RAW_FOLDER_ROOT = os.path.join(self.HOME_DATA_FOLDER, "images", "raw")
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.HOME_DATA_FOLDER, "images", "processed")
        
        # Precaution - raw and processed folders can't be the same one!
        assert self.RAW_FOLDER_ROOT != self.PROCESSED_FOLDER_ROOT, f"RAW_FOLDER_ROOT == PROCESSED_FOLDER_ROOT, {self.RAW_FOLDER_ROOT}"
                
        # Output
        self.OUTPUTS_FOLDER = os.path.join(self.HOME_FOLDER, "outputs")
        self.CONFIGS_USED_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "configs_used", __now.strftime("%d%m%y_%H%M%S_%f"))
        
        
        # Model
        self.PRETRAINED_MODEL_PATH = None
        
        # Logs
        self.__LOGS_FOLDER = os.path.join(self.HOME_FOLDER, 'logs')
        
        # For plotting
        self.UMAP_MAPPINGS_ALIAS_KEY = 'alias'
        self.UMAP_MAPPINGS_COLOR_KEY = 'color'
        
        self.UMAP_MAPPINGS_CONDITION = {
            'Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: '- Stress', self.UMAP_MAPPINGS_COLOR_KEY: '#52C5D5'},
            'stress': {self.UMAP_MAPPINGS_ALIAS_KEY: '+ Stress', self.UMAP_MAPPINGS_COLOR_KEY: '#F7810F'},
        }
        
        self.UMAP_MAPPINGS_ALS = {
            'WT_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Wild-Type', self.UMAP_MAPPINGS_COLOR_KEY: '#52C5D5'},
            'WT_stress': {self.UMAP_MAPPINGS_ALIAS_KEY: 'WT Stress', self.UMAP_MAPPINGS_COLOR_KEY: '#F7810F'},
            'FUSHeterozygous_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'FUS Heterozygous', self.UMAP_MAPPINGS_COLOR_KEY: '#A86343'},
            'FUSHomozygous_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'FUS Homozygous', self.UMAP_MAPPINGS_COLOR_KEY: '#6E3B0B'},
            'FUSRevertant_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'FUS Revertant', self.UMAP_MAPPINGS_COLOR_KEY: '#C7A036'},
            'OPTN_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'OPTN', self.UMAP_MAPPINGS_COLOR_KEY: '#7BA89C'},
            'TBK1_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TBK1', self.UMAP_MAPPINGS_COLOR_KEY: '#A89689'},
            'SCNA_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'SCNA', self.UMAP_MAPPINGS_COLOR_KEY: 'black'},
            'TDP43_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP43', self.UMAP_MAPPINGS_COLOR_KEY: '#93749E'},
        }
        
        self.UMAP_MAPPINGS_DOX = {
            'WT_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Wild-Type', self.UMAP_MAPPINGS_COLOR_KEY: '#2FA0C1'},
            'TDP43_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP43dNLS, -Dox', self.UMAP_MAPPINGS_COLOR_KEY: '#6BAD31'},
            'TDP43_dox': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP43dNLS, +Dox', self.UMAP_MAPPINGS_COLOR_KEY: '#90278E'},
        }
        
        self.UMAP_MAPPINGS_MARKERS = {
            'NCL': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Nucleolus', self.UMAP_MAPPINGS_COLOR_KEY: 'red'},
            'FUS': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Heterogeneous Nuclear Ribonucleoprotein (hnRNP) Complexes', self.UMAP_MAPPINGS_COLOR_KEY: 'salmon'},
            'DAPI': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Nucleus', self.UMAP_MAPPINGS_COLOR_KEY: 'mediumaquamarine'},
            'PML': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Promyelocytic Leukaemia (PML) Nuclear Bodies', self.UMAP_MAPPINGS_COLOR_KEY: 'darkred'},
            'ANXA11': {self.UMAP_MAPPINGS_ALIAS_KEY: 'ANXA11-granules', self.UMAP_MAPPINGS_COLOR_KEY: 'darkorange'},
            'NONO': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Paraspeckles', self.UMAP_MAPPINGS_COLOR_KEY: 'orange'},
            'TDP43': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP-43-granules', self.UMAP_MAPPINGS_COLOR_KEY: 'gold'},
            'PEX14': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Peroxisomes', self.UMAP_MAPPINGS_COLOR_KEY: 'black'},
            'Calreticulin': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Endoplasmic Reticulum (ER)', self.UMAP_MAPPINGS_COLOR_KEY: 'saddlebrown'},
            'Phalloidin': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Actin Cytoskeleton', self.UMAP_MAPPINGS_COLOR_KEY: 'darkviolet'},
            'mitotracker': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Mitochondria', self.UMAP_MAPPINGS_COLOR_KEY: 'pink'},
            'TOMM20': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Mitochondria Outer Membrane', self.UMAP_MAPPINGS_COLOR_KEY: 'palevioletred'},
            'PURA': {self.UMAP_MAPPINGS_ALIAS_KEY: 'PURA-granules', self.UMAP_MAPPINGS_COLOR_KEY: 'deeppink'},
            'CLTC': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Coated Vesicles', self.UMAP_MAPPINGS_COLOR_KEY: 'magenta'},
            'KIF5A': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Microtubule-Associated Transport Machinery', self.UMAP_MAPPINGS_COLOR_KEY: 'darkmagenta'},
            'SCNA': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Presynaptic Terminals', self.UMAP_MAPPINGS_COLOR_KEY: 'navy'},
            'CD41': {self.UMAP_MAPPINGS_ALIAS_KEY: 'CD41-granules', self.UMAP_MAPPINGS_COLOR_KEY: 'royalblue'},
            'SQSTM1': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Autophagosomes', self.UMAP_MAPPINGS_COLOR_KEY: 'deepskyblue'},
            'G3BP1': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Stress Granules', self.UMAP_MAPPINGS_COLOR_KEY: 'olive'},
            'GM130': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Golgi Apparatus', self.UMAP_MAPPINGS_COLOR_KEY: 'olivedrab'},
            'LAMP1': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Lysosomes', self.UMAP_MAPPINGS_COLOR_KEY: 'lime'},
            'DCP1A': {self.UMAP_MAPPINGS_ALIAS_KEY: 'P-Bodies', self.UMAP_MAPPINGS_COLOR_KEY: 'seagreen'},
            'NEMO': {self.UMAP_MAPPINGS_ALIAS_KEY: 'NEMO Condensates', self.UMAP_MAPPINGS_COLOR_KEY: 'darkgreen'},
            'PSD95': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Post-Synaptic Subcompartments', self.UMAP_MAPPINGS_COLOR_KEY: 'green'},            
            'FMRP': {self.UMAP_MAPPINGS_ALIAS_KEY: 'FMRP', self.UMAP_MAPPINGS_COLOR_KEY: 'gray'}
        }
        
        # Set the UMAPS mapping here!
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        
    @property
    def SEED(self):
        return self.__SEED
    
    @SEED.setter
    def SEED(self, value):
        self.__SEED = value
        np.random.seed(self.__SEED)
        random.seed(self.__SEED)
        
    @property
    def LOGS_FOLDER(self):
        return self.__LOGS_FOLDER
    
    @LOGS_FOLDER.setter
    def LOGS_FOLDER(self, path):
        if logging.getLogger().hasHandlers():
            return
    
        self.__LOGS_FOLDER = path
        __now = datetime.datetime.now()
        jobid = os.getenv('LSB_JOBID')
        log_file_path = os.path.join(self.__LOGS_FOLDER, __now.strftime("%d%m%y_%H%M%S_%f") + f'_{jobid}.log')
        if not os.path.exists(self.__LOGS_FOLDER):
            os.makedirs(self.__LOGS_FOLDER)
        init_logging(log_file_path)
        logging.info(f"[{self.__class__.__name__}] Init (log path: {log_file_path}; JOBID: {jobid})")
        logging.info(f"[{self.__class__.__name__}] MOMAPS_HOME={self.HOME_FOLDER}, MOMAPS_DATA_HOME={self.HOME_DATA_FOLDER}")
