import datetime
import logging
import os
import random
import numpy as np

from src.common.lib.utils import init_logging


class BaseConfig():
    def __init__(self):
        __now = datetime.datetime.now()
        
        @SEED.setter
        def SEED(self, value):
            self.SEED = value
            np.random.seed(self.SEED)
            random.seed(self.SEED)
                        
        self.SEED = 1
        np.random.seed(self.SEED)
        random.seed(self.SEED)

        self.HOME_FOLDER = os.environ['MOMAPS_HOME']

        # Data
        self.RAW_FOLDER_ROOT = os.path.join(self.HOME_FOLDER, "input", "images", "raw")
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.HOME_FOLDER, "input", "images", "processed")
        
        # Output
        self.CONFIGS_USED_FOLDER = os.path.join(self.HOME_FOLDER, "outputs", "configs_used", __now.strftime("%d%m%y_%H%M%S_%f"))
        
        
        # Model
        self.PRETRAINED_MODEL_PATH = os.path.join(self.MODEL_FOLDER, "pretrained_model.h5")
    
        
        # Logs
        @LOGS_FOLDER.setter
        def LOGS_FOLDER(self, path):
            self.LOGS_FOLDER = path
            
            log_file_path = os.path.join(self.LOGS_FOLDER, __now.strftime("%d%m%y_%H%M%S_%f"))
            init_logging(log_file_path)
            logging.info(f"[{self.__class__.__name__}] Init")
            
        self.LOGS_FOLDER = os.path.join(self.HOME_FOLDER, 'logs')
        
        
        # For plotting
        self.TERM_UNSTRESSED = "_unstressed"
        self.TERM_STRESSED = "_stressed"
        self.TERM_WT = "_WT"
        self.TERM_TDP43 = "_TDP43"
        self.TERM_FUS = "_FUS"
        self.TERM_OPTN = "_OPTN"
        self.TERM_TBK1 = "_TBK1"
        self.TERM_WT_UNSTRESS = "_WT_unstressed"
        self.TERM_TDP43_UNSTRESS = "_TDP43_unstressed"
        self.TERM_FUS_UNSTRESS = "_FUS_unstressed"
        self.TERM_OPTN_UNSTRESS = "_OPTN_unstressed"
        self.TERM_TBK1_UNSTRESS = "_TBK1_unstressed"
        self.TERM_WT_STRESS = "_WT_stressed"
        self.TERM_TDP43_STRESS = "_TDP43_stressed"
        self.TERM_FUS_STRESS = "_FUS_stressed"
        self.TERM_OPTN_STRESS = "_OPTN_stressed"
        self.TERM_TBK1_STRESS = "_TBK1_stressed"
        self.TERM_WT_MICROGLIA = "_WT_microglia"
        self.TERM_FUS_MICROGLIA = "_FUS_microglia"
        self.TERM_TDP43_MICROGLIA = "_TDP43_microglia"
        self.TERM_OPTN_MICROGLIA = "_OPTN_microglia"
        self.TERM_WT_NEURONS = "_WT_neurons"
        self.TERM_FUS_NEURONS = "_FUS_neurons"
        self.TERM_TDP43_NEURONS = "_TDP43_neurons"
        self.TERM_OPTN_NEURONS = "_OPTN_neurons"

        self.LEGEND_UNSTRESSED = "Unstressed"
        self.LEGEND_STRESSED = "Stressed"
        self.LEGEND_WT = "WT"
        self.LEGEND_TDP43 = "TDP43"
        self.LEGEND_FUS = "FUS"
        self.LEGEND_OPTN = "OPTN"
        self.LEGEND_TBK1 = "TBK1"
        self.LEGEND_WT_UNSTRESS = "WT Unstressed"
        self.LEGEND_TDP43_UNSTRESS = "TDP43 Unstressed"
        self.LEGEND_FUS_UNSTRESS = "FUS Unstressed"
        self.LEGEND_WT_UNSTRESS = "WT Unstressed"
        self.LEGEND_OPTN_UNSTRESS = "OPTN Unstressed"
        self.LEGEND_TBK1_UNSTRESS = "TBK1 Unstressed"
        self.LEGEND_WT_STRESS = "WT Stressed"
        self.LEGEND_TDP43_STRESS = "TDP43 Stressed"
        self.LEGEND_FUS_STRESS = "FUS Stressed"
        self.LEGEND_WT_STRESS = "WT Stressed"
        self.LEGEND_OPTN_STRESS = "OPTN Stressed"
        self.LEGEND_TBK1_STRESS = "TBK1 Stressed"
        self.LEGEND_WT_MICROGLIA = "WT microglia"
        self.LEGEND_FUS_MICROGLIA = "FUS microglia"
        self.LEGEND_TDP43_MICROGLIA = "TDP43 microglia"
        self.LEGEND_OPTN_MICROGLIA = "OPTN microglia"
        self.LEGEND_WT_NEURONS = "WT neurons"
        self.LEGEND_FUS_NEURONS = "FUS neurons"
        self.LEGEND_TDP43_NEURONS = "TDP43 neurons"
        self.LEGEND_OPTN_NEURONS = "OPTN neurons"

        self.TEMR_LEGEND_MAPPING = {
            self.TERM_UNSTRESSED: self.LEGEND_UNSTRESSED,
            self.TERM_STRESSED: self.LEGEND_STRESSED,
            self.TERM_WT: self.LEGEND_WT,
            self.TERM_TDP43: self.LEGEND_TDP43,
            self.TERM_FUS: self.LEGEND_FUS,
            self.TERM_OPTN: self.LEGEND_OPTN,
            self.TERM_TBK1: self.LEGEND_TBK1,
            self.TERM_WT_UNSTRESS: self.LEGEND_WT_UNSTRESS,
            self.TERM_TDP43_UNSTRESS: self.LEGEND_TDP43_UNSTRESS,
            self.TERM_FUS_UNSTRESS: self.LEGEND_FUS_UNSTRESS,
            self.TERM_OPTN_UNSTRESS: self.LEGEND_OPTN_UNSTRESS,
            self.TERM_TBK1_UNSTRESS: self.LEGEND_TBK1_UNSTRESS,
            self.TERM_WT_STRESS: self.LEGEND_WT_STRESS,
            self.TERM_TDP43_STRESS: self.LEGEND_TDP43_STRESS,
            self.TERM_FUS_STRESS: self.LEGEND_FUS_STRESS,
            self.TERM_OPTN_STRESS: self.LEGEND_OPTN_STRESS,
            self.TERM_TBK1_STRESS: self.LEGEND_TBK1_STRESS,
            self.TERM_WT_MICROGLIA: self.LEGEND_WT_MICROGLIA,
            self.TERM_FUS_MICROGLIA: self.LEGEND_FUS_MICROGLIA,
            self.TERM_TDP43_MICROGLIA: self.LEGEND_TDP43_MICROGLIA,
            self.TERM_OPTN_MICROGLIA: self.LEGEND_OPTN_MICROGLIA,
            self.TERM_WT_NEURONS: self.LEGEND_WT_NEURONS,
            self.TERM_FUS_NEURONS: self.LEGEND_FUS_NEURONS,
            self.TERM_TDP43_NEURONS: self.LEGEND_TDP43_NEURONS,
            self.TERM_OPTN_NEURONS: self.LEGEND_OPTN_NEURONS

        }

        self.COLORS_MAPPING = {
            self.TERM_UNSTRESSED: 'cyan',
            self.TERM_STRESSED: 'orange',
            self.TERM_WT: 'green',
            self.TERM_TDP43: 'blue',
            self.TERM_FUS: 'red',
            self.TERM_OPTN: 'yellow',
            self.TERM_TBK1: 'brown',
            self.TERM_WT_UNSTRESS: "cyan",
            self.TERM_TDP43_UNSTRESS: "green",
            self.TERM_FUS_UNSTRESS: "purple",
            self.TERM_OPTN_UNSTRESS: "yellow",
            self.TERM_TBK1_UNSTRESS: "brown",
            self.TERM_WT_STRESS: "orange",
            self.TERM_TDP43_STRESS: "orange",
            self.TERM_FUS_STRESS: "black",
            self.TERM_OPTN_STRESS: "cyan",
            self.TERM_TBK1_STRESS: "pink",
            self.TERM_WT_MICROGLIA: "lime",
            self.TERM_FUS_MICROGLIA: "magenta",
            self.TERM_TDP43_MICROGLIA: "cyan",
            self.TERM_OPTN_MICROGLIA: "orange",
            self.TERM_WT_NEURONS: "green",
            self.TERM_FUS_NEURONS: "red",
            self.TERM_TDP43_NEURONS: "blue",
            self.TERM_OPTN_NEURONS: "yellow"
        }
        
        