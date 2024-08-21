import datetime
import logging
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer
from cytoself.analysis.analysis_opencell import AnalysisOpenCell
from cytoself.trainer.utils.plot_history import plot_history_cytoself
from cytoself.datamanager.base import DataManagerBase

from src.common.lib import metrics 
from src.common.lib.utils import get_if_exists

from src.common.configs.model_config import ModelConfig

class Trainer():

    def __init__(self, conf:ModelConfig):
        self.__set_params(conf)
        self.vit = self.__get_vit()        
    
    def __set_params(self, conf:ModelConfig):
        """Extracting params from the configuration

        Args:
            conf (ModelConfig): The configuration
        """
        self.conf = conf
        self.vit_version = get_if_exists(self.conf, 'VIT_VERSION', None)
        self.image_size = get_if_exists(self.conf, 'IMAGE_SIZE', None)
        self.patch_size = get_if_exists(self.conf, 'PATCH_SIZE', None)
        self.num_channels = get_if_exists(self.conf, 'NUM_CHANNELS', None)
        self.num_classes = get_if_exists(self.conf, 'NUM_CLASSES', None)