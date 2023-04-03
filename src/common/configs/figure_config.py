import os
from src.common.configs.base_config import BaseConfig

class FigureConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.HOME_FIGURES_FOLDER = os.path.join(self.HOME_FOLDER, "figures")
        self.OUTPUT_DIR = None
        self.FIGURES = []
        
        
        # self.input_folders = ["./data/processed/220814_neurons",
        #          "./data/processed/220818_neurons",
        #          "./data/processed/220831_neurons",
        #          "./data/processed/220908",
        #          "./data/processed/220914"]
        # self.input_folders_cytoself = ["./data/processed/220714"]
        # self.input_folders_microglia = ["./data/processed/microglia_new"]
        # self.input_folders_combined = input_folders + input_folders_microglia
        # self.input_folders_pertrubations = ["./data/processed/Perturbations"]
        # self.input_folders_pertrubations_spd = ["./data/processed/SpinningDisk/Perturbations"]
        # self.input_folders_pertrubations_spd2 = ["./data/processed/spd2/SpinningDisk/Perturbations"]

        # self.cytoself_model_path = self.PRETRAINED_MODEL_PATH
        # self.neuroself_model_path = os.path.join(self.MODEL_FOLDER, "MODEL18_model_weights.0040.h5")
        # self.imgself_model_path = os.path.join("./model_outputs", "imgself", "model_ep0043_imgself_fixed.h5")
        # self.combined_model_path = os.path.join("./model_outputs", "comb_model_trying", "model_ep0050_default_fixed.h5")

        # self.groups_terms_condition = [self.TERM_UNSTRESSED, self.TERM_STRESSED]
        # self.groups_terms_line = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN, self.TERM_TBK1]
        # self.groups_terms_line_microglia = [self.TERM_WT, self.TERM_TDP43, self.TERM_FUS, self.TERM_OPTN]
        # self.groups_terms_type = ["_WT_microglia", "_WT_neurons", "_TDP43_microglia", "_TDP43_neurons",
        #                     "_FUS_microglia", "_FUS_neurons", "_OPTN_microglia", "_OPTN_neurons"]

