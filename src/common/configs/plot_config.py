import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.configs.base_config import BaseConfig


class PlotConfig(BaseConfig):
    def __init__(self):
        
        super().__init__()
        
        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

        # For label mapping in plots
        self.LABEL_MAPPINGS_ALIAS_KEY = 'alias'
        self.LABEL_MAPPINGS_COLOR_KEY = 'color'
        
        self.LABEL_MAPPINGS_CONDITION = {
            'Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: '- Stress', self.LABEL_MAPPINGS_COLOR_KEY: '#52C5D5'},
            'stress': {self.LABEL_MAPPINGS_ALIAS_KEY: '+ Stress', self.LABEL_MAPPINGS_COLOR_KEY: '#F7810F'},
        }
        
        self.LABEL_MAPPINGS_ALS = {
            'WT_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Wild-Type', self.LABEL_MAPPINGS_COLOR_KEY: '#37AFD7'},
            'FUSHeterozygous_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'FUS Heterozygous', self.LABEL_MAPPINGS_COLOR_KEY: '#AB7A5B'},
            'FUSHomozygous_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'FUS Homozygous', self.LABEL_MAPPINGS_COLOR_KEY: '#78491C'},
            'FUSRevertant_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'FUS Revertant', self.LABEL_MAPPINGS_COLOR_KEY: '#C8C512'},
            'OPTN_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'OPTN', self.LABEL_MAPPINGS_COLOR_KEY: '#FF98BB'},
            'TBK1_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'TBK1', self.LABEL_MAPPINGS_COLOR_KEY: '#319278'},
            'SCNA_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'SCNA', self.LABEL_MAPPINGS_COLOR_KEY: 'black'},
            'SNCA_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'SNCA', self.LABEL_MAPPINGS_COLOR_KEY: 'black'},
            'TDP43_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'TDP43', self.LABEL_MAPPINGS_COLOR_KEY: '#A8559E'},
        }
        self.LABEL_MAPPINGS_ALS['WT'] = self.LABEL_MAPPINGS_ALS['WT_Untreated']
        self.LABEL_MAPPINGS_ALS['FUSHeterozygous'] = self.LABEL_MAPPINGS_ALS['FUSHeterozygous_Untreated']
        self.LABEL_MAPPINGS_ALS['FUSHomozygous'] = self.LABEL_MAPPINGS_ALS['FUSHomozygous_Untreated']
        self.LABEL_MAPPINGS_ALS['FUSRevertant'] = self.LABEL_MAPPINGS_ALS['FUSRevertant_Untreated']
        self.LABEL_MAPPINGS_ALS['OPTN'] = self.LABEL_MAPPINGS_ALS['OPTN_Untreated']
        self.LABEL_MAPPINGS_ALS['TBK1'] = self.LABEL_MAPPINGS_ALS['TBK1_Untreated']
        self.LABEL_MAPPINGS_ALS['SCNA'] = self.LABEL_MAPPINGS_ALS['SCNA_Untreated']
        self.LABEL_MAPPINGS_ALS['SNCA'] = self.LABEL_MAPPINGS_ALS['SNCA_Untreated']
        self.LABEL_MAPPINGS_ALS['TDP43'] = self.LABEL_MAPPINGS_ALS['TDP43_Untreated']

        self.LABEL_MAPPINGS_CONDITION_AND_ALS = {
            'WT_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Wild-Type', self.LABEL_MAPPINGS_COLOR_KEY: '#37AFD7'},
            'WT_stress': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Wild-Type + Stress', self.LABEL_MAPPINGS_COLOR_KEY: '#F7810F'},
            'FUSHeterozygous_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'FUS Heterozygous', self.LABEL_MAPPINGS_COLOR_KEY: '#AB7A5B'},
            'FUSHomozygous_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'FUS Homozygous', self.LABEL_MAPPINGS_COLOR_KEY: '#78491C'},
            'FUSRevertant_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'FUS Revertant', self.LABEL_MAPPINGS_COLOR_KEY: '#C8C512'},
            'OPTN_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'OPTN', self.LABEL_MAPPINGS_COLOR_KEY: '#FF98BB'},
            'TBK1_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'TBK1', self.LABEL_MAPPINGS_COLOR_KEY: '#319278'},
            'SCNA_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'SCNA', self.LABEL_MAPPINGS_COLOR_KEY: 'black'},
            'SNCA_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'SNCA', self.LABEL_MAPPINGS_COLOR_KEY: 'black'},
            'TDP43_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'TDP43', self.LABEL_MAPPINGS_COLOR_KEY: '#A8559E'},
        }
        
        self.LABEL_MAPPINGS_DOX = {
            'WT_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Wild-Type', self.LABEL_MAPPINGS_COLOR_KEY: '#2FA0C1'},
            'TDP43_Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'TDP43dNLS, -Dox', self.LABEL_MAPPINGS_COLOR_KEY: '#6BAD31'},
            'TDP43_dox': {self.LABEL_MAPPINGS_ALIAS_KEY: 'TDP43dNLS, +Dox', self.LABEL_MAPPINGS_COLOR_KEY: '#90278E'},
        }
        
        self.LABEL_MAPPINGS_MARKERS = {
            'NCL': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Nucleolus', self.LABEL_MAPPINGS_COLOR_KEY: '#18E4CF'},
            'FUS': {self.LABEL_MAPPINGS_ALIAS_KEY: 'hnRNP complex', self.LABEL_MAPPINGS_COLOR_KEY: '#9968CB'},
            'DAPI': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Nucleus', self.LABEL_MAPPINGS_COLOR_KEY: '#AFBDFF'},
            'PML': {self.LABEL_MAPPINGS_ALIAS_KEY: 'PML bodies', self.LABEL_MAPPINGS_COLOR_KEY: '#F08F21'},
            'ANXA11': {self.LABEL_MAPPINGS_ALIAS_KEY: 'ANXA11 granules', self.LABEL_MAPPINGS_COLOR_KEY: '#37378D'},
            'NONO': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Paraspeckles', self.LABEL_MAPPINGS_COLOR_KEY: '#4343FE'},
            'TDP43': {self.LABEL_MAPPINGS_ALIAS_KEY: 'TDP43 granules', self.LABEL_MAPPINGS_COLOR_KEY: '#06A0E9'},
            'PEX14': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Peroxisome', self.LABEL_MAPPINGS_COLOR_KEY: '#168FB2'},
            'Calreticulin': {self.LABEL_MAPPINGS_ALIAS_KEY: 'ER', self.LABEL_MAPPINGS_COLOR_KEY: '#12F986'},
            'Phalloidin': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Cytoskeleton', self.LABEL_MAPPINGS_COLOR_KEY: '#921010'},
            'mitotracker': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Mitochondria', self.LABEL_MAPPINGS_COLOR_KEY: '#898700'},
            'TOMM20': {self.LABEL_MAPPINGS_ALIAS_KEY: 'MOM', self.LABEL_MAPPINGS_COLOR_KEY: '#66CDAA'},
            'PURA': {self.LABEL_MAPPINGS_ALIAS_KEY: 'PURA granules', self.LABEL_MAPPINGS_COLOR_KEY: '#AF8215'},
            'CLTC': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Coated Vesicles', self.LABEL_MAPPINGS_COLOR_KEY: '#32AC0E'},
            'KIF5A': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Transport machinery', self.LABEL_MAPPINGS_COLOR_KEY: '#ACE142'},
            'SCNA': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Presynapse', self.LABEL_MAPPINGS_COLOR_KEY: '#DEDB23'},
            'SNCA': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Presynapse', self.LABEL_MAPPINGS_COLOR_KEY: '#DEDB23'},
            'CD41': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Integrin puncta', self.LABEL_MAPPINGS_COLOR_KEY: '#F04521'},
            'SQSTM1': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Autophagosomes', self.LABEL_MAPPINGS_COLOR_KEY: '#FFBF0D'},
            'G3BP1': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Stress Granules', self.LABEL_MAPPINGS_COLOR_KEY: '#A80358'},
            'GM130': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Golgi', self.LABEL_MAPPINGS_COLOR_KEY: '#D257EA'},
            'LAMP1': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Lysosome', self.LABEL_MAPPINGS_COLOR_KEY: '#E6A9EA'},
            'DCP1A': {self.LABEL_MAPPINGS_ALIAS_KEY: 'P-Bodies', self.LABEL_MAPPINGS_COLOR_KEY: '#F0A3A3'},
            'NEMO': {self.LABEL_MAPPINGS_ALIAS_KEY: 'NEMO granules', self.LABEL_MAPPINGS_COLOR_KEY: '#EF218B'},
            'PSD95': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Postsynapse', self.LABEL_MAPPINGS_COLOR_KEY: '#F1CBDD'},            
            
            'FMRP': {self.LABEL_MAPPINGS_ALIAS_KEY: 'FMRP', self.LABEL_MAPPINGS_COLOR_KEY: 'gray'},
            'TDP43B': {self.LABEL_MAPPINGS_ALIAS_KEY: 'TDP43 granules 1', self.LABEL_MAPPINGS_COLOR_KEY: '#06A0E9'},
            'TDP43N': {self.LABEL_MAPPINGS_ALIAS_KEY: 'TDP43 granules 2', self.LABEL_MAPPINGS_COLOR_KEY: '#06A0E9'},
        }
        
        self.LABEL_MAPPINGS_CONDITION_FUS = {
            'Untreated': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Untreated', self.LABEL_MAPPINGS_COLOR_KEY: '#52C5D5'},
            'BMAA': {self.LABEL_MAPPINGS_ALIAS_KEY: 'BMAA', self.LABEL_MAPPINGS_COLOR_KEY: '#90278E'},
            'Cisplatin': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Cisplatin', self.LABEL_MAPPINGS_COLOR_KEY: '#AB7A5B'},
            'Colchicine': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Colchicine', self.LABEL_MAPPINGS_COLOR_KEY: '#FF98BB'},
            'DMSO': {self.LABEL_MAPPINGS_ALIAS_KEY: 'DMSO', self.LABEL_MAPPINGS_COLOR_KEY: '#F08F21'},
            'Etoposide': {self.LABEL_MAPPINGS_ALIAS_KEY: 'Etoposide', self.LABEL_MAPPINGS_COLOR_KEY: '#37378D'},
            'MG132': {self.LABEL_MAPPINGS_ALIAS_KEY: 'MG132', self.LABEL_MAPPINGS_COLOR_KEY: '#4343FE'},
            'ML240': {self.LABEL_MAPPINGS_ALIAS_KEY: 'ML240', self.LABEL_MAPPINGS_COLOR_KEY: '#06A0E9'},
            'NMS873': {self.LABEL_MAPPINGS_ALIAS_KEY: 'NMS873', self.LABEL_MAPPINGS_COLOR_KEY: '#168FB2'},
            'SA': {self.LABEL_MAPPINGS_ALIAS_KEY: 'SA', self.LABEL_MAPPINGS_COLOR_KEY: '#F7810F'},
        }
        
        # Set the UMAPS mapping here!
        self.LABEL_MAPPINGS = self.LABEL_MAPPINGS_ALS
        