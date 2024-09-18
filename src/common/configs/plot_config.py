import sys
import os
from typing import Dict

sys.path.insert(1, os.getenv("MOMAPS_HOME")) 
from src.common.configs.base_config import BaseConfig

class PlotConfig(BaseConfig):
    """Config for plotting
    """
    def __init__(self):
        
        super().__init__()

        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
        self.UMAP_MAPPINGS_ALIAS_KEY = 'alias'
        self.UMAP_MAPPINGS_COLOR_KEY = 'color'
        
        self.COLOR_MAPPINGS_CONDITION: Dict[str, Dict[str,str]] = {
            'Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'- Stress', self.UMAP_MAPPINGS_COLOR_KEY:'#52C5D5'},
            'stress':{self.UMAP_MAPPINGS_ALIAS_KEY:'+ Stress', self.UMAP_MAPPINGS_COLOR_KEY:'#F7810F'},
        }
        
        self.COLOR_MAPPINGS_ALS: Dict[str, Dict[str,str]] = {
            'WT_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'Wild-Type', self.UMAP_MAPPINGS_COLOR_KEY:'#37AFD7'},
            'FUSHeterozygous_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'FUS Heterozygous', self.UMAP_MAPPINGS_COLOR_KEY:'#AB7A5B'},
            'FUSHomozygous_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'FUS Homozygous', self.UMAP_MAPPINGS_COLOR_KEY:'#78491C'},
            'FUSRevertant_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'FUS Revertant', self.UMAP_MAPPINGS_COLOR_KEY:'#C8C512'},
            'OPTN_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'OPTN', self.UMAP_MAPPINGS_COLOR_KEY:'#FF98BB'},
            'TBK1_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'TBK1', self.UMAP_MAPPINGS_COLOR_KEY:'#319278'},
            'SCNA_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'SCNA', self.UMAP_MAPPINGS_COLOR_KEY:'black'},
            'SNCA_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'SNCA', self.UMAP_MAPPINGS_COLOR_KEY:'black'},
            'TDP43_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP43', self.UMAP_MAPPINGS_COLOR_KEY:'#A8559E'},
        }
        self.COLOR_MAPPINGS_ALS['WT'] = self.COLOR_MAPPINGS_ALS['WT_Untreated']
        self.COLOR_MAPPINGS_ALS['FUSHeterozygous'] = self.COLOR_MAPPINGS_ALS['FUSHeterozygous_Untreated']
        self.COLOR_MAPPINGS_ALS['FUSHomozygous'] = self.COLOR_MAPPINGS_ALS['FUSHomozygous_Untreated']
        self.COLOR_MAPPINGS_ALS['FUSRevertant'] = self.COLOR_MAPPINGS_ALS['FUSRevertant_Untreated']
        self.COLOR_MAPPINGS_ALS['OPTN'] = self.COLOR_MAPPINGS_ALS['OPTN_Untreated']
        self.COLOR_MAPPINGS_ALS['TBK1'] = self.COLOR_MAPPINGS_ALS['TBK1_Untreated']
        self.COLOR_MAPPINGS_ALS['SCNA'] = self.COLOR_MAPPINGS_ALS['SCNA_Untreated']
        self.COLOR_MAPPINGS_ALS['SNCA'] = self.COLOR_MAPPINGS_ALS['SNCA_Untreated']
        self.COLOR_MAPPINGS_ALS['TDP43'] = self.COLOR_MAPPINGS_ALS['TDP43_Untreated']
        
        self.COLOR_MAPPINGS_DOX: Dict[str, Dict[str,str]] = {
            'WT_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'Wild-Type', self.UMAP_MAPPINGS_COLOR_KEY:'#2FA0C1'},
            'TDP43_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP43dNLS, -DOX', self.UMAP_MAPPINGS_COLOR_KEY:'#6BAD31'},
            'TDP43_dox':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP43dNLS, +DOX', self.UMAP_MAPPINGS_COLOR_KEY:'#90278E'}
        }

        self.COLOR_MAPPINGS_MARKERS: Dict[str, Dict[str,str]] = {
            'NCL':{self.UMAP_MAPPINGS_ALIAS_KEY:'Nucleolus', self.UMAP_MAPPINGS_COLOR_KEY:'#18E4CF'},
            'FUS':{self.UMAP_MAPPINGS_ALIAS_KEY:'hnRNP complex', self.UMAP_MAPPINGS_COLOR_KEY:'#9968CB'},
            'DAPI':{self.UMAP_MAPPINGS_ALIAS_KEY:'Nucleus', self.UMAP_MAPPINGS_COLOR_KEY:'#AFBDFF'},
            'PML':{self.UMAP_MAPPINGS_ALIAS_KEY:'PML bodies', self.UMAP_MAPPINGS_COLOR_KEY:'#F08F21'},
            'ANXA11':{self.UMAP_MAPPINGS_ALIAS_KEY:'ANXA11 granules', self.UMAP_MAPPINGS_COLOR_KEY:'#37378D'},
            'NONO':{self.UMAP_MAPPINGS_ALIAS_KEY:'Paraspeckles', self.UMAP_MAPPINGS_COLOR_KEY:'#4343FE'},
            'TDP43':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP43 granules', self.UMAP_MAPPINGS_COLOR_KEY:'#06A0E9'},
            'PEX14':{self.UMAP_MAPPINGS_ALIAS_KEY:'Peroxisome', self.UMAP_MAPPINGS_COLOR_KEY:'#168FB2'},
            'Calreticulin':{self.UMAP_MAPPINGS_ALIAS_KEY:'ER', self.UMAP_MAPPINGS_COLOR_KEY:'#12F986'},
            'Phalloidin':{self.UMAP_MAPPINGS_ALIAS_KEY:'Cytoskeleton', self.UMAP_MAPPINGS_COLOR_KEY:'#921010'},
            'mitotracker':{self.UMAP_MAPPINGS_ALIAS_KEY:'Mitochondria', self.UMAP_MAPPINGS_COLOR_KEY:'#898700'},
            'TOMM20':{self.UMAP_MAPPINGS_ALIAS_KEY:'MOM', self.UMAP_MAPPINGS_COLOR_KEY:'#66CDAA'},
            'PURA':{self.UMAP_MAPPINGS_ALIAS_KEY:'PURA granules', self.UMAP_MAPPINGS_COLOR_KEY:'#AF8215'},
            'CLTC':{self.UMAP_MAPPINGS_ALIAS_KEY:'Coated Vesicles', self.UMAP_MAPPINGS_COLOR_KEY:'#32AC0E'},
            'KIF5A':{self.UMAP_MAPPINGS_ALIAS_KEY:'Transport machinery', self.UMAP_MAPPINGS_COLOR_KEY:'#ACE142'},
            'SCNA':{self.UMAP_MAPPINGS_ALIAS_KEY:'Presynapse', self.UMAP_MAPPINGS_COLOR_KEY:'#DEDB23'},
            'SNCA':{self.UMAP_MAPPINGS_ALIAS_KEY:'Presynapse', self.UMAP_MAPPINGS_COLOR_KEY:'#DEDB23'},
            'CD41':{self.UMAP_MAPPINGS_ALIAS_KEY:'Integrin puncta', self.UMAP_MAPPINGS_COLOR_KEY:'#F04521'},
            'SQSTM1':{self.UMAP_MAPPINGS_ALIAS_KEY:'Autophagosomes', self.UMAP_MAPPINGS_COLOR_KEY:'#FFBF0D'},
            'G3BP1':{self.UMAP_MAPPINGS_ALIAS_KEY:'Stress Granules', self.UMAP_MAPPINGS_COLOR_KEY:'#A80358'},
            'GM130':{self.UMAP_MAPPINGS_ALIAS_KEY:'Golgi', self.UMAP_MAPPINGS_COLOR_KEY:'#D257EA'},
            'LAMP1':{self.UMAP_MAPPINGS_ALIAS_KEY:'Lysosome', self.UMAP_MAPPINGS_COLOR_KEY:'#E6A9EA'},
            'DCP1A':{self.UMAP_MAPPINGS_ALIAS_KEY:'P-Bodies', self.UMAP_MAPPINGS_COLOR_KEY:'#F0A3A3'},
            'NEMO':{self.UMAP_MAPPINGS_ALIAS_KEY:'NEMO granules', self.UMAP_MAPPINGS_COLOR_KEY:'#EF218B'},
            'PSD95':{self.UMAP_MAPPINGS_ALIAS_KEY:'Postsynapse', self.UMAP_MAPPINGS_COLOR_KEY:'#F1CBDD'},
            'FMRP':{self.UMAP_MAPPINGS_ALIAS_KEY:'FMRP', self.UMAP_MAPPINGS_COLOR_KEY:'gray'},
            'TDP43B':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP43 granules', self.UMAP_MAPPINGS_COLOR_KEY:'#06A0E9'},
            # 'TDP43N':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP43 granules 2', self.UMAP_MAPPINGS_COLOR_KEY:'#06A0E9'}
        }

        self.COLOR_MAPPINGS_CONDITION_AND_ALS: Dict[str, Dict[str,str]] = {
            'WT_stress':{self.UMAP_MAPPINGS_ALIAS_KEY: 'Wild-Type + Stress',self.UMAP_MAPPINGS_COLOR_KEY: '#F7810F'},
            'WT_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'Wild-Type', self.UMAP_MAPPINGS_COLOR_KEY:'#37AFD7'},
            'FUSHeterozygous_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'FUS Heterozygous', self.UMAP_MAPPINGS_COLOR_KEY:'#AB7A5B'},
            'FUSHomozygous_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'FUS Homozygous', self.UMAP_MAPPINGS_COLOR_KEY:'#78491C'},
            'FUSRevertant_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'FUS Revertant', self.UMAP_MAPPINGS_COLOR_KEY:'#C8C512'},
            'OPTN_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'OPTN', self.UMAP_MAPPINGS_COLOR_KEY:'#FF98BB'},
            'TBK1_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'TBK1', self.UMAP_MAPPINGS_COLOR_KEY:'#319278'},
            'SCNA_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'SCNA', self.UMAP_MAPPINGS_COLOR_KEY:'black'},
            'SNCA_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'SNCA', self.UMAP_MAPPINGS_COLOR_KEY:'black'},
            'TDP43_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP43', self.UMAP_MAPPINGS_COLOR_KEY:'#A8559E'},
        }

        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS
