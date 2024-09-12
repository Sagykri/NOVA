import sys
import os
from typing import Dict

sys.path.insert(1, os.getenv("MOMAPS_HOME")) 
from src.common.configs.base_config import BaseConfig

class ColormapWithAlias():
    """Holds color and alias
    """
    def __init__(self, alias:str, color:str):
        self.alias:str = alias
        self.color:str = color
        
class PlotConfig(BaseConfig):
    """Config for plotting
    """
    def __init__(self):
        
        # Set the size of the dots
        self.SIZE = 30
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7

        self.COLOR_MAPPINGS_CONDITION: Dict[str, ColormapWithAlias] = {
            'Untreated': ColormapWithAlias(alias='- Stress', color='#52C5D5'),
            'stress': ColormapWithAlias(alias='+ Stress', color='#F7810F'),
        }
        
        self.COLOR_MAPPINGS_ALS:Dict[str, ColormapWithAlias] = {
            'WT_Untreated': ColormapWithAlias(alias='Wild-Type', color='#37AFD7'),
            'FUSHeterozygous_Untreated': ColormapWithAlias(alias='FUS Heterozygous', color='#AB7A5B'),
            'FUSHomozygous_Untreated': ColormapWithAlias(alias='FUS Homozygous', color='#78491C'),
            'FUSRevertant_Untreated': ColormapWithAlias(alias='FUS Revertant', color='#C8C512'),
            'OPTN_Untreated': ColormapWithAlias(alias='OPTN', color='#FF98BB'),
            'TBK1_Untreated': ColormapWithAlias(alias='TBK1', color='#319278'),
            'SCNA_Untreated': ColormapWithAlias(alias='SCNA', color='black'),
            'SNCA_Untreated': ColormapWithAlias(alias='SNCA', color='black'),
            'TDP43_Untreated': ColormapWithAlias(alias='TDP43', color='#A8559E'),
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
        
        self.COLOR_MAPPINGS_DOX: Dict[str, ColormapWithAlias] = {
            'WT_Untreated': ColormapWithAlias(alias='Wild-Type', color='#2FA0C1'),
            'TDP43_Untreated': ColormapWithAlias(alias='TDP43dNLS, -Dox', color='#6BAD31'),
            'TDP43_dox': ColormapWithAlias(alias='TDP43dNLS, +Dox', color='#90278E')
        }

        self.COLOR_MAPPINGS_MARKERS: Dict[str, ColormapWithAlias] = {
            'NCL': ColormapWithAlias(alias='Nucleolus', color='#18E4CF'),
            'FUS': ColormapWithAlias(alias='hnRNP complex', color='#9968CB'),
            'DAPI': ColormapWithAlias(alias='Nucleus', color='#AFBDFF'),
            'PML': ColormapWithAlias(alias='PML bodies', color='#F08F21'),
            'ANXA11': ColormapWithAlias(alias='ANXA11 granules', color='#37378D'),
            'NONO': ColormapWithAlias(alias='Paraspeckles', color='#4343FE'),
            'TDP43': ColormapWithAlias(alias='TDP43 granules', color='#06A0E9'),
            'PEX14': ColormapWithAlias(alias='Peroxisome', color='#168FB2'),
            'Calreticulin': ColormapWithAlias(alias='ER', color='#12F986'),
            'Phalloidin': ColormapWithAlias(alias='Cytoskeleton', color='#921010'),
            'mitotracker': ColormapWithAlias(alias='Mitochondria', color='#898700'),
            'TOMM20': ColormapWithAlias(alias='MOM', color='#66CDAA'),
            'PURA': ColormapWithAlias(alias='PURA granules', color='#AF8215'),
            'CLTC': ColormapWithAlias(alias='Coated Vesicles', color='#32AC0E'),
            'KIF5A': ColormapWithAlias(alias='Transport machinery', color='#ACE142'),
            'SCNA': ColormapWithAlias(alias='Presynapse', color='#DEDB23'),
            'SNCA': ColormapWithAlias(alias='Presynapse', color='#DEDB23'),
            'CD41': ColormapWithAlias(alias='Integrin puncta', color='#F04521'),
            'SQSTM1': ColormapWithAlias(alias='Autophagosomes', color='#FFBF0D'),
            'G3BP1': ColormapWithAlias(alias='Stress Granules', color='#A80358'),
            'GM130': ColormapWithAlias(alias='Golgi', color='#D257EA'),
            'LAMP1': ColormapWithAlias(alias='Lysosome', color='#E6A9EA'),
            'DCP1A': ColormapWithAlias(alias='P-Bodies', color='#F0A3A3'),
            'NEMO': ColormapWithAlias(alias='NEMO granules', color='#EF218B'),
            'PSD95': ColormapWithAlias(alias='Postsynapse', color='#F1CBDD'),
            'FMRP': ColormapWithAlias(alias='FMRP', color='gray'),
            'TDP43B': ColormapWithAlias(alias='TDP43 granules 1', color='#06A0E9'),
            'TDP43N': ColormapWithAlias(alias='TDP43 granules 2', color='#06A0E9')
        }

        self.COLOR_MAPPINGS_CONDITION_AND_ALS: Dict[str, ColormapWithAlias] = {
            'WT_stress': ColormapWithAlias(alias= 'Wild-Type + Stress',color= '#F7810F'),
            'WT_Untreated': ColormapWithAlias(alias='Wild-Type', color='#37AFD7'),
            'FUSHeterozygous_Untreated': ColormapWithAlias(alias='FUS Heterozygous', color='#AB7A5B'),
            'FUSHomozygous_Untreated': ColormapWithAlias(alias='FUS Homozygous', color='#78491C'),
            'FUSRevertant_Untreated': ColormapWithAlias(alias='FUS Revertant', color='#C8C512'),
            'OPTN_Untreated': ColormapWithAlias(alias='OPTN', color='#FF98BB'),
            'TBK1_Untreated': ColormapWithAlias(alias='TBK1', color='#319278'),
            'SCNA_Untreated': ColormapWithAlias(alias='SCNA', color='black'),
            'SNCA_Untreated': ColormapWithAlias(alias='SNCA', color='black'),
            'TDP43_Untreated': ColormapWithAlias(alias='TDP43', color='#A8559E'),
        }

        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS
