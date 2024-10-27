import sys
import os
from typing import Dict

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.figures import plot_config 

class PlotConfig(plot_config.PlotConfig):
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
            'SCNA_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'SNCA', self.UMAP_MAPPINGS_COLOR_KEY:'black'},
            'SNCA_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'SNCA', self.UMAP_MAPPINGS_COLOR_KEY:'black'},
            'TDP43_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP-43', self.UMAP_MAPPINGS_COLOR_KEY:'#A8559E'},
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
            'TDP43_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP-43dNLS, -DOX', self.UMAP_MAPPINGS_COLOR_KEY:'#6BAD31'},
            'TDP43_dox':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP-43dNLS, +DOX', self.UMAP_MAPPINGS_COLOR_KEY:'#90278E'}
        }

        self.COLOR_MAPPINGS_MARKERS: Dict[str, Dict[str,str]] = {
            'DAPI':{self.UMAP_MAPPINGS_ALIAS_KEY:'Nucleus', self.UMAP_MAPPINGS_COLOR_KEY:'#AFBDFF'},
            'Calreticulin':{self.UMAP_MAPPINGS_ALIAS_KEY:'ER', self.UMAP_MAPPINGS_COLOR_KEY:'#4343FE'},
            'NCL':{self.UMAP_MAPPINGS_ALIAS_KEY:'Nucleolus', self.UMAP_MAPPINGS_COLOR_KEY:'#921010'},
            'TDP43':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP-43 granules', self.UMAP_MAPPINGS_COLOR_KEY:'#12F986'},
            'TDP43B':{self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP-43 granules', self.UMAP_MAPPINGS_COLOR_KEY :'#12F986'},
            'NONO':{self.UMAP_MAPPINGS_ALIAS_KEY:'Paraspeckles', self.UMAP_MAPPINGS_COLOR_KEY:'#66CDAA'},
            'ANXA11':{self.UMAP_MAPPINGS_ALIAS_KEY:'ANXA11 granules', self.UMAP_MAPPINGS_COLOR_KEY:'#18E4CF'},
            'GM130':{self.UMAP_MAPPINGS_ALIAS_KEY:'Golgi', self.UMAP_MAPPINGS_COLOR_KEY:'#168FB2'},
            'LAMP1':{self.UMAP_MAPPINGS_ALIAS_KEY:'Lysosome', self.UMAP_MAPPINGS_COLOR_KEY:'#A80358'},
            'FUS':{self.UMAP_MAPPINGS_ALIAS_KEY:'hnRNP complex', self.UMAP_MAPPINGS_COLOR_KEY:'#9968CB'},
            'PEX14':{self.UMAP_MAPPINGS_ALIAS_KEY:'Peroxisome', self.UMAP_MAPPINGS_COLOR_KEY:'#D257EA'},
            'DCP1A':{self.UMAP_MAPPINGS_ALIAS_KEY:'P-Bodies', self.UMAP_MAPPINGS_COLOR_KEY:'#E6A9EA'},
            'CD41':{self.UMAP_MAPPINGS_ALIAS_KEY:'Integrin puncta', self.UMAP_MAPPINGS_COLOR_KEY:'#F04521'},
            'SQSTM1':{self.UMAP_MAPPINGS_ALIAS_KEY:'Autophagosomes', self.UMAP_MAPPINGS_COLOR_KEY:'#F08F21'},
            'PML':{self.UMAP_MAPPINGS_ALIAS_KEY:'PML bodies', self.UMAP_MAPPINGS_COLOR_KEY:'#F1CBDD'},
            'SCNA':{self.UMAP_MAPPINGS_ALIAS_KEY:'Presynapse', self.UMAP_MAPPINGS_COLOR_KEY:'#FFBF0D'},
            'SNCA':{self.UMAP_MAPPINGS_ALIAS_KEY: 'Presynapse', self.UMAP_MAPPINGS_COLOR_KEY : '#FFBF0D'},
            'NEMO':{self.UMAP_MAPPINGS_ALIAS_KEY:'NEMO granules', self.UMAP_MAPPINGS_COLOR_KEY:'#37378D'},
            'PSD95':{self.UMAP_MAPPINGS_ALIAS_KEY:'Postsynapse', self.UMAP_MAPPINGS_COLOR_KEY:'gray'},
            'KIF5A':{self.UMAP_MAPPINGS_ALIAS_KEY:'Transport machinery', self.UMAP_MAPPINGS_COLOR_KEY:'#DEDB23'},
            'CLTC':{self.UMAP_MAPPINGS_ALIAS_KEY:'Coated vesicles', self.UMAP_MAPPINGS_COLOR_KEY:'#AF8215'},
            'TOMM20':{self.UMAP_MAPPINGS_ALIAS_KEY:'MOM', self.UMAP_MAPPINGS_COLOR_KEY:'#32AC0E'},
            'mitotracker':{self.UMAP_MAPPINGS_ALIAS_KEY:'Mitochondria', self.UMAP_MAPPINGS_COLOR_KEY:'#898700'},
            'PURA':{self.UMAP_MAPPINGS_ALIAS_KEY:'PURA granules', self.UMAP_MAPPINGS_COLOR_KEY:'#ACE142'},
            'G3BP1':{self.UMAP_MAPPINGS_ALIAS_KEY:'Stress granules', self.UMAP_MAPPINGS_COLOR_KEY:'#F0A3A3'},
            'Phalloidin':{self.UMAP_MAPPINGS_ALIAS_KEY:'Cytoskeleton', self.UMAP_MAPPINGS_COLOR_KEY:'#06A0E9'},
            'FMRP':{self.UMAP_MAPPINGS_ALIAS_KEY:'FMRP granules', self.UMAP_MAPPINGS_COLOR_KEY:'#EF218B'},

            'MERGED':{self.UMAP_MAPPINGS_ALIAS_KEY:'MERGED', self.UMAP_MAPPINGS_COLOR_KEY:'gray'},
            'Map2':{self.UMAP_MAPPINGS_ALIAS_KEY:'Neuronal marker', self.UMAP_MAPPINGS_COLOR_KEY:'gray'},
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
            'SCNA_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'SNCA', self.UMAP_MAPPINGS_COLOR_KEY:'black'},
            'SNCA_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'SNCA', self.UMAP_MAPPINGS_COLOR_KEY:'black'},
            'TDP43_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'TDP-43', self.UMAP_MAPPINGS_COLOR_KEY:'#A8559E'},
        }

        self.COLOR_MAPPINGS_ALYSSA: Dict[str, Dict[str,str]] = {
            'Controls_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY: 'Controls',self.UMAP_MAPPINGS_COLOR_KEY: '#58cfdf'},
            'sALSPositiveCytoTDP43_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'sALS Positive TDP-43', self.UMAP_MAPPINGS_COLOR_KEY:'#f6ce55'},
            'sALSNegativeCytoTDP43_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'sALS Negative TDP-43', self.UMAP_MAPPINGS_COLOR_KEY:'#3ce23c'},
            'c9orf72ALSPatients_Untreated':{self.UMAP_MAPPINGS_ALIAS_KEY:'c9orf72', self.UMAP_MAPPINGS_COLOR_KEY:'#eb3440'},

            'Controls_rep1':{self.UMAP_MAPPINGS_ALIAS_KEY: '1',self.UMAP_MAPPINGS_COLOR_KEY: '#58cfdf'},
            'Controls_rep2':{self.UMAP_MAPPINGS_ALIAS_KEY: '2',self.UMAP_MAPPINGS_COLOR_KEY: '#4db6c4'},
            'Controls_rep3':{self.UMAP_MAPPINGS_ALIAS_KEY: '3',self.UMAP_MAPPINGS_COLOR_KEY: '#439da9'},
            'Controls_rep4':{self.UMAP_MAPPINGS_ALIAS_KEY: '4',self.UMAP_MAPPINGS_COLOR_KEY: '#38848f'},
            'Controls_rep5':{self.UMAP_MAPPINGS_ALIAS_KEY: '5',self.UMAP_MAPPINGS_COLOR_KEY: '#2e6c74'},
            'Controls_rep6':{self.UMAP_MAPPINGS_ALIAS_KEY: '6',self.UMAP_MAPPINGS_COLOR_KEY: '#235359'},
            
            'sALSPositiveCytoTDP43_rep1':{self.UMAP_MAPPINGS_ALIAS_KEY:'1', self.UMAP_MAPPINGS_COLOR_KEY:'#f6ce55'},
            'sALSPositiveCytoTDP43_rep10':{self.UMAP_MAPPINGS_ALIAS_KEY:'10', self.UMAP_MAPPINGS_COLOR_KEY:'#e8c350'},
            'sALSPositiveCytoTDP43_rep2':{self.UMAP_MAPPINGS_ALIAS_KEY:'2', self.UMAP_MAPPINGS_COLOR_KEY:'#dbb74c'},
            'sALSPositiveCytoTDP43_rep3':{self.UMAP_MAPPINGS_ALIAS_KEY:'3', self.UMAP_MAPPINGS_COLOR_KEY:'#cdac47'},
            'sALSPositiveCytoTDP43_rep4':{self.UMAP_MAPPINGS_ALIAS_KEY:'4', self.UMAP_MAPPINGS_COLOR_KEY:'#bfa042'},
            'sALSPositiveCytoTDP43_rep5':{self.UMAP_MAPPINGS_ALIAS_KEY:'5', self.UMAP_MAPPINGS_COLOR_KEY:'#b2953d'},
            'sALSPositiveCytoTDP43_rep6':{self.UMAP_MAPPINGS_ALIAS_KEY:'6', self.UMAP_MAPPINGS_COLOR_KEY:'#a48939'},
            'sALSPositiveCytoTDP43_rep7':{self.UMAP_MAPPINGS_ALIAS_KEY:'7', self.UMAP_MAPPINGS_COLOR_KEY:'#967e34'},
            'sALSPositiveCytoTDP43_rep8':{self.UMAP_MAPPINGS_ALIAS_KEY:'8', self.UMAP_MAPPINGS_COLOR_KEY:'#89722f'},
            'sALSPositiveCytoTDP43_rep9':{self.UMAP_MAPPINGS_ALIAS_KEY:'9', self.UMAP_MAPPINGS_COLOR_KEY:'#7b672a'},

            'sALSNegativeCytoTDP43_rep1':{self.UMAP_MAPPINGS_ALIAS_KEY:'1', self.UMAP_MAPPINGS_COLOR_KEY:'#3ce23c'},
            'sALSNegativeCytoTDP43_rep2':{self.UMAP_MAPPINGS_ALIAS_KEY:'2', self.UMAP_MAPPINGS_COLOR_KEY:'#1e711e'},

            'c9orf72ALSPatients_rep1':{self.UMAP_MAPPINGS_ALIAS_KEY:'1', self.UMAP_MAPPINGS_COLOR_KEY:'#eb3440'},
            'c9orf72ALSPatients_rep2':{self.UMAP_MAPPINGS_ALIAS_KEY:'2', self.UMAP_MAPPINGS_COLOR_KEY:'#bc2a33'},
            'c9orf72ALSPatients_rep3':{self.UMAP_MAPPINGS_ALIAS_KEY:'3', self.UMAP_MAPPINGS_COLOR_KEY:'#8d1f26'},

        }
        self.COLOR_MAPPINGS_ALYSSA['Controls'] = self.COLOR_MAPPINGS_ALYSSA['Controls_Untreated']
        self.COLOR_MAPPINGS_ALYSSA['sALSPositiveCytoTDP43'] = self.COLOR_MAPPINGS_ALYSSA['sALSPositiveCytoTDP43_Untreated']
        self.COLOR_MAPPINGS_ALYSSA['sALSNegativeCytoTDP43'] = self.COLOR_MAPPINGS_ALYSSA['sALSNegativeCytoTDP43_Untreated']
        self.COLOR_MAPPINGS_ALYSSA['c9orf72ALSPatients'] = self.COLOR_MAPPINGS_ALYSSA['c9orf72ALSPatients_Untreated']

        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS
