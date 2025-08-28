import sys
import os
from typing import Dict

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.figures.plot_config import PlotConfig 

class PlotConfig(PlotConfig):
    """Config for plotting
    """
    def __init__(self):
        
        super().__init__()

        # Set the size of the dots
        self.SIZE = 5
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.7
        
        self.COLOR_MAPPINGS_CONDITION: Dict[str, Dict[str,str]] = {
            'Untreated':{self.MAPPINGS_ALIAS_KEY:'- Stress', self.MAPPINGS_COLOR_KEY:'#52C5D5'},
            'stress':{self.MAPPINGS_ALIAS_KEY:'+ Stress', self.MAPPINGS_COLOR_KEY:'#F7810F'},
        }
        
        self.COLOR_MAPPINGS_ALS: Dict[str, Dict[str,str]] = {
            'WT_Untreated':{self.MAPPINGS_ALIAS_KEY:'Wild-Type', self.MAPPINGS_COLOR_KEY:'#37AFD7'},
            'FUSHeterozygous_Untreated':{self.MAPPINGS_ALIAS_KEY:'FUS Heterozygous', self.MAPPINGS_COLOR_KEY:'#AB7A5B'},
            'FUSHomozygous_Untreated':{self.MAPPINGS_ALIAS_KEY:'FUS Homozygous', self.MAPPINGS_COLOR_KEY:'#78491C'},
            'FUSRevertant_Untreated':{self.MAPPINGS_ALIAS_KEY:'FUS Revertant', self.MAPPINGS_COLOR_KEY:'#C8C512'},
            'OPTN_Untreated':{self.MAPPINGS_ALIAS_KEY:'OPTN', self.MAPPINGS_COLOR_KEY:'#FF98BB'},
            'TBK1_Untreated':{self.MAPPINGS_ALIAS_KEY:'TBK1', self.MAPPINGS_COLOR_KEY:'#319278'},
            'SCNA_Untreated':{self.MAPPINGS_ALIAS_KEY:'SNCA', self.MAPPINGS_COLOR_KEY:'black'},
            'SNCA_Untreated':{self.MAPPINGS_ALIAS_KEY:'SNCA', self.MAPPINGS_COLOR_KEY:'black'},
            'TDP43_Untreated':{self.MAPPINGS_ALIAS_KEY:'TDP-43', self.MAPPINGS_COLOR_KEY:'#A8559E'},
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
            'WT_Untreated':{self.MAPPINGS_ALIAS_KEY:'Wild-Type', self.MAPPINGS_COLOR_KEY:'#2FA0C1'},
            'TDP43_Untreated':{self.MAPPINGS_ALIAS_KEY:'TDP-43dNLS, -DOX', self.MAPPINGS_COLOR_KEY:'#6BAD31'},
            'TDP43_dox':{self.MAPPINGS_ALIAS_KEY:'TDP-43dNLS, +DOX', self.MAPPINGS_COLOR_KEY:'#90278E'},
            'dNLS_Untreated':{self.MAPPINGS_ALIAS_KEY:'TDP-43dNLS, -DOX', self.MAPPINGS_COLOR_KEY:'#6BAD31'},
            'dNLS_DOX':{self.MAPPINGS_ALIAS_KEY:'TDP-43dNLS, +DOX', self.MAPPINGS_COLOR_KEY:'#90278E'}
        }

        self.COLOR_MAPPINGS_MARKERS: Dict[str, Dict[str,str]] = {
            'DAPI':{self.MAPPINGS_ALIAS_KEY:'Nucleus', self.MAPPINGS_COLOR_KEY:'#AFBDFF'},
            'Calreticulin':{self.MAPPINGS_ALIAS_KEY:'ER', self.MAPPINGS_COLOR_KEY:'#4343FE'},
            'NCL':{self.MAPPINGS_ALIAS_KEY:'Nucleolus', self.MAPPINGS_COLOR_KEY:'#921010'},
            'TDP43':{self.MAPPINGS_ALIAS_KEY:'TDP-43 granules', self.MAPPINGS_COLOR_KEY:'#12F986'},
            'TDP43B':{self.MAPPINGS_ALIAS_KEY: 'TDP-43 granules', self.MAPPINGS_COLOR_KEY :'#12F986'},
            'NONO':{self.MAPPINGS_ALIAS_KEY:'Paraspeckles', self.MAPPINGS_COLOR_KEY:'#66CDAA'},
            'ANXA11':{self.MAPPINGS_ALIAS_KEY:'ANXA11 granules', self.MAPPINGS_COLOR_KEY:'#18E4CF'},
            'GM130':{self.MAPPINGS_ALIAS_KEY:'Golgi', self.MAPPINGS_COLOR_KEY:'#168FB2'},
            'LAMP1':{self.MAPPINGS_ALIAS_KEY:'Lysosome', self.MAPPINGS_COLOR_KEY:'#A80358'},
            'FUS':{self.MAPPINGS_ALIAS_KEY:'hnRNP complex', self.MAPPINGS_COLOR_KEY:'#9968CB'},
            'PEX14':{self.MAPPINGS_ALIAS_KEY:'Peroxisome', self.MAPPINGS_COLOR_KEY:'#D257EA'},
            'DCP1A':{self.MAPPINGS_ALIAS_KEY:'P-Bodies', self.MAPPINGS_COLOR_KEY:'#E6A9EA'},
            'CD41':{self.MAPPINGS_ALIAS_KEY:'Integrin puncta', self.MAPPINGS_COLOR_KEY:'#F04521'},
            'SQSTM1':{self.MAPPINGS_ALIAS_KEY:'Autophagosomes', self.MAPPINGS_COLOR_KEY:'#F08F21'},
            'PML':{self.MAPPINGS_ALIAS_KEY:'PML bodies', self.MAPPINGS_COLOR_KEY:'#F1CBDD'},
            'SCNA':{self.MAPPINGS_ALIAS_KEY:'Presynapse', self.MAPPINGS_COLOR_KEY:'#FFBF0D'},
            'SNCA':{self.MAPPINGS_ALIAS_KEY: 'Presynapse', self.MAPPINGS_COLOR_KEY : '#FFBF0D'},
            'NEMO':{self.MAPPINGS_ALIAS_KEY:'NEMO granules', self.MAPPINGS_COLOR_KEY:'#37378D'},
            'PSD95':{self.MAPPINGS_ALIAS_KEY:'Postsynapse', self.MAPPINGS_COLOR_KEY:'gray'},
            'KIF5A':{self.MAPPINGS_ALIAS_KEY:'Transport machinery', self.MAPPINGS_COLOR_KEY:'#DEDB23'},
            'CLTC':{self.MAPPINGS_ALIAS_KEY:'Coated vesicles', self.MAPPINGS_COLOR_KEY:'#AF8215'},
            'TOMM20':{self.MAPPINGS_ALIAS_KEY:'MOM', self.MAPPINGS_COLOR_KEY:'#32AC0E'},
            'mitotracker':{self.MAPPINGS_ALIAS_KEY:'Mitochondria', self.MAPPINGS_COLOR_KEY:'#898700'},
            'PURA':{self.MAPPINGS_ALIAS_KEY:'PURA granules', self.MAPPINGS_COLOR_KEY:'#ACE142'},
            'G3BP1':{self.MAPPINGS_ALIAS_KEY:'Stress granules', self.MAPPINGS_COLOR_KEY:'#F0A3A3'},
            'Phalloidin':{self.MAPPINGS_ALIAS_KEY:'Actin Cytoskeleton', self.MAPPINGS_COLOR_KEY:'#06A0E9'},
            'FMRP':{self.MAPPINGS_ALIAS_KEY:'FMRP granules', self.MAPPINGS_COLOR_KEY:'#EF218B'},

            'MERGED':{self.MAPPINGS_ALIAS_KEY:'MERGED', self.MAPPINGS_COLOR_KEY:'gray'},
            'Map2':{self.MAPPINGS_ALIAS_KEY:'Neuronal marker', self.MAPPINGS_COLOR_KEY:'gray'},

            'LSM14A':{self.MAPPINGS_ALIAS_KEY:'P-bodies #2', self.MAPPINGS_COLOR_KEY:"#B91EC4"},
            'SON':{self.MAPPINGS_ALIAS_KEY:'Nuclear Speckles', self.MAPPINGS_COLOR_KEY:"#EC5F5F"},
            'HNRNPA1':{self.MAPPINGS_ALIAS_KEY:'hnRNP complex #2', self.MAPPINGS_COLOR_KEY:"#033A7A"},
            'Tubulin':{self.MAPPINGS_ALIAS_KEY:'Microtubule', self.MAPPINGS_COLOR_KEY:"#00D612"},
            'TIA1':{self.MAPPINGS_ALIAS_KEY:'TIA1 granules', self.MAPPINGS_COLOR_KEY:"#BB00E0"},

            'Nup153': {self.MAPPINGS_ALIAS_KEY:'Nup153', self.MAPPINGS_COLOR_KEY:"#0F8500"},
            'POM121': {self.MAPPINGS_ALIAS_KEY:'POM121', self.MAPPINGS_COLOR_KEY:"#FF6F00"},
            'Calnexin': {self.MAPPINGS_ALIAS_KEY:'Calnexin', self.MAPPINGS_COLOR_KEY:"#FFB300"},
            'EEA1': {self.MAPPINGS_ALIAS_KEY:'EEA1', self.MAPPINGS_COLOR_KEY:"#FF8F00"},
            'hnRNPA2B1': {self.MAPPINGS_ALIAS_KEY:'hnRNPA2B1', self.MAPPINGS_COLOR_KEY:"#FF3D00"},
            'Nup62': {self.MAPPINGS_ALIAS_KEY:'Nup62', self.MAPPINGS_COLOR_KEY:'#FFB6C1'},
            'Nup98': {self.MAPPINGS_ALIAS_KEY:'Nup98', self.MAPPINGS_COLOR_KEY:'#FF69B4'},

        }

        self.COLOR_MAPPINGS_MARKERS['hnRNPA1'] = self.COLOR_MAPPINGS_MARKERS['HNRNPA1']
        self.COLOR_MAPPINGS_MARKERS['LaminB1'] = {self.MAPPINGS_ALIAS_KEY:'LaminB1', self.MAPPINGS_COLOR_KEY:"#770516"}
        self.COLOR_MAPPINGS_MARKERS['Lamp1'] = self.COLOR_MAPPINGS_MARKERS['LAMP1']

        # NIH
        self.COLOR_MAPPINGS_MARKERS['ANAX11'] = self.COLOR_MAPPINGS_MARKERS['ANXA11']
        self.COLOR_MAPPINGS_MARKERS['MitoTracker'] = self.COLOR_MAPPINGS_MARKERS['mitotracker']
        self.COLOR_MAPPINGS_MARKERS['P54'] = self.COLOR_MAPPINGS_MARKERS['NONO']
        self.COLOR_MAPPINGS_MARKERS['TUJ1'] = {self.MAPPINGS_ALIAS_KEY:'Microtubule', self.MAPPINGS_COLOR_KEY:"#65FF72"}
        ##


        self.COLOR_MAPPINGS_CONDITION_AND_ALS: Dict[str, Dict[str,str]] = {
            'WT_stress':{self.MAPPINGS_ALIAS_KEY: 'Wild-Type + Stress',self.MAPPINGS_COLOR_KEY: '#F7810F'},
            'WT_Untreated':{self.MAPPINGS_ALIAS_KEY:'Wild-Type', self.MAPPINGS_COLOR_KEY:'#37AFD7'},
            'FUSHeterozygous_Untreated':{self.MAPPINGS_ALIAS_KEY:'FUS Heterozygous', self.MAPPINGS_COLOR_KEY:'#AB7A5B'},
            'FUSHomozygous_Untreated':{self.MAPPINGS_ALIAS_KEY:'FUS Homozygous', self.MAPPINGS_COLOR_KEY:'#78491C'},
            'FUSRevertant_Untreated':{self.MAPPINGS_ALIAS_KEY:'FUS Revertant', self.MAPPINGS_COLOR_KEY:'#C8C512'},
            'OPTN_Untreated':{self.MAPPINGS_ALIAS_KEY:'OPTN', self.MAPPINGS_COLOR_KEY:'#FF98BB'},
            'TBK1_Untreated':{self.MAPPINGS_ALIAS_KEY:'TBK1', self.MAPPINGS_COLOR_KEY:'#319278'},
            'SCNA_Untreated':{self.MAPPINGS_ALIAS_KEY:'SNCA', self.MAPPINGS_COLOR_KEY:'black'},
            'SNCA_Untreated':{self.MAPPINGS_ALIAS_KEY:'SNCA', self.MAPPINGS_COLOR_KEY:'black'},
            'TDP43_Untreated':{self.MAPPINGS_ALIAS_KEY:'TDP-43', self.MAPPINGS_COLOR_KEY:'#A8559E'},
        }

        self.COLOR_MAPPINGS_ALYSSA: Dict[str, Dict[str,str]] = {
            'Controls_Untreated':{self.MAPPINGS_ALIAS_KEY: 'Controls',self.MAPPINGS_COLOR_KEY: '#58cfdf'},
            'sALSPositiveCytoTDP43_Untreated':{self.MAPPINGS_ALIAS_KEY:'sALS Positive TDP-43', self.MAPPINGS_COLOR_KEY:'#f6ce55'},
            'sALSNegativeCytoTDP43_Untreated':{self.MAPPINGS_ALIAS_KEY:'sALS Negative TDP-43', self.MAPPINGS_COLOR_KEY:'#3ce23c'},
            'c9orf72ALSPatients_Untreated':{self.MAPPINGS_ALIAS_KEY:'c9orf72', self.MAPPINGS_COLOR_KEY:'#eb3440'},

            'Controls_rep1':{self.MAPPINGS_ALIAS_KEY: '1',self.MAPPINGS_COLOR_KEY: '#58cfdf'},
            'Controls_rep2':{self.MAPPINGS_ALIAS_KEY: '2',self.MAPPINGS_COLOR_KEY: '#4db6c4'},
            'Controls_rep3':{self.MAPPINGS_ALIAS_KEY: '3',self.MAPPINGS_COLOR_KEY: '#439da9'},
            'Controls_rep4':{self.MAPPINGS_ALIAS_KEY: '4',self.MAPPINGS_COLOR_KEY: '#38848f'},
            'Controls_rep5':{self.MAPPINGS_ALIAS_KEY: '5',self.MAPPINGS_COLOR_KEY: '#2e6c74'},
            'Controls_rep6':{self.MAPPINGS_ALIAS_KEY: '6',self.MAPPINGS_COLOR_KEY: '#235359'},
            
            'sALSPositiveCytoTDP43_rep1':{self.MAPPINGS_ALIAS_KEY:'1', self.MAPPINGS_COLOR_KEY:'#f6ce55'},
            'sALSPositiveCytoTDP43_rep10':{self.MAPPINGS_ALIAS_KEY:'10', self.MAPPINGS_COLOR_KEY:'#e8c350'},
            'sALSPositiveCytoTDP43_rep2':{self.MAPPINGS_ALIAS_KEY:'2', self.MAPPINGS_COLOR_KEY:'#dbb74c'},
            'sALSPositiveCytoTDP43_rep3':{self.MAPPINGS_ALIAS_KEY:'3', self.MAPPINGS_COLOR_KEY:'#cdac47'},
            'sALSPositiveCytoTDP43_rep4':{self.MAPPINGS_ALIAS_KEY:'4', self.MAPPINGS_COLOR_KEY:'#bfa042'},
            'sALSPositiveCytoTDP43_rep5':{self.MAPPINGS_ALIAS_KEY:'5', self.MAPPINGS_COLOR_KEY:'#b2953d'},
            'sALSPositiveCytoTDP43_rep6':{self.MAPPINGS_ALIAS_KEY:'6', self.MAPPINGS_COLOR_KEY:'#a48939'},
            'sALSPositiveCytoTDP43_rep7':{self.MAPPINGS_ALIAS_KEY:'7', self.MAPPINGS_COLOR_KEY:'#967e34'},
            'sALSPositiveCytoTDP43_rep8':{self.MAPPINGS_ALIAS_KEY:'8', self.MAPPINGS_COLOR_KEY:'#89722f'},
            'sALSPositiveCytoTDP43_rep9':{self.MAPPINGS_ALIAS_KEY:'9', self.MAPPINGS_COLOR_KEY:'#7b672a'},

            'sALSNegativeCytoTDP43_rep1':{self.MAPPINGS_ALIAS_KEY:'1', self.MAPPINGS_COLOR_KEY:'#3ce23c'},
            'sALSNegativeCytoTDP43_rep2':{self.MAPPINGS_ALIAS_KEY:'2', self.MAPPINGS_COLOR_KEY:'#1e711e'},

            'c9orf72ALSPatients_rep1':{self.MAPPINGS_ALIAS_KEY:'1', self.MAPPINGS_COLOR_KEY:'#eb3440'},
            'c9orf72ALSPatients_rep2':{self.MAPPINGS_ALIAS_KEY:'2', self.MAPPINGS_COLOR_KEY:'#bc2a33'},
            'c9orf72ALSPatients_rep3':{self.MAPPINGS_ALIAS_KEY:'3', self.MAPPINGS_COLOR_KEY:'#8d1f26'},

            'Ctrl':{self.MAPPINGS_ALIAS_KEY: 'Controls',self.MAPPINGS_COLOR_KEY: '#58cfdf'},
            'SALSPositive':{self.MAPPINGS_ALIAS_KEY:'sALS Positive TDP-43', self.MAPPINGS_COLOR_KEY:'#f6ce55'},
            'SALSNegative':{self.MAPPINGS_ALIAS_KEY:'sALS Negative TDP-43', self.MAPPINGS_COLOR_KEY:'#3ce23c'},
            'C9':{self.MAPPINGS_ALIAS_KEY:'c9orf72', self.MAPPINGS_COLOR_KEY:'#eb3440'},

            'Ctrl-EDi022':{self.MAPPINGS_ALIAS_KEY: 'Control (EDi022)', self.MAPPINGS_COLOR_KEY: '#58cfdf'},
            'Ctrl-EDi029':{self.MAPPINGS_ALIAS_KEY: 'Control (EDi029)', self.MAPPINGS_COLOR_KEY: '#4db6c4'},
            'Ctrl-EDi037':{self.MAPPINGS_ALIAS_KEY: 'Control (EDi037)', self.MAPPINGS_COLOR_KEY: '#439da9'},
            'SALSPositive-CS7TN6':{self.MAPPINGS_ALIAS_KEY:'sALS+ (CS7TN6)', self.MAPPINGS_COLOR_KEY:'#f6ce55'},
            'SALSPositive-CS2FN3':{self.MAPPINGS_ALIAS_KEY:'sALS+ (CS2FN3)', self.MAPPINGS_COLOR_KEY:'#dbb74c'},
            'SALSPositive-CS4ZCD':{self.MAPPINGS_ALIAS_KEY:'sALS+ (CS4ZCD)', self.MAPPINGS_COLOR_KEY:'#cdac47'},
            'SALSNegative-CS0JPP':{self.MAPPINGS_ALIAS_KEY:'sALS- (CS0JPP)', self.MAPPINGS_COLOR_KEY:'#3ce23c'},
            'SALSNegative-CS0ANK':{self.MAPPINGS_ALIAS_KEY:'sALS- (CS0ANK)', self.MAPPINGS_COLOR_KEY:'#1e711e'},
            'SALSNegative-CS6ZU8':{self.MAPPINGS_ALIAS_KEY:'sALS- (CS6ZU8)', self.MAPPINGS_COLOR_KEY:'#469E46'},
            'C9-CS7VCZ':{self.MAPPINGS_ALIAS_KEY:'C9 (CS7VCZ)', self.MAPPINGS_COLOR_KEY:'#eb3440'},
            'C9-CS8RFT':{self.MAPPINGS_ALIAS_KEY:'C9 (CS8RFT)', self.MAPPINGS_COLOR_KEY:'#bc2a33'},
            'C9-CS2YNL':{self.MAPPINGS_ALIAS_KEY:'C9 (CS2YNL)', self.MAPPINGS_COLOR_KEY:'#8d1f26'},

        }
        self.COLOR_MAPPINGS_ALYSSA['Controls'] = self.COLOR_MAPPINGS_ALYSSA['Controls_Untreated']
        self.COLOR_MAPPINGS_ALYSSA['sALSPositiveCytoTDP43'] = self.COLOR_MAPPINGS_ALYSSA['sALSPositiveCytoTDP43_Untreated']
        self.COLOR_MAPPINGS_ALYSSA['sALSNegativeCytoTDP43'] = self.COLOR_MAPPINGS_ALYSSA['sALSNegativeCytoTDP43_Untreated']
        self.COLOR_MAPPINGS_ALYSSA['c9orf72ALSPatients'] = self.COLOR_MAPPINGS_ALYSSA['c9orf72ALSPatients_Untreated']

        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS
