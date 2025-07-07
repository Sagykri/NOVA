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
        self.SIZE = 30
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
            'TDP43_dox':{self.MAPPINGS_ALIAS_KEY:'TDP-43dNLS, +DOX', self.MAPPINGS_COLOR_KEY:'#90278E'}
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
            'Phalloidin':{self.MAPPINGS_ALIAS_KEY:'Cytoskeleton', self.MAPPINGS_COLOR_KEY:'#06A0E9'},
            'FMRP':{self.MAPPINGS_ALIAS_KEY:'FMRP granules', self.MAPPINGS_COLOR_KEY:'#EF218B'},

            'MERGED':{self.MAPPINGS_ALIAS_KEY:'MERGED', self.MAPPINGS_COLOR_KEY:'gray'},
            'Map2':{self.MAPPINGS_ALIAS_KEY:'Neuronal marker', self.MAPPINGS_COLOR_KEY:'gray'},
            # 'TDP43N':{self.MAPPINGS_ALIAS_KEY:'TDP43 granules 2', self.MAPPINGS_COLOR_KEY:'#06A0E9'}
        }
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

        }
        self.COLOR_MAPPINGS_ALYSSA['Controls'] = self.COLOR_MAPPINGS_ALYSSA['Controls_Untreated']
        self.COLOR_MAPPINGS_ALYSSA['sALSPositiveCytoTDP43'] = self.COLOR_MAPPINGS_ALYSSA['sALSPositiveCytoTDP43_Untreated']
        self.COLOR_MAPPINGS_ALYSSA['sALSNegativeCytoTDP43'] = self.COLOR_MAPPINGS_ALYSSA['sALSNegativeCytoTDP43_Untreated']
        self.COLOR_MAPPINGS_ALYSSA['c9orf72ALSPatients'] = self.COLOR_MAPPINGS_ALYSSA['c9orf72ALSPatients_Untreated']

        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS

        self.COLOR_MAPPINGS_NIH = {**self.COLOR_MAPPINGS_MARKERS} 
        self.COLOR_MAPPINGS_NIH.update({
            'ANAX11':{self.MAPPINGS_ALIAS_KEY:'ANXA11 granules', self.MAPPINGS_COLOR_KEY:'#18E4CF'},
            'TUJ1':{self.MAPPINGS_ALIAS_KEY:'Microtubule', self.MAPPINGS_COLOR_KEY:'#FF5733'},
            'P54':{self.MAPPINGS_ALIAS_KEY:'Paraspeckles', self.MAPPINGS_COLOR_KEY:'#66CDAA'},
            'TIA1':{self.MAPPINGS_ALIAS_KEY:'Stress granule 2', self.MAPPINGS_COLOR_KEY:'#5733FF'},
        })

        self.COLOR_MAPPINGS_FUNOVA = {
            'DAPI': {self.MAPPINGS_ALIAS_KEY: 'Nucleus', self.MAPPINGS_COLOR_KEY: '#AFBDFF'},
            'Stress-initiation': {self.MAPPINGS_ALIAS_KEY: 'Stress initiation', self.MAPPINGS_COLOR_KEY: '#4343FE'},
            'mature-Autophagosome': {self.MAPPINGS_ALIAS_KEY: 'Mature Autophagosome', self.MAPPINGS_COLOR_KEY: '#921010'},
            'Cytoskeleton': {self.MAPPINGS_ALIAS_KEY: 'Cytoskeleton', self.MAPPINGS_COLOR_KEY: '#12F986'},
            'Ubiquitin-levels': {self.MAPPINGS_ALIAS_KEY: 'Ubiquitin levels', self.MAPPINGS_COLOR_KEY: '#66CDAA'},
            'UPR-IRE1a': {self.MAPPINGS_ALIAS_KEY: 'UPR IRE1a', self.MAPPINGS_COLOR_KEY: '#18E4CF'},
            'UPR-ATF4': {self.MAPPINGS_ALIAS_KEY: 'UPR ATF4', self.MAPPINGS_COLOR_KEY: '#168FB2'},
            'UPR-ATF6': {self.MAPPINGS_ALIAS_KEY: 'UPR ATF6', self.MAPPINGS_COLOR_KEY: '#A80358'},
            'impaired-Autophagosome': {self.MAPPINGS_ALIAS_KEY: 'Impaired Autophagosome', self.MAPPINGS_COLOR_KEY: '#9968CB'},
            'Autophagy': {self.MAPPINGS_ALIAS_KEY: 'Autophagy', self.MAPPINGS_COLOR_KEY: '#D257EA'},
            'Aberrant-splicing': {self.MAPPINGS_ALIAS_KEY: 'Aberrant splicing', self.MAPPINGS_COLOR_KEY: '#E6A9EA'},
            'Parthanatos-late': {self.MAPPINGS_ALIAS_KEY: 'Parthanatos late', self.MAPPINGS_COLOR_KEY: '#F04521'},
            'Nuclear-speckles-SC35': {self.MAPPINGS_ALIAS_KEY: 'Nuclear speckles SC35', self.MAPPINGS_COLOR_KEY: '#F08F21'},
            'Splicing-factories': {self.MAPPINGS_ALIAS_KEY: 'Splicing factories', self.MAPPINGS_COLOR_KEY: '#F1CBDD'},
            'TDP-43': {self.MAPPINGS_ALIAS_KEY: 'TDP-43', self.MAPPINGS_COLOR_KEY: '#FFBF0D'},
            'Nuclear-speckles-SON': {self.MAPPINGS_ALIAS_KEY: 'Nuclear speckles SON', self.MAPPINGS_COLOR_KEY: '#37378D'},
            'DNA-damage-pH2Ax': {self.MAPPINGS_ALIAS_KEY: 'DNA damage pH2Ax', self.MAPPINGS_COLOR_KEY: 'gray'},
            'Parthanatos-early': {self.MAPPINGS_ALIAS_KEY: 'Parthanatos early', self.MAPPINGS_COLOR_KEY: '#DEDB23'},
            'Necrosis': {self.MAPPINGS_ALIAS_KEY: 'Necrosis', self.MAPPINGS_COLOR_KEY: '#AF8215'},
            'Necroptosis-HMGB1': {self.MAPPINGS_ALIAS_KEY: 'Necroptosis HMGB1', self.MAPPINGS_COLOR_KEY: '#32AC0E'},
            'Neuronal-activity': {self.MAPPINGS_ALIAS_KEY: 'Neuronal activity', self.MAPPINGS_COLOR_KEY: '#898700'},
            'DNA-damage-P53BP1': {self.MAPPINGS_ALIAS_KEY: 'DNA damage P53BP1', self.MAPPINGS_COLOR_KEY: '#ACE142'},
            'Apoptosis': {self.MAPPINGS_ALIAS_KEY: 'Apoptosis', self.MAPPINGS_COLOR_KEY: '#F0A3A3'},
            'Necroptosis-pMLKL': {self.MAPPINGS_ALIAS_KEY: 'Necroptosis pMLKL', self.MAPPINGS_COLOR_KEY: '#06A0E9'},
            'Protein-degradation': {self.MAPPINGS_ALIAS_KEY: 'Protein degradation', self.MAPPINGS_COLOR_KEY: '#EF218B'},
            'Senescence-signaling': {self.MAPPINGS_ALIAS_KEY: 'Senescence signaling', self.MAPPINGS_COLOR_KEY: '#FF5733'}
        }

        self.COLOR_MAPPINGS_FUNOVA_CONDITIONS = {
            'DAPI_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Nucleus', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Stress-initiation_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Stress initiation', self.MAPPINGS_COLOR_KEY: '#000000'},
            'mature-Autophagosome_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Mature Autophagosome', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Cytoskeleton_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Cytoskeleton', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Ubiquitin-levels_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Ubiquitin levels', self.MAPPINGS_COLOR_KEY: '#000000'},
            'UPR-IRE1a_Untreated': {self.MAPPINGS_ALIAS_KEY: 'UPR IRE1a', self.MAPPINGS_COLOR_KEY: '#000000'},
            'UPR-ATF4_Untreated': {self.MAPPINGS_ALIAS_KEY: 'UPR ATF4', self.MAPPINGS_COLOR_KEY: '#000000'},
            'UPR-ATF6_Untreated': {self.MAPPINGS_ALIAS_KEY: 'UPR ATF6', self.MAPPINGS_COLOR_KEY: '#000000'},
            'impaired-Autophagosome_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Impaired Autophagosome', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Autophagy_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Autophagy', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Aberrant-splicing_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Aberrant splicing', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Parthanatos-late_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Parthanatos late', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Nuclear-speckles-SC35_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Nuclear speckles SC35', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Splicing-factories_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Splicing factories', self.MAPPINGS_COLOR_KEY: '#000000'},
            'TDP-43_Untreated': {self.MAPPINGS_ALIAS_KEY: 'TDP-43', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Nuclear-speckles-SON_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Nuclear speckles SON', self.MAPPINGS_COLOR_KEY: '#000000'},
            'DNA-damage-pH2Ax_Untreated': {self.MAPPINGS_ALIAS_KEY: 'DNA damage pH2Ax', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Parthanatos-early_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Parthanatos early', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Necrosis_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Necrosis', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Necroptosis-HMGB1_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Necroptosis HMGB1', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Neuronal-activity_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Neuronal activity', self.MAPPINGS_COLOR_KEY: '#000000'},
            'DNA-damage-P53BP1_Untreated': {self.MAPPINGS_ALIAS_KEY: 'DNA damage P53BP1', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Apoptosis_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Apoptosis', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Necroptosis-pMLKL_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Necroptosis pMLKL', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Protein-degradation_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Protein degradation', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Senescence-signaling_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Senescence signaling', self.MAPPINGS_COLOR_KEY: '#000000'},
            
            'DAPI_stress': {self.MAPPINGS_ALIAS_KEY: 'Nucleus', self.MAPPINGS_COLOR_KEY: '#AFBDFF'},
            'Stress-initiation_stress': {self.MAPPINGS_ALIAS_KEY: 'Stress initiation', self.MAPPINGS_COLOR_KEY: '#4343FE'},
            'mature-Autophagosome_stress': {self.MAPPINGS_ALIAS_KEY: 'Mature Autophagosome', self.MAPPINGS_COLOR_KEY: '#921010'},
            'Cytoskeleton_stress': {self.MAPPINGS_ALIAS_KEY: 'Cytoskeleton', self.MAPPINGS_COLOR_KEY: '#12F986'},
            'Ubiquitin-levels_stress': {self.MAPPINGS_ALIAS_KEY: 'Ubiquitin levels', self.MAPPINGS_COLOR_KEY: '#66CDAA'},
            'UPR-IRE1a_stress': {self.MAPPINGS_ALIAS_KEY: 'UPR IRE1a', self.MAPPINGS_COLOR_KEY: '#18E4CF'},
            'UPR-ATF4_stress': {self.MAPPINGS_ALIAS_KEY: 'UPR ATF4', self.MAPPINGS_COLOR_KEY: '#168FB2'},
            'UPR-ATF6_stress': {self.MAPPINGS_ALIAS_KEY: 'UPR ATF6', self.MAPPINGS_COLOR_KEY: '#A80358'},
            'impaired-Autophagosome_stress': {self.MAPPINGS_ALIAS_KEY: 'Impaired Autophagosome', self.MAPPINGS_COLOR_KEY: '#9968CB'},
            'Autophagy_stress': {self.MAPPINGS_ALIAS_KEY: 'Autophagy', self.MAPPINGS_COLOR_KEY: '#D257EA'},
            'Aberrant-splicing_stress': {self.MAPPINGS_ALIAS_KEY: 'Aberrant splicing', self.MAPPINGS_COLOR_KEY: '#E6A9EA'},
            'Parthanatos-late_stress': {self.MAPPINGS_ALIAS_KEY: 'Parthanatos late', self.MAPPINGS_COLOR_KEY: '#F04521'},
            'Nuclear-speckles-SC35_stress': {self.MAPPINGS_ALIAS_KEY: 'Nuclear speckles SC35', self.MAPPINGS_COLOR_KEY: '#F08F21'},
            'Splicing-factories_stress': {self.MAPPINGS_ALIAS_KEY: 'Splicing factories', self.MAPPINGS_COLOR_KEY: '#F1CBDD'},
            'TDP-43_stress': {self.MAPPINGS_ALIAS_KEY: 'TDP-43', self.MAPPINGS_COLOR_KEY: '#FFBF0D'},
            'Nuclear-speckles-SON_stress': {self.MAPPINGS_ALIAS_KEY: 'Nuclear speckles SON', self.MAPPINGS_COLOR_KEY: '#37378D'},
            'DNA-damage-pH2Ax_stress': {self.MAPPINGS_ALIAS_KEY: 'DNA damage pH2Ax', self.MAPPINGS_COLOR_KEY: 'gray'},
            'Parthanatos-early_stress': {self.MAPPINGS_ALIAS_KEY: 'Parthanatos early', self.MAPPINGS_COLOR_KEY: '#DEDB23'},
            'Necrosis_stress': {self.MAPPINGS_ALIAS_KEY: 'Necrosis', self.MAPPINGS_COLOR_KEY: '#AF8215'},
            'Necroptosis-HMGB1_stress': {self.MAPPINGS_ALIAS_KEY: 'Necroptosis HMGB1', self.MAPPINGS_COLOR_KEY: '#32AC0E'},
            'Neuronal-activity_stress': {self.MAPPINGS_ALIAS_KEY: 'Neuronal activity', self.MAPPINGS_COLOR_KEY: '#898700'},
            'DNA-damage-P53BP1_stress': {self.MAPPINGS_ALIAS_KEY: 'DNA damage P53BP1', self.MAPPINGS_COLOR_KEY: '#ACE142'},
            'Apoptosis_stress': {self.MAPPINGS_ALIAS_KEY: 'Apoptosis', self.MAPPINGS_COLOR_KEY: '#F0A3A3'},
            'Necroptosis-pMLKL_stress': {self.MAPPINGS_ALIAS_KEY: 'Necroptosis pMLKL', self.MAPPINGS_COLOR_KEY: '#06A0E9'},
            'Protein-degradation_stress': {self.MAPPINGS_ALIAS_KEY: 'Protein degradation', self.MAPPINGS_COLOR_KEY: '#EF218B'},
            'Senescence-signaling_stress': {self.MAPPINGS_ALIAS_KEY: 'Senescence signaling', self.MAPPINGS_COLOR_KEY: '#FF5733'}
        }

        self.COLOR_MAPPINGS_ALS_FUNOVA: Dict[str, Dict[str, str]] = { 
        # Controls (shades of blue)
        'Control-1001733_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Control 1001733', self.MAPPINGS_COLOR_KEY: '#1F77B4'}, 
        'Control-1017118_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Control 1017118', self.MAPPINGS_COLOR_KEY: '#2A91D2'}, 
        'Control-1025045_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Control 1025045', self.MAPPINGS_COLOR_KEY: '#4BA3D8'}, 
        'Control-1048087_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Control 1048087', self.MAPPINGS_COLOR_KEY: '#6CB4DE'}, 
        'Control': {self.MAPPINGS_ALIAS_KEY: 'Control', self.MAPPINGS_COLOR_KEY: '#6CB4DE'}, 
        
        # C9orf72-HRE (shades of green)
        'C9orf72-HRE-1008566_Untreated': {self.MAPPINGS_ALIAS_KEY: 'C9orf72-HRE 1008566', self.MAPPINGS_COLOR_KEY: '#2E8B57'}, 
        'C9orf72-HRE-981344_Untreated': {self.MAPPINGS_ALIAS_KEY: 'C9orf72-HRE 981344', self.MAPPINGS_COLOR_KEY: '#3CB371'}, 
        
        # TDP-43 variants (shades of pink/red)
        'TDP--43-G348V-1057052_Untreated': {self.MAPPINGS_ALIAS_KEY: 'TDP-43 G348V 1057052', self.MAPPINGS_COLOR_KEY: '#E377C2'}, 
        'TDP--43-N390D-1005373_Untreated': {self.MAPPINGS_ALIAS_KEY: 'TDP-43 N390D 1005373', self.MAPPINGS_COLOR_KEY: '#FF66A1'} 
         }

        self.COLOR_MAPPINGS_ALS_FUNOVA['Control-1001733'] = self.COLOR_MAPPINGS_ALS_FUNOVA['Control-1001733_Untreated']
        self.COLOR_MAPPINGS_ALS_FUNOVA['Control-1017118'] = self.COLOR_MAPPINGS_ALS_FUNOVA['Control-1017118_Untreated']
        self.COLOR_MAPPINGS_ALS_FUNOVA['Control-1025045'] = self.COLOR_MAPPINGS_ALS_FUNOVA['Control-1025045_Untreated']
        self.COLOR_MAPPINGS_ALS_FUNOVA['Control-1048087'] = self.COLOR_MAPPINGS_ALS_FUNOVA['Control-1048087_Untreated']
        self.COLOR_MAPPINGS_ALS_FUNOVA['C9orf72-HRE-1008566'] = self.COLOR_MAPPINGS_ALS_FUNOVA['C9orf72-HRE-1008566_Untreated']
        self.COLOR_MAPPINGS_ALS_FUNOVA['C9orf72-HRE-981344'] = self.COLOR_MAPPINGS_ALS_FUNOVA['C9orf72-HRE-981344_Untreated']
        self.COLOR_MAPPINGS_ALS_FUNOVA['TDP--43-G348V-1057052'] = self.COLOR_MAPPINGS_ALS_FUNOVA['TDP--43-G348V-1057052_Untreated']
        self.COLOR_MAPPINGS_ALS_FUNOVA['TDP--43-N390D-1005373'] = self.COLOR_MAPPINGS_ALS_FUNOVA['TDP--43-N390D-1005373_Untreated']
        
        self.COLOR_MAPPINGS_FUNOVA_CATEGORIES = {
            # Proteostasis (Blue)
            'Stress-initiation': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'mature-Autophagosome': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'Ubiquitin-levels': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'UPR-IRE1a': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'UPR-ATF4': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'UPR-ATF6': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'impaired-Autophagosome': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'Protein-degradation': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},

            # Neuronal Cell Death / Senescence (Red)
            'Autophagy': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'Parthanatos-late': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'DNA-damage-pH2Ax': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'Parthanatos-early': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'Necrosis': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'Necroptosis-HMGB1': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'DNA-damage-P53BP1': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'Apoptosis': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'Necroptosis-pMLKL': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},

            # Synaptic and Neuronal Function (Green)
            'Cytoskeleton': {self.MAPPINGS_ALIAS_KEY: 'Synaptic and Neuronal Function', self.MAPPINGS_COLOR_KEY: '#238B45'},
            'Neuronal-activity': {self.MAPPINGS_ALIAS_KEY: 'Synaptic and Neuronal Function', self.MAPPINGS_COLOR_KEY: '#238B45'},
            'Senescence-signaling': {self.MAPPINGS_ALIAS_KEY: 'Synaptic and Neuronal Function', self.MAPPINGS_COLOR_KEY: '#238B45'},

            # DNA and RNA Defects (Purple)
            'Aberrant-splicing': {self.MAPPINGS_ALIAS_KEY: 'DNA and RNA Defects', self.MAPPINGS_COLOR_KEY: '#6A3D9A'},
            'Nuclear-speckles-SC35': {self.MAPPINGS_ALIAS_KEY: 'DNA and RNA Defects', self.MAPPINGS_COLOR_KEY: '#6A3D9A'},
            'Splicing-factories': {self.MAPPINGS_ALIAS_KEY: 'DNA and RNA Defects', self.MAPPINGS_COLOR_KEY: '#6A3D9A'},
            'Nuclear-speckles-SON': {self.MAPPINGS_ALIAS_KEY: 'DNA and RNA Defects', self.MAPPINGS_COLOR_KEY: '#6A3D9A'},

            # Pathological Protein Aggregation (Orange)
            'TDP-43': {self.MAPPINGS_ALIAS_KEY: 'Pathological Protein Aggregation', self.MAPPINGS_COLOR_KEY: '#FF7F0E'}
        }

        self.COLOR_MAPPINGS_ALS_CONDITIONS_FUNOVA: Dict[str, Dict[str, str]] = {
            # Controls (muted blue)
            'Control-1001733_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Control Untreated', self.MAPPINGS_COLOR_KEY: '#5271A5'},
            'Control-1017118_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Control Untreated', self.MAPPINGS_COLOR_KEY: '#5271A5'},
            'Control-1025045_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Control Untreated', self.MAPPINGS_COLOR_KEY: '#5271A5'},
            'Control-1048087_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Control Untreated', self.MAPPINGS_COLOR_KEY: '#5271A5'},
            'Control_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Control Untreated', self.MAPPINGS_COLOR_KEY: '#5271A5'},

            # C9orf72-HRE (soft green)
            'C9orf72-HRE-1008566_Untreated': {self.MAPPINGS_ALIAS_KEY: 'C9orf72-HRE 1008566 Untreated', self.MAPPINGS_COLOR_KEY: '#6A9E6D'},
            'C9orf72-HRE-981344_Untreated': {self.MAPPINGS_ALIAS_KEY: 'C9orf72-HRE 981344 Untreated', self.MAPPINGS_COLOR_KEY: '#6A9E6D'},
            'C9orf72-HRE_Untreated': {self.MAPPINGS_ALIAS_KEY: 'C9orf72-HRE Untreated', self.MAPPINGS_COLOR_KEY: '#6A9E6D'},

            # TDP-43 variants (muted purple)
            'TDP--43-G348V-1057052_Untreated': {self.MAPPINGS_ALIAS_KEY: 'TDP-43 G348V Untreated', self.MAPPINGS_COLOR_KEY: '#86608E'},
            'TDP--43-N390D-1005373_Untreated': {self.MAPPINGS_ALIAS_KEY: 'TDP-43 N390D Untreated', self.MAPPINGS_COLOR_KEY: '#86608E'},
            'TDP--43_Untreated': {self.MAPPINGS_ALIAS_KEY: 'TDP-43 Untreated', self.MAPPINGS_COLOR_KEY: '#86608E'},

            # Controls under stress (warm orange)
            'Control-1001733_stress': {self.MAPPINGS_ALIAS_KEY: 'Control Stress', self.MAPPINGS_COLOR_KEY: '#D08C60'},
            'Control-1017118_stress': {self.MAPPINGS_ALIAS_KEY: 'Control Stress', self.MAPPINGS_COLOR_KEY: '#D08C60'},
            'Control-1025045_stress': {self.MAPPINGS_ALIAS_KEY: 'Control Stress', self.MAPPINGS_COLOR_KEY: '#D08C60'},
            'Control-1048087_stress': {self.MAPPINGS_ALIAS_KEY: 'Control Stress', self.MAPPINGS_COLOR_KEY: '#D08C60'},
            'Control_stress': {self.MAPPINGS_ALIAS_KEY: 'Control Stress', self.MAPPINGS_COLOR_KEY: '#D08C60'},

            # C9orf72-HRE under stress (soft yellow)
            'C9orf72-HRE-1008566_stress': {self.MAPPINGS_ALIAS_KEY: 'C9orf72-HRE-1008566 Stress', self.MAPPINGS_COLOR_KEY: '#E6D96A'},
            'C9orf72-HRE-981344_stress': {self.MAPPINGS_ALIAS_KEY: 'C9orf72-HRE-981344 Stress', self.MAPPINGS_COLOR_KEY: '#E6D96A'},
            'C9orf72-HRE_stress': {self.MAPPINGS_ALIAS_KEY: 'C9orf72-HRE Stress', self.MAPPINGS_COLOR_KEY: '#E6D96A'},

            # TDP-43 variants under stress (muted red)
            'TDP--43-G348V-1057052_stress': {self.MAPPINGS_ALIAS_KEY: 'TDP-43 G348V-1057052 Stress', self.MAPPINGS_COLOR_KEY: '#C75D6A'},
            'TDP--43-N390D-1005373_stress': {self.MAPPINGS_ALIAS_KEY: 'TDP-43 N390D-1005373 Stress', self.MAPPINGS_COLOR_KEY: '#C75D6A'},
            'TDP--43_stress': {self.MAPPINGS_ALIAS_KEY: 'TDP-43 Stress', self.MAPPINGS_COLOR_KEY: '#C75D6A'}
        }

        # self.COLOR_MAPPINGS_IASTROCYTES: Dict[str, Dict[str, str]] = {
        #     'DAPI': {self.MAPPINGS_ALIAS_KEY: 'Nucleus', self.MAPPINGS_COLOR_KEY: '#AFBDFF'},
        #     'ARL13B': {self.MAPPINGS_ALIAS_KEY: 'ARL13B', self.MAPPINGS_COLOR_KEY: '#F7810F'},
        #     'PML': {self.MAPPINGS_ALIAS_KEY: 'PML Bodies', self.MAPPINGS_COLOR_KEY: '#37AFD7'},
        #     'Vimentin': {self.MAPPINGS_ALIAS_KEY: 'Vimentin', self.MAPPINGS_COLOR_KEY: '#AB7A5B'},
        #     'WDR49': {self.MAPPINGS_ALIAS_KEY: 'WDR49', self.MAPPINGS_COLOR_KEY: '#78491C'},
        #     'Calreticulin': {self.MAPPINGS_ALIAS_KEY: 'ER', self.MAPPINGS_COLOR_KEY: '#C8C512'},
        #     'TDP43': {self.MAPPINGS_ALIAS_KEY: 'TDP-43 granules', self.MAPPINGS_COLOR_KEY: '#FF98BB'},
        # }
        self.COLOR_MAPPINGS_IASTROCYTES = {**self.COLOR_MAPPINGS_MARKERS} 
        self.COLOR_MAPPINGS_IASTROCYTES.update({
            'ARL13B': {self.MAPPINGS_ALIAS_KEY: 'ARL13B', self.MAPPINGS_COLOR_KEY: '#F7810F'},
            'Vimentin': {self.MAPPINGS_ALIAS_KEY: 'Vimentin', self.MAPPINGS_COLOR_KEY: '#AB7A5B'},
            'WDR49': {self.MAPPINGS_ALIAS_KEY: 'WDR49', self.MAPPINGS_COLOR_KEY: '#78491C'},
        })

        self.COLOR_MAPPINGS_ALS_IASTROCYTES: Dict[str, Dict[str,str]] = {
            'WT_Untreated':{self.MAPPINGS_ALIAS_KEY:'Wild-Type', self.MAPPINGS_COLOR_KEY:'#37AFD7'},
            'C9_Untreated':{self.MAPPINGS_ALIAS_KEY:'C9', self.MAPPINGS_COLOR_KEY:'#AB7A5B'},
        }
        self.COLOR_MAPPINGS_ALS_IASTROCYTES['WT'] = self.COLOR_MAPPINGS_ALS_IASTROCYTES['WT_Untreated']
        self.COLOR_MAPPINGS_ALS_IASTROCYTES['C9'] = self.COLOR_MAPPINGS_ALS_IASTROCYTES['C9_Untreated']