import pandas as pd
import seaborn as sns
import numpy as np

# TODO: Pretify this

# regular neurons
panels = pd.DataFrame([['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','ANXA11','Calreticulin',np.nan,'mitotracker'],
             ['KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','FUS','SCNA','LAMP1','TIA1','PML'],
             ['PURA','CD41','Phalloidin',np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,'PEX14'],
             ['DAPI']*11], columns=['A','B','C','D','E','F','G','H','I','J','K'],
            index=['Cy5', 'mCherry', 'GFP','DAPI'])
markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','ANXA11','Calreticulin','mitotracker',
                                 'KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','FUS','SCNA','LAMP1','TIA1','PML',
                                 'PURA','CD41','Phalloidin', 'PEX14','DAPI']

marker_info = pd.DataFrame([[['Cy5']]*10 + [['mCherry']]*11 + [['GFP']]*4,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['K'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],
                          ['A'],['B'],['C'],['K']]], index=['Antibody','panel'],
                         columns = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','ANXA11','Calreticulin','mitotracker',
                                 'KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','FUS','SCNA','LAMP1','TIA1','PML',
                                 'PURA','CD41','Phalloidin', 'PEX14']).T  #order here is important - taken from Lena's sheet
cell_lines = ['FUSHomozygous', 'TDP43', 'TBK1', 'WT', 'SCNA', 'FUSRevertant','OPTN', 'FUSHeterozygous']
cell_lines_to_cond = {'FUSHomozygous':['Untreated'], 'TDP43':['Untreated'], 'TBK1':['Untreated'],
                      'WT':['Untreated','stress'],'SCNA':['Untreated'], 'FUSRevertant':['Untreated'],
                      'OPTN':['Untreated'], 'FUSHeterozygous':['Untreated']}
cell_lines_for_disp = {'FUSHomozygous_Untreated':'FUSHomozygous', 'TDP43_Untreated':'TDP43', 
                       'TBK1_Untreated':'TBK1', 'WT_stress':'WT_stress', 'WT_Untreated':'WT_Untreated',
                        'SCNA_Untreated':'SCNA','FUSRevertant_Untreated':'FUSRevertant',
                        'OPTN_Untreated':'OPTN', 'FUSHeterozygous_Untreated':'FUSHeterozygous'}
reps = ['rep1','rep2']
colorblind_palette = sns.color_palette('colorblind')
line_colors = {
    'FUSHeterozygous': colorblind_palette[0],
    'FUSHomozygous': colorblind_palette[1],
    'FUSRevertant': colorblind_palette[2],
    'OPTN': colorblind_palette[8],
    'SCNA': colorblind_palette[4],
    'TBK1': colorblind_palette[5],
    'TDP43': colorblind_palette[6],
    'WT Untreated': colorblind_palette[9],
    'WT stress': colorblind_palette[3]
}
lines_order = line_colors.keys()
custom_palette = [line_colors[line] for line in lines_order]
expected_dapi_raw = 1100
expected_raw=2200
# Perturbations
per_panels = pd.DataFrame([['Calreticulin','NCL'],
             ['NONO','SQSTM1'],
             ['PURA',np.nan],
             ['DAPI']*2], columns=['A','B'],
            index=['Cy5', 'mCherry', 'GFP','DAPI'])

per_markers = ['Calreticulin','NCL','NONO','PURA','DAPI','SQSTM1']
per_reps = ['rep1']
per_marker_info = pd.DataFrame([[['Cy5']]*2 + [['mCherry']]*2 + [['GFP']]*1,
                          [['A'],['B'],['A'],['B'],['A']]], index=['Antibody','panel'],
                         columns = ['NONO','SQSTM1','Calreticulin','NCL','PURA']).T #order here is important - taken from Lena's sheet

per_cell_lines = ['WT','TDP43']
per_cell_lines_to_cond = {'WT':['Chloroquine','DMSO1uM','Riluzole','Untreated',
                        'DMSO100uM','Edavarone','Pridopine','Tubastatin'],
                'TDP43':['Chloroquine','DMSO1uM','Riluzole','Untreated',
                        'DMSO100uM','Edavarone','Pridopine','Tubastatin']}
pers=['Chloroquine','DMSO1uM','Riluzole','Untreated',
                        'DMSO100uM','Edavarone','Pridopine','Tubastatin']
per_expected_dapi_raw = 200
per_expected_raw=200
condition_colors = {
    'Chloroquine': colorblind_palette[1],
    'DMSO100uM': colorblind_palette[2],
    'DMSO1uM': colorblind_palette[8],
    'Edavarone': colorblind_palette[4],
    'Pridopine': colorblind_palette[5],
    'Riluzole': colorblind_palette[6],
    'Untreated': colorblind_palette[9],
    'Tubastatin': colorblind_palette[3]
}
condition_order = condition_colors.keys()
per_custom_palette = [condition_colors[cond] for cond in condition_order]
per_cell_lines_for_disp = {f'{cell_line}_{per}':f'{cell_line}_{per}' for cell_line in per_cell_lines for per in pers}

# deltaNLS
dnls_cell_lines = per_cell_lines
dnls_cell_lines_to_cond = {'WT':['Untreated'], 'TDP43':['dox','Untreated']}
dnls_panels = pd.DataFrame([['G3BP1','TDP43','SQSTM1','PSD95',np.nan,'GM130','NCL','ANXA11','Calreticulin','Pericentrin','Rab5','KIFC1','mitotracker',np.nan],
                            ['KIF5A','DCP1A','FMRP','CLTC','KIF20A','TOMM20','FUS','SCNA','LAMP1','TIA1','NONO','NEMO','PML','TDP43'],
                            ['PURA','Tubulin','Phalloidin','CD41',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'PEX14',np.nan],
             ['DAPI']*14], columns=['A','B','C','D','E','F','G','H','I','J','K','L','M','N'],
            index=['Cy5', 'mCherry', 'GFP','DAPI'])
dnls_markers = ['G3BP1','TDP43','SQSTM1','PSD95','GM130','NCL','ANXA11','Calreticulin','Pericentrin',
                'Rab5','KIFC1','mitotracker','KIF5A','DCP1A','FMRP','CLTC','KIF20A','TOMM20','FUS','SCNA','LAMP1',
                'TIA1','NONO','NEMO','PML','PURA','Tubulin','Phalloidin','CD41','PEX14','DAPI']

dnls_cell_lines_for_disp = {f'{cell_line}_{cond}':f'{cell_line}_{cond}' 
                            for cell_line in dnls_cell_lines for cond in dnls_cell_lines_to_cond[cell_line] }
dnls_marker_info = pd.DataFrame([[['Cy5']]*11 + [['mCherry']]*13 + [['GFP']]*5 + [['Cy5','mCherry']],
                          [['A'],['C'],['D'],['F'],['G'],['H'],['I'],['J'], ['K'],['L'],['M'], 
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],['L'], ['M'], 
                          ['A'],['B'],['C'],['D'],['M'], ['B','N']]], index=['Antibody','panel'],
                         columns = ['G3BP1','SQSTM1','PSD95','GM130','NCL','ANXA11','Calreticulin','Pericentrin',
                'Rab5','KIFC1','mitotracker','KIF5A','DCP1A','FMRP','CLTC','KIF20A','TOMM20','FUS','SCNA','LAMP1',
                'TIA1','NONO','NEMO','PML', 'PURA','Tubulin','Phalloidin','CD41','PEX14','TDP43']).T #order here is important - taken from Lena's sheet

dnls_line_colors = {
    'TDP43 Untreated': colorblind_palette[4],
    'WT Untreated': colorblind_palette[9],
    'TDP43 dox': colorblind_palette[2]
}
dnls_lines_order = dnls_line_colors.keys()
dnls_custom_palette = [dnls_line_colors[line] for line in dnls_lines_order]
dnls_expected_dapi_raw = 100*len(dnls_panels.columns)
dnls_expected_raw=100*2*len(dnls_panels.columns)


# NP
np_cell_lines = ['WT','KO']
np_cell_lines_to_cond = {cell_line:['Untreated','HPBCD'] for cell_line in np_cell_lines}
np_panels = pd.DataFrame([['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130', 'NCL','ANXA11','Calreticulin','PML'],
                            ['KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','FUS','SCNA','LAMP1','mitotracker'],
                            ['PURA','CD41','Phalloidin', np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'PEX14'],
             ['DAPI']*10], columns=['A','B','C','D','E','F','G','H','I','K'],
            index=['mCherry', 'Cy5', 'GFP','DAPI'])
np_markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130', 'NCL','ANXA11','Calreticulin','PML',
              'KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','FUS','SCNA','LAMP1','mitotracker',
              'PURA','CD41','Phalloidin','PEX14',
              'DAPI']

np_cell_lines_for_disp = {f'{cell_line}_{cond}':f'{cell_line}_{cond}' 
                            for cell_line in np_cell_lines for cond in np_cell_lines_to_cond[cell_line] }
np_marker_info = pd.DataFrame([[['Cy5']]*10 + [['mCherry']]*10 + [['GFP']]*4 ,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['K'],
                           ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['K'],
                           ['A'],['B'],['C'],['K']]], index=['Antibody','panel'],
                         columns = ['KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','FUS','SCNA','LAMP1','mitotracker',
                                    'G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130', 'NCL','ANXA11','Calreticulin','PML',
                                    'PURA','CD41','Phalloidin','PEX14']).T #order here is important - taken from Lena's sheet

np_line_colors = {
    'WT Untreated': colorblind_palette[9],
    'WT HPBCD': colorblind_palette[0],
    'KO Untreated': colorblind_palette[2],
    'KO HPBCD': colorblind_palette[4]
}
np_lines_order = np_line_colors.keys()
np_custom_palette = [np_line_colors[line] for line in np_lines_order]
np_expected_dapi_raw = 100*len(np_panels.columns)
np_expected_raw=100*2*len(np_panels.columns)