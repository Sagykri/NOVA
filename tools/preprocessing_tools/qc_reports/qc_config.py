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
cell_lines = ['FUSHomozygous', 'TDP43', 'TBK1', 'WT', 'FUSRevertant','OPTN', 'FUSHeterozygous', 'SCNA']
cell_lines_to_cond = {'FUSHomozygous':['Untreated'], 'TDP43':['Untreated'], 'TBK1':['Untreated'],
                      'WT':['Untreated','stress'], 'FUSRevertant':['Untreated'],
                      'OPTN':['Untreated'], 'FUSHeterozygous':['Untreated'],'SCNA':['Untreated']}
cell_lines_for_disp = {'FUSHomozygous_Untreated':'FUSHomozygous', 'TDP43_Untreated':'TDP43', 
                       'TBK1_Untreated':'TBK1', 'WT_stress':'WT_stress', 'WT_Untreated':'WT_Untreated',
                        'FUSRevertant_Untreated':'FUSRevertant',
                        'OPTN_Untreated':'OPTN', 'FUSHeterozygous_Untreated':'FUSHeterozygous','SCNA_Untreated':'SNCA'}
reps = ['rep1','rep2']
colorblind_palette = sns.color_palette('colorblind')
line_colors = {
    'FUSHeterozygous': colorblind_palette[0],
    'FUSHomozygous': colorblind_palette[1],
    'FUSRevertant': colorblind_palette[2],
    'OPTN': colorblind_palette[8],
    'SNCA': colorblind_palette[4],
    'TBK1': colorblind_palette[5],
    'TDP43': colorblind_palette[6],
    'WT Untreated': colorblind_palette[9],
    'WT stress': colorblind_palette[3]
}
lines_order = line_colors.keys()
custom_palette = [line_colors[line] for line in lines_order]
expected_dapi_raw = 1100

# regular neurons FUS pertubations
fus_panels = pd.DataFrame( [['KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','NCL','ANXA11','Calreticulin','mitotracker'],
                             ['G3BP1','NONO','SQSTM1',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'PML'],
                        ['PURA','CD41','Phalloidin','PSD95', 'NEMO', 'GM130', 'FUS', 'SNCA', 'LAMP1', 'PEX14'],
                        ['DAPI']*10], columns=['A','B','C','D','E','F','G','H','I','J'],
                        index=['Cy5', 'mCherry', 'GFP','DAPI'])
fus_markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO',
           'GM130','NCL','ANXA11','Calreticulin','mitotracker',
            'KIF5A','TDP43','FMRP','CLTC','DCP1A',
            'TOMM20','FUS','SNCA','LAMP1','PML',
            'PURA','CD41','Phalloidin', 'PEX14','DAPI']

fus_marker_info = pd.DataFrame([[['Cy5']]*10 + [['mCherry']]*4 + [['GFP']]*10,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],
                          ['A'],['B'],['C'],['J'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J']]], index=['Antibody','panel'],
                         columns = ['KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','NCL','ANXA11','Calreticulin','mitotracker',
                                 'G3BP1','NONO','SQSTM1','PML',
                                 'PURA','CD41','Phalloidin', 'PSD95', 'NEMO', 'GM130', 'FUS', 'SNCA', 'LAMP1', 'PEX14']).T  #order here is important - 
fus_cell_lines = ['KOLF', 'FUSRevertant', 'FUSHeterozygous'] #'SCNA',
fus_cell_lines_to_cond = {'KOLF':['Untreated','BMAA','Cisplatin','Colchicine', 'DMSO', 'Etoposide', 'MG132', 'ML240', 'NMS873', 'SA'],
                      'FUSRevertant':['BMAA','Cisplatin','Colchicine', 'DMSO', 'Etoposide', 'MG132', 'ML240', 'NMS873', 'SA'],
                      'FUSHeterozygous':['BMAA','Cisplatin','Colchicine', 'DMSO', 'Etoposide', 'MG132', 'ML240', 'NMS873', 'SA']} #'SCNA':['Untreated'],
fus_cell_lines_for_disp = {f'{k}_{c}':f'{k}_{c}' for k,conds in fus_cell_lines_to_cond.items() for c in conds}
# fus_cell_lines_for_disp = {'FUSHomozygous_Untreated':'FUSHomozygous', 'TDP43_Untreated':'TDP43', 
#                        'TBK1_Untreated':'TBK1', 'WT_stress':'WT_stress', 'WT_Untreated':'WT_Untreated',
#                         'FUSRevertant_Untreated':'FUSRevertant',
#                         'OPTN_Untreated':'OPTN', 'FUSHeterozygous_Untreated':'FUSHeterozygous'} #'SCNA_Untreated':'SCNA',
fus_reps = ['rep1','rep2']
fus_colorblind_palette = sns.color_palette('colorblind')
fus_line_colors = {f'{k} {c}': fus_colorblind_palette[i%len(fus_colorblind_palette)] 
                   for i, (k,conds) in enumerate(fus_cell_lines_to_cond.items())
                   for c in conds}
# fus_line_colors = {
#     'KOLF Untreated': colorblind_palette[9],
#     'KOLF BMAA': colorblind_palette[3],
    
#     'FUSHeterozygous': colorblind_palette[0],
#     'FUSRevertant': colorblind_palette[2],
    
# }
fus_lines_order = fus_line_colors.keys()
fus_custom_palette = [fus_line_colors[line] for line in fus_line_colors]
fus_expected_dapi_raw = 1000

# regular neurons 28 days
days28_panels = pd.DataFrame( [['PURA'],
                             ['TDP43'],
                        ['GFP'],
                        ['DAPI']], columns=['A'],
                        index=['Cy5', 'mCherry', 'GFP','DAPI'])
days28_markers = ['PURA','TDP43','GFP','DAPI']

days28_marker_info = pd.DataFrame([[['Cy5']] + [['mCherry']] + [['GFP']],
                          [['A'],
                          ['A'],
                          ['A']]], index=['Antibody','panel'],
                         columns = ['PURA','TDP43','GFP']).T  #order here is importbant - 
days28_cell_lines = ['iW11', 'KOLF7', 'KOLF6', 'KOLF5', 'KOLF4', 'KOLF'] 
days28_cell_lines_to_cond = {line:['Untreated'] for line in days28_cell_lines} #'SCNA':['Untreated'],
days28_cell_lines_for_disp = {f'{k}_{c}':f'{k}_{c}' for k,conds in days28_cell_lines_to_cond.items() for c in conds}
# fus_cell_lines_for_disp = {'FUSHomozygous_Untreated':'FUSHomozygous', 'TDP43_Untreated':'TDP43', 
#                        'TBK1_Untreated':'TBK1', 'WT_stress':'WT_stress', 'WT_Untreated':'WT_Untreated',
#                         'FUSRevertant_Untreated':'FUSRevertant',
#                         'OPTN_Untreated':'OPTN', 'FUSHeterozygous_Untreated':'FUSHeterozygous'} #'SCNA_Untreated':'SCNA',
days28_reps = ['rep1']
days28_colorblind_palette = sns.color_palette('colorblind')
days28_line_colors = {f'{k} {c}': days28_colorblind_palette[i%len(days28_colorblind_palette)] 
                   for i, (k,conds) in enumerate(days28_cell_lines_to_cond.items())
                   for c in conds}
# fus_line_colors = {
#     'KOLF Untreated': colorblind_palette[9],
#     'KOLF BMAA': colorblind_palette[3],
    
#     'FUSHeterozygous': colorblind_palette[0],
#     'FUSRevertant': colorblind_palette[2],
    
# }
days28_lines_order = days28_line_colors.keys()
days28_custom_palette = [days28_line_colors[line] for line in days28_line_colors]
days28_expected_dapi_raw = 1000

# Opera
opera_panels = pd.DataFrame( [['KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','NCL','ANXA11','Calreticulin','mitotracker'],
                             ['G3BP1','NONO','SQSTM1',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'PML'],
                        ['PURA','CD41','Phalloidin','PSD95', 'NEMO', 'GM130', 'FUS', 'SNCA', 'LAMP1', 'PEX14'],
                        ['DAPI']*10], columns=['A','B','C','D','E','F','G','H','I','J'],
                        index=['Cy5', 'mCherry', 'GFP','DAPI'])
opera_markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO',
           'GM130','NCL','ANXA11','Calreticulin','mitotracker',
            'KIF5A','TDP43','FMRP','CLTC','DCP1A',
            'TOMM20','FUS','SNCA','LAMP1','PML',
            'PURA','CD41','Phalloidin', 'PEX14','DAPI']

opera_marker_info = pd.DataFrame([[['Cy5']]*10 + [['mCherry']]*4 + [['GFP']]*10,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],
                          ['A'],['B'],['C'],['J'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J']]], index=['Antibody','panel'],
                         columns = ['KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','NCL','ANXA11','Calreticulin','mitotracker',
                                 'G3BP1','NONO','SQSTM1','PML',
                                 'PURA','CD41','Phalloidin', 'PSD95', 'NEMO', 'GM130', 'FUS', 'SNCA', 'LAMP1', 'PEX14']).T  #order here is important - 
opera_cell_lines = ['KOLF', 'FUSRevertant', 'FUSHeterozygous'] 
opera_cell_lines_to_cond = {'KOLF':['Untreated','BMAA', 'DMSO', 'SA'],
                      'FUSRevertant':['BMAA', 'DMSO', 'SA'],
                      'FUSHeterozygous':['BMAA', 'DMSO', 'SA']} #'SCNA':['Untreated'],
opera_cell_lines_for_disp = {f'{k}_{c}':f'{k}_{c}' for k,conds in opera_cell_lines_to_cond.items() for c in conds}
# fus_cell_lines_for_disp = {'FUSHomozygous_Untreated':'FUSHomozygous', 'TDP43_Untreated':'TDP43', 
#                        'TBK1_Untreated':'TBK1', 'WT_stress':'WT_stress', 'WT_Untreated':'WT_Untreated',
#                         'FUSRevertant_Untreated':'FUSRevertant',
#                         'OPTN_Untreated':'OPTN', 'FUSHeterozygous_Untreated':'FUSHeterozygous'} #'SCNA_Untreated':'SCNA',
opera_reps = ['rep1','rep2']
opera_colorblind_palette = sns.color_palette('colorblind')
opera_line_colors = {f'{k} {c}': opera_colorblind_palette[i%len(opera_colorblind_palette)] 
                   for i, (k,conds) in enumerate(opera_cell_lines_to_cond.items())
                   for c in conds}
# fus_line_colors = {
#     'KOLF Untreated': colorblind_palette[9],
#     'KOLF BMAA': colorblind_palette[3],
    
#     'FUSHeterozygous': colorblind_palette[0],
#     'FUSRevertant': colorblind_palette[2],
    
# }
opera_lines_order = opera_line_colors.keys()
opera_custom_palette = [opera_line_colors[line] for line in opera_line_colors]
opera_expected_dapi_raw = 1000

# Opera 18 days
opera18days_panels = pd.DataFrame( [['TOMM20','TDP43','FMRP','PSD95','NEMO','GM130','FUS','SNCA','LAMP1','Tubulin','VDAC1', 'PML',],
                                    ['PURA','CD41','Phalloidin','CLTC','DCP1A','KIF5A','NCL','ANXA11','Calreticulin','PSPC1',np.nan,'PEX14'],
                                    ['G3BP1','AGO2','SQSTM1',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'HNRNPA1','NONO','mitotracker'],
                                    ['DAPI']*12],
                                  columns=['A','B','C','D','E','F','G','H','I','J', 'K', 'L'],
                                    index=['ch2', 'ch3', 'ch4','ch1'])
opera18days_markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO',
           'GM130','NCL','ANXA11','Calreticulin','mitotracker',
            'KIF5A','TDP43','FMRP','CLTC','DCP1A',
            'TOMM20','FUS','SNCA','LAMP1','PML',
            'PURA','CD41','Phalloidin', 'PEX14','DAPI',
            'Tubulin', 'PSPC1', 'VDAC1', 'AGO2', 'HNRNPA1']

opera18days_marker_info = pd.DataFrame([[['ch2']]*12 + [['ch3']]*11 + [['ch4']]*6,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'], ['L'],
                           ['A'],['B'],['C'],['D'],['E'], ['F'],['G'],['H'],['I'],['J'],['L'],
                          ['A'],['B'],['C'],['J'],['K'], ['L']]], index=['Antibody','panel'],
                         columns = ['TOMM20','TDP43','FMRP','PSD95','NEMO','GM130','FUS','SNCA','LAMP1','Tubulin','VDAC1', 'PML',
                                    'PURA','CD41','Phalloidin','CLTC','DCP1A','KIF5A','NCL','ANXA11','Calreticulin','PSPC1','PEX14',
                                    'G3BP1','AGO2','SQSTM1','HNRNPA1','NONO','mitotracker']).T  #order here is important - 
opera18days_cell_lines = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'TBK1', 'TDP43', 'FUSRevertant', 'OPTN', 'SNCA'] 
opera18days_cell_lines_to_cond = {'WT':['Untreated','stress'],
                      'FUSHomozygous':['Untreated', 'stress'],
                      'FUSHeterozygous':['Untreated'],
                      'TBK1': ['Untreated'],
                      'TDP43': ['Untreated'],
                      'FUSRevertant': ['Untreated'],
                      'OPTN': ['Untreated'],
                      'SNCA': ['Untreated']} #'SCNA':['Untreated'],
opera18days_cell_lines_for_disp = {f'{k}_{c}':f'{k}_{c}' for k,conds in opera18days_cell_lines_to_cond.items() for c in conds}
# fus_cell_lines_for_disp = {'FUSHomozygous_Untreated':'FUSHomozygous', 'TDP43_Untreated':'TDP43', 
#                        'TBK1_Untreated':'TBK1', 'WT_stress':'WT_stress', 'WT_Untreated':'WT_Untreated',
#                         'FUSRevertant_Untreated':'FUSRevertant',
#                         'OPTN_Untreated':'OPTN', 'FUSHeterozygous_Untreated':'FUSHeterozygous'} #'SCNA_Untreated':'SCNA',
opera18days_reps = ['rep1','rep2']
opera18days_colorblind_palette = sns.color_palette('colorblind')
opera18days_line_colors = {f'{k} {c}': opera_colorblind_palette[count % len(opera_colorblind_palette)] 
                           for count, (k, c) in enumerate((k, c) 
                                                          for k, conds in opera18days_cell_lines_to_cond.items() 
                                                          for c in conds)}
# fus_line_colors = {
#     'KOLF Untreated': colorblind_palette[9],
#     'KOLF BMAA': colorblind_palette[3],
    
#     'FUSHeterozygous': colorblind_palette[0],
#     'FUSRevertant': colorblind_palette[2],
    
# }
opera18days_lines_order = opera18days_line_colors.keys()
opera18days_custom_palette = [opera18days_line_colors[line] for line in opera18days_line_colors]
opera18days_expected_dapi_raw = 1200


# SPD 18 days
spd18days_panels = pd.DataFrame( [['PURA','CD41','Phalloidin',np.nan,     np.nan, np.nan, np.nan, np.nan, np.nan,            'Tubulin', np.nan, 'PEX14'],
                                  ['TOMM20','TDP43','FMRP',     'CLTC',     'DCP1A', 'KIF5A', 'NCL','ANXA11', 'Calreticulin',     'PSPC1',  'VDAC1', 'PML'],
                                    ['G3BP1','AGO2','SQSTM1',   'PSD95',    'NEMO', 'GM130', 'FUS', 'SNCA',   'LAMP1',          'HNRNPA1',  'NONO', 'mitotracker'],
                                    ['DAPI']*12],
                                  columns=['A','B','C','D','E','F','G','H','I','J', 'K', 'L'],
                                    index=['GFP', 'mCherry', 'Cy5','DAPI'])
spd18days_markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO',
           'GM130','NCL','ANXA11','Calreticulin','mitotracker',
            'KIF5A','TDP43','FMRP','CLTC','DCP1A',
            'TOMM20','FUS','SNCA','LAMP1','PML',
            'PURA','CD41','Phalloidin', 'PEX14','DAPI',
            'Tubulin', 'PSPC1', 'VDAC1', 'AGO2', 'HNRNPA1']

spd18days_marker_info = pd.DataFrame([[['GFP']]*5 + [['mCherry']]*12 + [['Cy5']]*12,
                          [['A'],['B'],['C'],['J'],['L'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'], ['L'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'], ['L']]], index=['Antibody','panel'],
                         columns = ['PURA','CD41','Phalloidin','Tubulin','PEX14',
                                    'TOMM20','TDP43','FMRP',     'CLTC',     'DCP1A', 'KIF5A', 'NCL','ANXA11', 'Calreticulin',     'PSPC1',  'VDAC1', 'PML',
                                    'G3BP1','AGO2','SQSTM1',   'PSD95',    'NEMO', 'GM130', 'FUS', 'SNCA',   'LAMP1',  'HNRNPA1',  'NONO', 'mitotracker']).T  #order here is important - 
spd18days_cell_lines = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'TBK1', 'TDP43', 'FUSRevertant', 'OPTN', 'SNCA'] 
spd18days_cell_lines_to_cond = {'WT':['Untreated','stress'],
                      'FUSHomozygous':['Untreated', 'stress'],
                      'FUSHeterozygous':['Untreated'],
                      'TBK1': ['Untreated'],
                      'TDP43': ['Untreated'],
                      'FUSRevertant': ['Untreated'],
                      'OPTN': ['Untreated'],
                      'SNCA': ['Untreated']} #'SCNA':['Untreated'],
spd18days_cell_lines_for_disp = {f'{k}_{c}':f'{k}_{c}' for k,conds in spd18days_cell_lines_to_cond.items() for c in conds}
# fus_cell_lines_for_disp = {'FUSHomozygous_Untreated':'FUSHomozygous', 'TDP43_Untreated':'TDP43', 
#                        'TBK1_Untreated':'TBK1', 'WT_stress':'WT_stress', 'WT_Untreated':'WT_Untreated',
#                         'FUSRevertant_Untreated':'FUSRevertant',
#                         'OPTN_Untreated':'OPTN', 'FUSHeterozygous_Untreated':'FUSHeterozygous'} #'SCNA_Untreated':'SCNA',
spd18days_reps = ['rep1','rep2']
spd18days_colorblind_palette = sns.color_palette('colorblind')
spd18days_line_colors = {f'{k} {c}': opera_colorblind_palette[i%len(spd18days_colorblind_palette)] 
                   for i, (k,conds) in enumerate(spd18days_cell_lines_to_cond.items())
                   for c in conds}
# fus_line_colors = {
#     'KOLF Untreated': colorblind_palette[9],
#     'KOLF BMAA': colorblind_palette[3],
    
#     'FUSHeterozygous': colorblind_palette[0],
#     'FUSRevertant': colorblind_palette[2],
    
# }
spd18days_lines_order = opera18days_line_colors.keys()
spd18days_custom_palette = [spd18days_line_colors[line] for line in spd18days_line_colors]
spd18days_expected_dapi_raw = 1200


# Microglia
microglia_cell_lines_to_cond = {'FUSHomozygous':['Untreated'], 'TDP43':['Untreated'], 'TBK1':['Untreated'],
                      'WT':['Untreated'],'SCNA':['Untreated'], 'FUSRevertant':['Untreated'],
                      'OPTN':['Untreated'], 'FUSHeterozygous':['Untreated']}
microglia_cell_lines_for_disp = {f'{cell_line}_{cond}':f'{cell_line}' 
                            for cell_line in cell_lines for cond in microglia_cell_lines_to_cond[cell_line] }
microglia_line_colors = {
    'FUSHeterozygous': colorblind_palette[0],
    'FUSHomozygous': colorblind_palette[1],
    'FUSRevertant': colorblind_palette[2],
    'OPTN': colorblind_palette[8],
    'SCNA': colorblind_palette[4],
    'TBK1': colorblind_palette[5],
    'TDP43': colorblind_palette[6],
    'WT': colorblind_palette[9]
}
microglia_lines_order = microglia_line_colors.keys()
microglia_custom_palette = [microglia_line_colors[line] for line in microglia_lines_order]

# Microglia_LPS
microglia_LPS_panels = pd.DataFrame([['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','FUS','SCNA','LAMP1','TIA1',np.nan],
             ['KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','NCL','ANXA11','Calreticulin','pNFKB','PML'],
             ['PURA','CD41','Phalloidin',np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,'PEX14'],
             ['DAPI']*11], columns=['A','B','C','D','E','F','G','H','I','J','K'],
            index=['Cy5', 'mCherry', 'GFP','DAPI'])

microglia_LPS_markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','ANXA11','Calreticulin', 'pNFKB',
                                 'KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','FUS','SCNA','LAMP1','TIA1','PML',
                                 'PURA','CD41','Phalloidin', 'PEX14','DAPI']

microglia_LPS_marker_info = pd.DataFrame([[['Cy5']]*10 + [['mCherry']]*11 + [['GFP']]*4,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],
                          ['A'],['B'],['C'],['K']]], index=['Antibody','panel'],
                         columns = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','FUS','SCNA','LAMP1','TIA1',
                                 'KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','NCL','ANXA11','Calreticulin','pNFKB','PML',
                                 'PURA','CD41','Phalloidin', 'PEX14']).T  #order here is important - taken from Lena's sheet
microglia_LPS_cell_lines = ['FUSHomozygous', 'TDP43', 'TBK1', 'WT', 'OPTN']
microglia_LPS_cell_lines_to_cond = {cell_line:['Untreated','LPS'] for cell_line in microglia_LPS_cell_lines}
microglia_LPS_cell_lines_for_disp = {f'{cell_line}_{cond}':f'{cell_line}_{cond}' 
                            for cell_line in microglia_LPS_cell_lines for cond in microglia_LPS_cell_lines_to_cond[cell_line] }
microglia_LPS_line_colors = {
    'FUSHomozygous Untreated': colorblind_palette[1],
    'OPTN Untreated': colorblind_palette[3],
    'TBK1 Untreated': colorblind_palette[5],
    'TDP43 Untreated': (0.09, 0.39, 0.07),
    'WT Untreated': colorblind_palette[9],
    'FUSHomozygous LPS': colorblind_palette[2],
    'OPTN LPS': colorblind_palette[4],
    'TBK1 LPS': colorblind_palette[6],
    'TDP43 LPS': colorblind_palette[8],
    'WT LPS': colorblind_palette[0]
}
microglia_LPS_lines_order = microglia_LPS_line_colors.keys()
microglia_LPS_custom_palette = [microglia_LPS_line_colors[line] for line in microglia_LPS_lines_order]
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
pers=['Chloroquine','DMSO1uM','Riluzole','Untreated',
                        'DMSO100uM','Edavarone','Pridopine','Tubastatin']
per_cell_lines_to_cond = {cell_line:pers for cell_line in per_cell_lines}

per_expected_dapi_raw = 200
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
dnls_cell_lines = ['TDP43']
dnls_cell_lines_to_cond = {'TDP43':['dox','Untreated']}
dnls_panels = pd.DataFrame([['G3BP1','TDP43','SQSTM1','PSD95',np.nan,'GM130','NCL','ANXA11','Calreticulin','Pericentrin','Rab5','KIFC1','mitotracker',np.nan],
                            ['KIF5A','DCP1A','FMRP','CLTC','KIF20A','TOMM20','FUS','SCNA','LAMP1','TIA1','NONO','NEMO','PML','TDP43'],
                            ['PURA','Tubulin','Phalloidin','CD41',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'PEX14',np.nan],
             ['DAPI']*14], columns=['A','B','C','D','E','F','G','H','I','J','K','L','M','N'],
            index=['Cy5', 'mCherry', 'GFP','DAPI'])
dnls_markers = ['G3BP1','TDP43N','TDP43B','SQSTM1','PSD95','GM130','NCL','ANXA11','Calreticulin','Pericentrin',
                'Rab5','KIFC1','mitotracker','KIF5A','DCP1A','FMRP','CLTC','KIF20A','TOMM20','FUS','SCNA','LAMP1',
                'TIA1','NONO','NEMO','PML','PURA','Tubulin','Phalloidin','CD41','PEX14','DAPI']

dnls_cell_lines_for_disp = {f'{cell_line}_{cond}':f'{cell_line}_{cond}' 
                            for cell_line in dnls_cell_lines for cond in dnls_cell_lines_to_cond[cell_line] }
dnls_marker_info = pd.DataFrame([[['Cy5']]*12 + [['mCherry']]*14 + [['GFP']]*5 + [['Cy5','mCherry']],
                          [['A'], ['B'], ['C'],['D'],['F'],['G'],['H'],['I'],['J'], ['K'],['L'],['M'], 
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],['L'], ['M'], ['N'], 
                          ['A'],['B'],['C'],['D'],['M'], ['B','N']]], index=['Antibody','panel'],
                         columns = ['G3BP1','TDP43B','SQSTM1','PSD95','GM130','NCL','ANXA11','Calreticulin','Pericentrin',
                'Rab5','KIFC1','mitotracker','KIF5A','DCP1A','FMRP','CLTC','KIF20A','TOMM20','FUS','SCNA','LAMP1', 
                'TIA1','NONO','NEMO','PML', 'TDP43N','PURA','Tubulin','Phalloidin','CD41','PEX14','TDP43']).T #order here is important - taken from Lena's sheet

dnls_line_colors = {
    'TDP43 Untreated': colorblind_palette[4],
    'TDP43 dox': colorblind_palette[2]
}
dnls_lines_order = dnls_line_colors.keys()
dnls_custom_palette = [dnls_line_colors[line] for line in dnls_lines_order]
dnls_expected_dapi_raw = 100*len(dnls_panels.columns)


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


### Alyssa Coyne data
AC_panels = pd.DataFrame([['DCP1A'],['Map2'],['TDP43'],['DAPI']], columns=['A'],
            index=['Cy5', 'mCherry', 'GFP','DAPI'])

AC_markers = ['DCP1A','Map2','TDP43','DAPI']

AC_reps = ['rep1', 'rep2', 'rep3', 'rep4', 'rep5', 'rep6', 'rep7', 'rep8', 'rep9', 'rep10']

AC_marker_info = pd.DataFrame([[['Cy5']]*1 + [['mCherry']]*1 + [['GFP']]*1,
                          [['A'],['A'],['A']]], index=['Antibody','panel'],
                         columns = ['DCP1A','Map2','TDP43']).T  #order here is important - taken from Lena's sheet

AC_cell_lines = ['Controls','sALS_Negative_cytoTDP43','sALS_Positive_cytoTDP43','c9orf72_ALS_patients']

AC_cell_lines_to_cond = {
                    'Controls':['Untreated'], 
                    'sALSNegativeCytoTDP43':['Untreated'], 
                    'sALSPositiveCytoTDP43':['Untreated'],
                    'c9orf72ALSPatients':['Untreated']}

AC_cell_lines_for_disp = {'Controls_Untreated':'Controls', 
                       'sALSNegativeCytoTDP43_Untreated':'sALS_Negative_cytoTDP43', 
                       'sALSPositiveCytoTDP43_Untreated':'sALS_Positive_cytoTDP43', 
                       'c9orf72ALSPatients_Untreated':'c9orf72_ALS_patients'
                      } 

AC_colorblind_palette = sns.color_palette('colorblind')
AC_line_colors = {
    'c9orf72ALSPatients': AC_colorblind_palette[8],
    'sALSPositiveCytoTDP43': AC_colorblind_palette[5],
    'sALSNegativeCytoTDP43': AC_colorblind_palette[6],
    'Controls': AC_colorblind_palette[9]
}
AC_lines_order = AC_line_colors.keys()
AC_custom_palette = [AC_line_colors[line] for line in AC_lines_order]
AC_expected_dapi_raw = 10

AC_cell_lines_to_reps = {
    'c9orf72ALSPatients': [f'rep{i}' for i in range(1,4)],
    'sALSPositiveCytoTDP43': [f'rep{i}' for i in range(1,10)],
    'sALSNegativeCytoTDP43': [f'rep{i}' for i in range(1,3)],
    'Controls': [f'rep{i}' for i in range(1,7)]
}

### NIH data
NIH_cell_lines = ['WT', 'FUSRevertant', 'FUSHomozygous', 'FUSHeterozygous']

NIH_cell_lines_to_cond = {
                    'WT':['Untreated', 'stress'], 
                    'FUSRevertant':['Untreated'], 
                    'FUSHomozygous':['Untreated'],
                    'FUSHeterozygous':['Untreated']}

NIH_markers = ['ANAX11', 'DAPI', 'SNCA', 'TUJ1', 'DCP1A', 'NEMO', 'CLTC', 'PSD95', 'CD41', 'P54',
 'TDP43', 'GM130', 'Phalloidin', 'TOMM20', 'Calreticulin', 'LAMP1', 'FMRP',
 'SQSTM1', 'G3BP1', 'KIF5A', 'PEX14', 'PML', 'PURA', 'TIA1', 'FUS', 'MitoTracker', 'NCL']

NIH_reps = ['rep1', 'rep2', 'rep3', 'rep4', 'rep5', 'rep6', 'rep7', 'rep8']


NIH_panels = pd.DataFrame([['DAPI']*11,
             ['FMRP', 'CLTC', 'DCP1A', 'ANAX11', 'PURA', 'PEX14', 'G3BP1', 'LAMP1', 'Phalloidin', 'P54', 'NCL'],
             ['SQSTM1', 'PSD95', 'NEMO', 'SNCA', 'TIA1', 'PML', 'KIF5A', 'Calreticulin', 'GM130', 'CD41', 'FUS'],
             ['TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TOMM20', 'TDP43', 'MitoTracker']], 
             columns=['A','B','C','D','E','F','G','H','I','J','K'],)
            # index=['Cy5', 'mCherry', 'GFP','DAPI'])


NIH_marker_info = pd.DataFrame([[['']]*27,
             [['D'], ['K'], ['D'], ['H'], ['C'], ['C'], ['B'], ['B'], ['J'], ['J'], ['J'], ['I'], ['I'], 
             ['I'], ['H'], ['H'], ['A'], ['A'], ['G'], ['G'], ['F'], ['F'], ['E'], ['E'], ['K'], ['K'], ['K']]], 
             index=['Antibody','panel'],
             columns = ['ANAX11', 'DAPI', 'SNCA', 'TUJ1', 'DCP1A', 'NEMO', 'CLTC', 'PSD95', 'CD41', 'P54',
             'TDP43', 'GM130', 'Phalloidin', 'TOMM20', 'Calreticulin', 'LAMP1', 'FMRP',
             'SQSTM1', 'G3BP1', 'KIF5A', 'PEX14', 'PML', 'PURA', 'TIA1', 'FUS', 'MitoTracker', 'NCL']).T  #order here is important - taken from Lena's sheet 


NIH_cell_lines_for_disp =  {f'{cell_line}_{cond}':f'{cell_line}_{cond}' 
                            for cell_line in NIH_cell_lines for cond in NIH_cell_lines_to_cond[cell_line] }

NIH_colorblind_palette = sns.color_palette('colorblind')
NIH_line_colors = line_colors = {
    'FUSHeterozygous': colorblind_palette[0],
    'FUSHomozygous': colorblind_palette[1],
    'FUSRevertant': colorblind_palette[2],
    'WT Untreated': colorblind_palette[9],
    'WT stress': colorblind_palette[3]
}
NIH_lines_order = NIH_line_colors.keys()
NIH_custom_palette = [NIH_line_colors[line] for line in NIH_lines_order]
NIH_expected_dapi_raw = 20