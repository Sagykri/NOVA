import pandas as pd
import seaborn as sns
import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from manuscript.preprocessing_config_FuNOVA_Screen import PreprocessingBaseConfigFuNOVAScreen
config = PreprocessingBaseConfigFuNOVAScreen() 

funova_cell_lines = config.CELL_LINES

funova_cell_lines_to_cond = {
    config.CELL_LINES[0]: config.CONDITIONS,
}

funova_markers = config.MARKERS


funova_reps = config.REPS


temp_panel_data = {
    "1": ["p62","DAPI", "TDP-43"], # "FK-2", 
    "2": ["ATF6", "DAPI", "pTDP-43"], # "HDGFL2", 
    "3": ["G3BP1", "Calreticulin", "DAPI", "pAMPK"],
    "4": ["Aggreagtes", "pS6", "DAPI", "Cas3"],
}  # DISCARD - 'HDGFL2', 'FK-2'

# Convert the dictionary to a DataFrame
funova_panels = pd.DataFrame.from_dict(temp_panel_data, orient="index").T

# build mapping: marker -> list of panels
marker_to_panels = {m: [] for m in funova_markers}
for panel, markers in temp_panel_data.items():
    for m in markers:
        marker_to_panels[m].append(panel)

funova_marker_info = pd.DataFrame(
    [
        [[] for _ in range(len(config.MARKERS))],  # 26 empty lists for the 'Antibody' row
        list(marker_to_panels.values())
    ],
    index=['Antibody', 'panel'],  # Two rows: 'Antibody' and 'panel'
    columns=funova_markers
).T

funova_cell_lines_for_disp =  {f'{cell_line}_{cond}':f'{cell_line}_{cond}' 
                            for cell_line in funova_cell_lines for cond in funova_cell_lines_to_cond[cell_line] }

funova_colorblind_palette = sns.color_palette('colorblind')
funova_line_colors = {f'{k} {c}': funova_colorblind_palette[i%len(funova_colorblind_palette)] 
                   for i, (k,conds) in enumerate(funova_cell_lines_to_cond.items())
                   for c in conds}
funova_lines_order = funova_line_colors.keys()
funova_custom_palette = [funova_line_colors[line] for line in funova_lines_order]

funova_expected_marker_raw = 169
funova_expected_dapi_raw = funova_expected_marker_raw * len(funova_markers)