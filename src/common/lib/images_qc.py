import os
import pandas as pd
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import cv2
from src.common.lib.preprocessing_utils import rescale_intensity
from scipy.stats import f_oneway
#from scipy.stats import dunnett
from IPython.display import display, HTML
from src.common.lib.image_sampling_utils import sample_images_all_markers_all_lines, sample_images_all_markers
from multiprocessing import Pool
import matplotlib
import contextlib
import io
import pathlib
from src.common.lib.qc_config_tmp import *
from src.common.lib.calc_dataset_variance import _multiproc_calc_variance

MOMAPS_HOME = '/home/labs/hornsteinlab/Collaboration/MOmaps' # because I'm running from Sagy's user
BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR_RAW = os.path.join(BASE_DIR,'input','images','raw','SpinningDisk')

INPUT_DIR_PROC = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')

def sample_and_calc_variance(INPUT_DIR, batch, sample_size_per_markers=200, num_markers=26):
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, batch)

    images = sample_images_all_markers_all_lines(INPUT_DIR_BATCH, sample_size_per_markers, num_markers)
    
    variance = _multiproc_calc_variance(images_paths=images)
    
    return variance


def validate_files_proc(path, batch_df, bad_files, marker_info, cell_lines_for_disp):
    path_split = path.split('/')
    cur_marker = path_split[-1]
    cur_cond = path_split[-2]
    cur_cell_line = path_split[-3]
    all_files_of_marker = os.listdir(path)
    cell_line_for_disp = cell_lines_for_disp[f'{cur_cell_line}_{cur_cond}']
    if cur_marker !='DAPI':
        cur_panels = marker_info.loc[cur_marker, 'panel']
        cur_antybodies = marker_info.loc[cur_marker, 'Antibody']
        for rep in batch_df.index.get_level_values(1):
            len_rep = len([file for file in all_files_of_marker if rep in file])
            batch_df.loc[cur_marker,rep][cell_line_for_disp] = len_rep

    else:
        cur_antybodies = ['DAPI']
        for rep in batch_df.index.get_level_values(1):
            len_rep = len([file for file in all_files_of_marker if rep in file])
            batch_df.loc[cur_marker,rep][cell_line_for_disp] = len_rep

    for file in all_files_of_marker:
        try:
            size = os.path.getsize(os.path.join(path, file))
            if size < 100000: #size in bytes
                bad_files.append(f'{path}, {file} small size ({size/1000} kB)')
        except:
            bad_files.append(f'{path}, {file} cannot read')
        good_file = False
        if cur_marker!='DAPI':
            for i, antibody in enumerate(cur_antybodies):
                if f'panel{cur_panels[i]}' in file and antibody  in file and cur_cell_line in file:
                    good_file = True
                    break
        else:
            for antibody in cur_antybodies:
                if antibody in file and cur_cell_line in file:
                    good_file = True
                    break
        if not good_file:
                bad_files.append(f'{path}, {file}')
    return bad_files, batch_df

def validate_files_raw(path, batch_df, bad_files, marker_info,cell_lines_for_disp):
    path_split = path.split('/')
    cur_marker = path_split[-1]
    cur_cond = path_split[-3]
    cur_cell_line = path_split[-5]
    all_files_of_marker_rep = os.listdir(path)
    cell_line_for_disp = cell_lines_for_disp[f'{cur_cell_line}_{cur_cond}']
    cur_panel = path_split[-4]
    cur_rep = path_split[-2]
    if cur_marker !='DAPI':
        cur_antybodies = marker_info.loc[cur_marker, 'Antibody']
    else:
        cur_antybodies = ['DAPI']
    if pd.isna(batch_df.loc[cur_marker,cur_rep][cell_line_for_disp]):
        batch_df.loc[cur_marker,cur_rep][cell_line_for_disp] = len(all_files_of_marker_rep)
    else:
        batch_df.loc[cur_marker,cur_rep][cell_line_for_disp] += len(all_files_of_marker_rep)

    for file in all_files_of_marker_rep:
        if '.tif' not in file:
            bad_files.append(file)
            continue
        try:
            size = os.path.getsize(os.path.join(path, file))
            if size < 2049000: #size in bytes
                bad_files.append(f'{path}, {file} small size ({size/1000} kB)')
                continue
        except:
            bad_files.append(f'{path}, {file} cannot read')
            continue
        good_file = False
        for antibody in cur_antybodies:
            if antibody in file:
                good_file = True
                break
        if not good_file:
                bad_files.append(f'{path}, {file}')
    return bad_files, batch_df

                        
def validate_folder_structure(root_dir, folder_structure, missing_paths, bad_files, batch_df,
                               marker_info, cell_lines_for_disp, proc=False):
    for name, content in folder_structure.items():
        path = os.path.join(root_dir, name)

        if not os.path.exists(path):
            #print(f"Invalid path: {path}")
            missing_paths.append(path)
            continue

        if isinstance(content, dict):
            validate_folder_structure(path, content, missing_paths, bad_files, batch_df, marker_info, cell_lines_for_disp, 
                                      proc=proc)
        else: # end of recursion of folders, need to check files
            if proc:
                bad_files, batch_df = validate_files_proc(path, batch_df, bad_files, marker_info, cell_lines_for_disp)
            else:
                bad_files, batch_df = validate_files_raw(path, batch_df, bad_files, marker_info, cell_lines_for_disp)

                
    return missing_paths, bad_files, batch_df

    



def display_diff(batches, raws, procs, plot_path, fig_height=8, fig_width=8):
    for batch_proc, batch_raw, batch in zip(procs, raws,batches):
        diff = batch_raw - batch_proc
        print(batch)
        plot_table_diff(diff, plot_path, batch, fig_height, fig_width)
        print('=' * 8)


def get_array_sum(array_string):
    # Remove square brackets and split the string into individual elements
    elements = array_string[1:-1].split()
    # Convert elements to integers and create the NumPy array
    array = np.array([int(elem) for elem in elements])
    return np.sum(array)

def log_files_qc(LOGS_PATH):
    files_pds = []

    # Go over all files under logs
    for file in os.listdir(LOGS_PATH):
        # Take only "cell_count_stats" CSV files
        if file.endswith(".csv") and file.startswith("cell_count_stats"):

            # Load each CSV
            df = pd.read_csv(os.path.join(LOGS_PATH,file), 
                             index_col=None, 
                             header=0, 
                             # NY: converters make the code slow...
                             #converters={'cells_counts': pd.eval, 'whole_cells_counts': pd.eval}
                            )

            ##print(file, df.shape) 

            # Combine them to a single dataframe
            if (not df.empty):
                files_pds.append(df)

    print("\nTotal of", len(files_pds), "files were read.")
    all_df = pd.concat(files_pds, axis=0).reset_index()
    print("Before dup handeling ", all_df.shape)
    
    #handle duplicates
    all_df['site_num'] = all_df.filename.str.split("_").str[-1]
    # drop rows with duplicated "filename-batch-cellline-panel-site"  combination, but keep the row with max "cells_count_mean"
    df_tmp = all_df.sort_values('cells_count_mean', ascending=False).drop_duplicates(subset=['filename','batch', 'cell_line', 'panel', 'condition', 'rep', 'marker'], 
                                keep='first',
                                inplace=False).sort_index()

    print("After duplication removal #1:", df_tmp.shape)

    # Now handle correpted duplicated rows (same batch-cellline-panel-site, but different file name)
    _subset=['site_num', 'batch', 'cell_line', 'panel', 'condition', 'rep', 'marker', 
            'cells_counts', 'cells_count_mean', 'cells_count_std',
            'whole_cells_counts', 'whole_cells_count_mean', 'whole_cells_count_std',
            'n_valid_tiles', 'cells_count_in_valid_tiles_mean',
            'cells_count_in_valid_tiles_std',
            'whole_cells_count_in_valid_tiles_mean',
            'whole_cells_count_in_valid_tiles_std']


    df = df_tmp.drop_duplicates(subset=_subset,inplace=False)

    print("After duplication removal #2:", df.shape)
    
    #fix some name issues (if exists) and make pretty
    df = df.copy()
    df.loc[df.cell_line=='FUSHetero', 'cell_line'] = 'FUSHeterozygous'
    df.loc[df.cell_line=='FUSHomo', 'cell_line'] = 'FUSHomozygous'
    df.loc[df.condition=='stressed', 'condition'] = 'stress'

    df['cell_line_cond'] = df['cell_line']
    df.loc[df.cell_line=='WT', 'cell_line_cond'] = df.cell_line + " " + df.condition
    df.loc[df.batch=='Perturbations', 'cell_line_cond'] = df.cell_line + " " + df.condition

    if 'dox' in np.unique(df.condition) or 'HPBCD' in np.unique(df.condition):
        df['cell_line_cond'] = df.cell_line + " " + df.condition

    df['site_cells_counts'] = df['cells_counts'].apply(get_array_sum)
    df['site_whole_cells_counts'] = df['whole_cells_counts'].apply(get_array_sum)
    return df.sort_values(by=['batch'])


def create_folder_structure(folder_type, markers,cell_lines_to_cond, reps, panels):
    folder_structure = {}
    if folder_type == 'processed':
        for cell_line in cell_lines_to_cond.keys():
            folder_structure[cell_line] = {cond : {marker: {marker} for marker in markers} 
                                                for cond in cell_lines_to_cond[cell_line]}

    elif folder_type == 'raw':
        for cell_line in cell_lines_to_cond.keys():
            folder_structure[cell_line] = {f'panel{panel}':
                                            {cond: 
                                            {f"{rep}":
                                            {marker:{marker} for marker in panels[panel] if not pd.isna(marker)} for rep in reps}
                                            for cond in cell_lines_to_cond[cell_line]}
                                                for panel in panels.columns}
    return folder_structure


    
color_light_green = '#8DF980'
color_yellow = 'yellow'
color_gray = 'gray'

# Function to apply custom colors based on both values and index
def apply_color(value):
    # Check the conditions and return the corresponding color
    if (value == 100):
        return color_light_green
    elif (0 < value < 100):
        return color_yellow
    else:
        return color_gray
    
def apply_color_dapi(value, expected=1100):
    if value==expected:
        return color_light_green
    if 0 < value < expected:
        return color_yellow
    else:
        return color_gray

def apply_color_diff(value):
    if value==0:
        return color_light_green
    elif value > 0 :
        return color_yellow
    else:
        return color_gray
    
def plot_table(df, file_name, plot_path, reps, expected_dapi, fig_height=8, fig_width=8):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colored_df_without_DAPI = df.drop('DAPI', level=0).applymap(apply_color)
    dapi_index_data = [['DAPI']*len(reps),reps]
    
    dapi = df.loc['DAPI']
    dapi  = dapi.set_index(pd.MultiIndex.from_arrays(dapi_index_data))
    colored_df_DAPI = dapi.applymap(lambda x: apply_color_dapi(x, expected_dapi))
    colored_df = pd.concat([colored_df_without_DAPI, colored_df_DAPI])
    colored_df = colored_df.reset_index(level=1)
    colored_df['Rep'] = 'white'
    df_reset = df.reset_index(level=1)
    dapi_row = df_reset.loc['DAPI']
    df_reset = df_reset.drop('DAPI')
    df_reset = df_reset.append(dapi_row)
    col_labels = [col.replace("_", "\n") for col in df_reset.columns]
    table = ax.table(cellText=df_reset.applymap(str).values,
             rowLabels=df_reset.index,
             colLabels=col_labels, #df_reset.columns,
             cellLoc='center',
             rowLoc='center',
             loc='center',
             cellColours=colored_df.values,
             bbox=[0, 0, 3, 3],
             colWidths=[0.05]+ [0.1] * (len(df_reset.columns)-1))
    # cell_props = table.get_celld()
    # height=0.01
    # for i in range(df.shape[0]+1):
    #     for j in range(df.shape[1]+1):
    #         cell_props[(i,j)].set_height(height)
    #     if i >0:
    #         cell_props[(i,-1)].set_height(height)
    plt.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    fig.set_size_inches(5, 5)  # Example: width=10 inches, height=6 inches
    pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f'{file_name}.png'))

    # Displaying the plot with a smaller figure size in the notebook
    #plt.figure(figsize=(8, fig_height))  # Example: width=8 inches, height=4 inches
    fig.set_size_inches(fig_width, fig_height)
    plt.show()

def plot_table_diff(df, plot_path, file_name,fig_height=8, fig_width=8):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colored_df = df.applymap(apply_color_diff)
    colored_df = colored_df.reset_index(level=1)
    colored_df['Rep'] = 'white'
    df_reset = df.reset_index(level=1)
    col_labels = [col.replace("_", "\n") for col in df_reset.columns]

    table = ax.table(cellText=df_reset.applymap(str).values,
             rowLabels=df_reset.index,
             colLabels=col_labels, #df_reset.columns,
             cellLoc='center',
             rowLoc='center',
             loc='center',
             cellColours=colored_df.values,
             bbox=[0, 0, 3, 3],
             colWidths=[0.05] + [0.1] * (len(df_reset.columns)-1))
    plt.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f'{file_name}_diff.png'))
    plt.show()

def run_validate_folder_structure(root_dir, proc, panels, markers,plot_path, marker_info,
                                    cell_lines_to_cond, reps, cell_lines_for_disp,
                                    expected_dapi_raw,fig_height=8, fig_width=8,
                                    batches=[f'batch{i}' for i in range(3,10)]):
    folder_type = 'processed' if proc else 'raw'
    folder_structure = create_folder_structure(folder_type, markers,cell_lines_to_cond, reps, panels)
    batch_dfs = []
    for batch in batches: 
        print(batch)
        # Specify the root directory to validate
        batch_root_dir = os.path.join(root_dir, batch)
        index_data = [[marker for marker in markers for _ in reps],reps * len(markers)]
        multi_index = pd.MultiIndex.from_arrays(index_data, names=['Marker', 'Rep'])
        batch_df = pd.DataFrame(index=multi_index, columns=cell_lines_for_disp.values())

        # Validate the folder structure and track missing paths
        missing_paths, bad_files, batch_df = validate_folder_structure(batch_root_dir, folder_structure, [], [],
                                                                       batch_df,marker_info, 
                                                                       cell_lines_for_disp, proc=proc)
        if len(missing_paths) == 0:
            print("Folder structure is valid.")
        else:
            print("Folder structure is invalid. Missing paths:")
            for path in missing_paths:
                print(path)
        if len(bad_files) == 0:
            print('All files exists.')
        else:
            print('Some files are bad:')
            for file in bad_files:
                print(file)

        title = f'{folder_type}_table_{batch}'
        plot_table(batch_df, title, plot_path, reps, expected_dapi_raw, fig_height,fig_width)
        print('=' * 8)
        batch_dfs.append(batch_df)
    print('=' * 20)
    return batch_dfs
    

def plot_cell_count(df, order, custom_palette, whole_cells=False, norm=False):
    y = 'site_whole_cells_counts' if whole_cells else 'site_cells_counts'
    if len(np.unique(df.batch))==1:
        ylabel="count"
        if norm:
            max_average_cell_line = df.groupby('cell_line_cond')[y].mean().max()
            print("normalizing by ", df.groupby('cell_line_cond')[y].mean().idxmax(), " average is ",
                df.groupby('cell_line_cond')[y].mean().max())
            df['percentage'] = (df[y] / max_average_cell_line)*100
            y='percentage'
            ylabel = '%'
        fig, axs = plt.subplots(figsize=(12,6))
        c = sns.barplot(data=df, x='cell_line', hue='condition', y=y,
                     hue_order=order, palette = custom_palette,
                       errorbar='sd', ax=axs)
        c.legend(title='Cell Line', loc='upper left', bbox_to_anchor=(1, 0.8), fontsize=14)
        c.set_ylabel(ylabel)

    else:
        no_batches = len(np.unique(df.batch))
        ylabel="count"

        if norm:
            max_average_cell_line = df.groupby('cell_line_cond')[y].mean().max()
            print("normalizing by ", df.groupby('cell_line_cond')[y].mean().idxmax(), " average is ",
                df.groupby('cell_line_cond')[y].mean().max())
            df['percentage'] = (df[y] / max_average_cell_line)*100
            y='percentage'
            ylabel = '%'

        fig, axs = plt.subplots(nrows=1, ncols=no_batches, sharey=True, sharex=False, figsize=(12,6))
        fig.subplots_adjust(wspace=0)
        for i, (batch_name, batch) in enumerate(df.groupby('batch')):
            c = sns.barplot(data=batch, x='rep', hue='cell_line_cond', y=y, hue_order = order, 
                            ax=axs[i], palette=custom_palette, errorbar='sd')
            c.set_xlabel(batch_name, fontsize=12) 
            c.tick_params(axis='x', labelsize=10)
            # # ANOVA test
            # for j, (rep, rep_data) in enumerate(batch.groupby('rep')):
            #     anova_data = []
            #     group_labels = []
            #     for cell_line_cond, cond_data in rep_data.groupby('cell_line_cond'):
            #         anova_data.append(cond_data[y].values)
            #         group_labels.append(cell_line_cond)
            #     anova_f, anova_pvalue = f_oneway(*anova_data)
            #     anova_text = f"{rep} {anova_pvalue:.10f} {anova_f}"
            #     print(f'{batch_name} {anova_text}')
                # if anova_pvalue < 0.05:
                #     # Dunnett's test (comparing each group to the control)
                #     control_group_idx = group_labels.index('WT Untreated')
                #     control_group_data = anova_data[control_group_idx]
                #     dunnett_pvalues = dunnett(*anova_data, control=control_group_data)

                #     dunnett_text = "\n".join([f"{group_labels[i]} vs. WT Untreated: {p:.3f}" for i, p in enumerate(dunnett_pvalues) if i != control_group_idx])
                #     print(dunnett_text)
                # #c.text(0.5*j, 0.95, anova_text, transform=c.transAxes, ha='center', fontsize=12)

            if 0<i<no_batches-1: #middle plots
                c.spines['left'].set_visible(False)
                c.spines['right'].set_visible(False)
                c.legend_.remove()
                c.set_ylabel('')
            if i==no_batches-1:
                c.spines['left'].set_visible(False)
                c.set_ylabel('')
                c.legend(title='Cell Line', loc='upper left', bbox_to_anchor=(1, 0.8), fontsize=14)

            if i==0:
                c.spines['right'].set_visible(False)
                c.legend_.remove()
                c.set_ylabel(ylabel)
    title = 'Cell Count Average per Site with STD'
    if whole_cells:
        title = f'Whole {title}'
    plt.suptitle(title)
    plt.show()

def plot_sites_count(df, expected, order, custom_palette, split_to_reps=False):
    title = 'Number of Raw images in each batch and cell line'
    if not split_to_reps:
        plt.figure(figsize=(12, 8))
        if len(np.unique(df.batch))==1:
            x = 'cell_line'
            hue = 'condition'
        else:
            x='batch'
            hue='cell_line_cond'
        ax = sns.countplot(data=df, x=x, hue=hue, hue_order=order, palette=custom_palette)
        heights = [p.get_height() for p in ax.patches]
        # Loop through the patches and set the color based on height
        for patch, height in zip(ax.patches, heights):
            if height == expected:
                patch.set_facecolor('gray')

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.8))
        ax.set_ylabel('Site Count')
        ax.set_title(f'{title}\nexpected count = {expected}', fontsize=20)
        plt.show()

    elif split_to_reps:
        expected=int(expected/2)
        no_batches = len(np.unique(df.batch))
        fig, axs = plt.subplots(nrows=1, ncols=no_batches, sharey=True, sharex=False, figsize=(12,6))
        fig.subplots_adjust(wspace=0, hspace=0)
        for i, (batch_name, batch) in enumerate(df.groupby('batch')):
            c = sns.countplot(data=batch, x='rep', hue='cell_line_cond', hue_order = order, 
                            ax=axs[i], palette=custom_palette)
            c.set_xlabel(batch_name, fontsize=12) 
            c.tick_params(axis='x', labelsize=10)
            if 0<i<no_batches-1: #middle plots
                c.spines['left'].set_visible(False)
                c.spines['right'].set_visible(False)
                c.legend_.remove()
                c.set_ylabel('')
            if i==no_batches-1:
                c.spines['left'].set_visible(False)
                c.set_ylabel('')
                c.legend(title='Cell Line', loc='upper left', bbox_to_anchor=(1, 0.8), fontsize=14)

            if i==0:
                c.spines['right'].set_visible(False)
                c.legend_.remove()
                c.set_ylabel('Site Count')
            heights = [p.get_height() for p in axs[i].patches]
            # Loop through the patches and set the color based on height
            for patch, height in zip(axs[i].patches, heights):
                if height == expected:
                    patch.set_facecolor('gray')
        plt.suptitle(f'{title}\nexpected count = {expected}', fontsize=20)
        plt.show()



def calc_hist_from_files_raw(path, batch_df_raw, batch_df_norm):
    path_split = path.split('/')
    cur_marker = path_split[-1]
    cur_cond = path_split[-3]
    cur_cell_line = path_split[-5]
    all_files_of_marker_rep = os.listdir(path)
    cell_line_for_disp = cur_cell_line
    if cur_cell_line=='WT':
        cell_line_for_disp = f'{cur_cell_line}_{cur_cond}'
    random.seed(42)
    subsampled_files = random.sample(all_files_of_marker_rep, 3)
    for file in subsampled_files:
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH).flatten()
        img_rescale = rescale_intensity(img)
        bins_raw = np.concatenate(([0], np.arange(350,1000, 20), [1000, 2**16]))
        bins_rescale = np.arange(0,1.1, 0.1)

        cur_hist, _ = np.histogram(img, bins=bins_raw)
        hist_with_site_count =  np.concatenate([cur_hist,[1]])

        cur_hist_rescale, _ = np.histogram(img_rescale, bins=bins_rescale)
        hist_with_site_count_rescale =  np.concatenate([cur_hist_rescale,[1]])
        for df, hist in zip([batch_df_raw, batch_df_norm], [hist_with_site_count, hist_with_site_count_rescale]):
            if pd.isna(df.loc[cell_line_for_disp,cur_marker]).any():
                df.loc[cell_line_for_disp,cur_marker] = hist
            else: 
                df.loc[cell_line_for_disp,cur_marker] += hist
    return  batch_df_raw, batch_df_norm


                      
def calc_hists_per_batch(root_dir, folder_structure, batch_df_raw, batch_df_norm):
    for name, content in folder_structure.items():
        path = os.path.join(root_dir, name)
        if not os.path.exists(path):
            continue

        if isinstance(content, dict):
            batch_df_raw, batch_df_norm = calc_hists_per_batch(path, content, batch_df_raw, batch_df_norm)
        
        else: # end of recursion of folders
            batch_df_raw, batch_df_norm = calc_hist_from_files_raw(path, batch_df_raw, batch_df_norm)
                
    return batch_df_raw, batch_df_norm

def run_calc_hist(batches, root_directory_raw, cell_lines_for_disp, markers):
    for batch in [f'batch{i}' for i in batches]:# + ['Perturbations']:
        print(batch)
        batch_root_dir = os.path.join(root_directory_raw, batch)
        cell_lines_for_df = [cell_line for cell_line in cell_lines_for_disp.values() for _ in range(26)]
        batch_df_raw = pd.DataFrame(index=[cell_lines_for_df, markers*len(cell_lines_for_disp.values())], 
                                    columns=np.concatenate(([0], np.arange(350,1000, 20), [1000])))
        batch_df_raw['site_count'] = np.nan
        batch_df_norm = pd.DataFrame(index=[cell_lines_for_df, markers*len(cell_lines_for_disp.values())], 
                                    columns=np.arange(0,1, 0.1))
        batch_df_norm['site_count'] = np.nan
        batch_df_raw, batch_df_norm = calc_hists_per_batch(batch_root_dir, create_folder_structure('raw'), batch_df_raw, batch_df_norm)
        mean_hist_raw = batch_df_raw.copy()
        mean_hist_raw[batch_df_raw.columns.difference(['site_count'])] = batch_df_raw.drop(columns=['site_count']).div(batch_df_raw['site_count'], axis=0).astype(int)
        
        mean_hist_rescale = batch_df_norm.copy()
        mean_hist_rescale[batch_df_norm.columns.difference(['site_count'])] = batch_df_norm.drop(columns=['site_count']).div(batch_df_norm['site_count'], axis=0).astype(int)

        for hist_df, name in zip([mean_hist_raw, mean_hist_rescale], ['raw', 'rescaled']):
            fig, axs = plt.subplots(figsize=(15, 8), ncols=3, nrows=3, sharey=True, dpi=200)
            for j, (cell_line, cell_line_df) in enumerate(hist_df.drop(columns=['site_count']).groupby(level=[0])):
                df = cell_line_df.reset_index(level=0, drop=True).T

                # Generate positions for the bars using numpy.arange
                bar_positions = np.arange(df.shape[0])
                ax = axs[j//3, j%3]
                # Plot each column separately using a different color for each column
                for col in df.columns:
                    ax.bar(bar_positions, df.loc[:,col], alpha=0.7, label=col)
                xticks_size = 8 if name=='rescaled' else 3
                ax.set_xticks(bar_positions,[round(idx, 1) for idx in df.index],  fontsize=xticks_size)
                ax.set_xlabel('Intestiy value', fontsize=8)
                if j%3==0:
                    ax.set_ylabel('Count', fontsize=10)
                    ax.tick_params(axis='y', labelsize=8)
                ax.set_title(f'{cell_line}', fontsize=8)
                ax.grid(False)
                
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=13, bbox_to_anchor=(0.5, -0.1), fontsize='xx-small')
            plt.suptitle(f'{name} {batch}')
            plt.tight_layout()
            plt.show()

def _calc_hist_raw(paths):
    bins_raw = np.concatenate(([0], np.arange(350,1000, 20), [1000, 2**16]))
    bins_rescale = np.arange(0,1.1, 0.1)
    raw_hist = np.zeros(len(bins_raw), dtype=np.int64)
    norm_hist = np.zeros(len(bins_rescale), dtype=np.int64)
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH).flatten()
        img_rescale = rescale_intensity(img)

        cur_hist, _ = np.histogram(img, bins=bins_raw)
        hist_with_site_count =  np.concatenate([cur_hist,[1]])

        cur_hist_rescale, _ = np.histogram(img_rescale, bins=bins_rescale)
        hist_with_site_count_rescale =  np.concatenate([cur_hist_rescale,[1]])
        raw_hist += hist_with_site_count
        norm_hist += hist_with_site_count_rescale
            
    return raw_hist, norm_hist

def _calc_hist_proc(paths):
    bins_rescale = np.arange(0,1.1, 0.1)
    norm_hist = np.zeros(len(bins_rescale), dtype=np.int64)
    
    for path in paths:
        img = np.load(path)
        for i in range(img.shape[0]):
            tile = img[i,:,:,0].flatten() # take one tile, always target channel
            cur_hist, _ = np.histogram(tile, bins=bins_rescale)
            hist_with_site_count =  np.concatenate([cur_hist,[1]])
            norm_hist += hist_with_site_count
            
    return norm_hist


def create_sublists_by_marker_cell_line(images, raw, show_cond=False):
    sublists_dict = {}

    for file_path in images:
        # Extract marker and cell line from the file path
        parts = file_path.split("/")
        if raw:
            marker = parts[-2]
            cur_cond = parts[-4]
            cur_cell_line = parts[-6]
        elif not raw:
            marker = parts[-2]
            cur_cond = parts[-3]
            cur_cell_line = parts[-4]
        cell_line_for_disp = cur_cell_line
        if cur_cell_line=='WT' or show_cond:
            cell_line_for_disp = f'{cur_cell_line}_{cur_cond}'

        # Create a key from the combination of marker and cell line
        key = f"{marker}_{cell_line_for_disp}"

        # Check if the key already exists in the dictionary
        if key in sublists_dict:
            # If the key exists, append the file path to the existing sublist
            sublists_dict[key].append(file_path)
        else:
            # If the key does not exist, create a new sublist with the file path
            sublists_dict[key] = [file_path]
    
    return sublists_dict

def multiproc_calc_hists_per_batch_raw(images_paths, batch_df_raw, batch_df_norm, show_cond=False):
    images = images_paths
    n_images  = len(images)
    print("\n\nTotal of", n_images, "images were sampled for hist calculation.")
    sublists_dict = create_sublists_by_marker_cell_line(images, raw=True, show_cond=show_cond)
    with Pool() as mp_pool:    
        
        #batch_df_raw,batch_df_norm = mp_pool.map(_calc_hist, sublists_dict,batch_df_raw,batch_df_norm,batch_num)
        result = mp_pool.starmap(_calc_hist_raw, ([[sublists_dict[cell_line_marker]] for cell_line_marker in sublists_dict]))
        mp_pool.close()
        mp_pool.join()
    
    for res, cell_line_marker in zip(result, sublists_dict):
        cur_marker = cell_line_marker.split("_")[0]
        cur_cell_line = "_".join(cell_line_marker.split("_")[1:])
        batch_df_raw.loc[cur_cell_line, cur_marker] = res[0]
        batch_df_norm.loc[cur_cell_line, cur_marker] = res[1]

    return batch_df_raw, batch_df_norm

def multiproc_calc_hists_per_batch_proc(images_paths, batch_df_proc, show_cond=False):
    images = images_paths
    n_images  = len(images)
    print("\n\nTotal of", n_images, "images were sampled for hist calculation.")
    sublists_dict = create_sublists_by_marker_cell_line(images, raw=False, show_cond=show_cond)
    with Pool() as mp_pool:    
        
        #batch_df_raw,batch_df_norm = mp_pool.map(_calc_hist, sublists_dict,batch_df_raw,batch_df_norm,batch_num)
        result = mp_pool.starmap(_calc_hist_proc, ([[sublists_dict[cell_line_marker]] for cell_line_marker in sublists_dict]))
        mp_pool.close()
        mp_pool.join()
    
    for res, cell_line_marker in zip(result, sublists_dict):
        cur_marker = cell_line_marker.split("_")[0]
        cur_cell_line = "_".join(cell_line_marker.split("_")[1:])
        batch_df_proc.loc[cur_cell_line, cur_marker] = res

    return batch_df_proc


def plot_hist_sep_by_type(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num):
    for hist_df, name in zip([mean_hist_raw, mean_hist_rescale, mean_hist_proc], ['raw', 'rescaled','processed']):
        fig, axs = plt.subplots(figsize=(15, 8), ncols=3, nrows=3, sharey=True, dpi=200)
        fig.subplots_adjust(top=0.85) 
        plt.rcParams.update({'figure.autolayout': True})
        for j, (cell_line, cell_line_df) in enumerate(hist_df.drop(columns=['site_count']).groupby(level=[0])):
            df = cell_line_df.reset_index(level=0, drop=True).T
            x_ticks = [str(round(idx, 1)) for idx in df.index]
            x_ticks[-1] = x_ticks[-1] + "\n - 65000" if name=='raw' else x_ticks[-1] + "-1"

            # Generate positions for the bars using numpy.arange
            bar_positions = np.arange(df.shape[0])
            ax = axs[j//3, j%3]
            # Plot each column separately using a different color for each column
            for col in df.columns:
                ax.bar(bar_positions, df.loc[:,col], alpha=0.7, label=col)
            xticks_size = 8 if name=='rescaled' else 3
            ax.set_xticks(bar_positions,x_ticks,  fontsize=xticks_size)
            ax.set_xlabel('Intestiy value', fontsize=8)
            if j%3==0:
                ax.set_ylabel('Count', fontsize=10)
                ax.tick_params(axis='y', labelsize=8)
            ax.set_title(f'{cell_line}', fontsize=8)
            ax.grid(False)

        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center', ncol=13, bbox_to_anchor=(0.5,0), fontsize='xx-small')
        fig.legend(handles, labels, loc='center right', ncol=1, fontsize=8, bbox_to_anchor=(1.1,0.5))
        # Create a ScalarMappable object for the entire figure
        sm = plt.cm.ScalarMappable(cmap='gray')
        sm.set_array([])  # Dummy array to satisfy the ScalarMappable

        # Add shared colorbar for the entire figure
        cbar_ax = fig.add_axes([1.11, 0.5, 0.02, 0.2])  # Adjust the position as needed
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_ticks([0, 1])  # Assuming the range of values is from 0 to 1
        cbar.set_label('Intestiy Colorbar', fontsize=10)
        cbar.set_ticklabels(['Low', 'High'], fontsize=8)
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)

        plt.suptitle(f'{name} batch{batch_num}')
        plt.tight_layout()
        plt.show()



def plot_hist_sep_by_cell_line(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num):
    mean_hist_raw = (mean_hist_raw/(1024*1024))*100
    mean_hist_rescale = (mean_hist_rescale/(1024*1024))*100
    mean_hist_proc = (mean_hist_proc/(100*100))*100
    for cell_line in np.unique(mean_hist_raw.index.get_level_values(0)):
        cur_raw = mean_hist_raw.drop(columns=['site_count']).loc[cell_line].T
        cur_norm = mean_hist_rescale.drop(columns=['site_count']).loc[cell_line].T
        cur_proc = mean_hist_proc.drop(columns=['site_count']).loc[cell_line].T
        fig, axs = plt.subplots(figsize=(10, 8), ncols=1, nrows=3, dpi=200, sharey=True)
        fig.subplots_adjust(top=0.85) 
        plt.rcParams.update({'figure.autolayout': True})
        for i, (df,name) in enumerate(zip([cur_raw, cur_norm, cur_proc],['raw', 'rescaled','processed'])):
            bar_positions = np.arange(df.shape[0])
            x_ticks = [str(round(idx, 1)) for idx in df.index]
            x_ticks[-1] = x_ticks[-1] + "\n - 65000" if name=='raw' else x_ticks[-1] + "-1"
            xticks_size = 6 if name=='raw' else 8

            for marker in df.columns:
                axs[i].bar(bar_positions, df.loc[:,marker], alpha=0.7, label=marker)
            axs[i].set_xticks(bar_positions,x_ticks,  fontsize=xticks_size)
            axs[i].set_xlabel('')
            axs[i].set_ylabel('')

            axs[i].tick_params(axis='y', labelsize=8)
            axs[i].grid(False)
            axs[i].set_title(name, fontsize=10)
            if i==2:
                axs[i].set_xlabel('Intestiy value', fontsize=18,labelpad=20)
            if i==1:
                axs[i].set_ylabel('%', fontsize=20,labelpad=20)

            
        # Create a ScalarMappable object for the entire figure
        sm = plt.cm.ScalarMappable(cmap='gray')
        sm.set_array([])  # Dummy array to satisfy the ScalarMappable

        # Add shared colorbar for the entire figure
        cbar_ax = fig.add_axes([0.7, 0.05, 0.2, 0.02])  # Adjust the position as needed
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([0, 1])  # Assuming the range of values is from 0 to 1
        cbar.set_ticklabels(['Low', 'High'], fontsize=8)
        cbar.set_label('Intestiy Colorbar', fontsize=10)
        plt.suptitle(f'{cell_line} batch {batch_num}', fontsize=20)
        handles, labels = axs[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', ncol=1, fontsize=8, bbox_to_anchor=(1.1,0.5))
        plt.tight_layout()
        plt.show()
        
        


def plot_hists(batch_df_raw,batch_df_norm, batch_df_proc, batch_num, plot_sep_by_cell_line=False ):
    mean_hist_raw = batch_df_raw.copy()
    mean_hist_raw[batch_df_raw.columns.difference(['site_count'])] = batch_df_raw.drop(columns=['site_count']).div(batch_df_raw['site_count'], axis=0).astype(int)
    
    mean_hist_rescale = batch_df_norm.copy()
    mean_hist_rescale[batch_df_norm.columns.difference(['site_count'])] = batch_df_norm.drop(columns=['site_count']).div(batch_df_norm['site_count'], axis=0).astype(int)

    mean_hist_proc = batch_df_proc.copy()
    mean_hist_proc[batch_df_proc.columns.difference(['site_count'])] = batch_df_proc.drop(columns=['site_count']).div(batch_df_proc['site_count'], axis=0).astype(int)

    plot_hist_sep_by_type(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num)
    if plot_sep_by_cell_line:
        plot_hist_sep_by_cell_line(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num)

def run_calc_hist_new(batch, cell_lines_for_disp, markers, show_cond=False, sample_size_per_markers=200, num_markers=72):    
    INPUT_DIR_BATCH_RAW = os.path.join(INPUT_DIR_RAW, batch.replace('_16bit',''))
    INPUT_DIR_BATCH_PROC = os.path.join(INPUT_DIR_PROC, batch.replace("_sort",""))

    images_raw = sample_images_all_markers_all_lines(INPUT_DIR_BATCH_RAW, sample_size_per_markers, num_markers, raw=True, quiet=True)
    images_proc = sample_images_all_markers_all_lines(INPUT_DIR_BATCH_PROC, _sample_size_per_markers=sample_size_per_markers*2, 
                                                 _num_markers=len(markers), raw=False, quiet=True)
    cell_lines_for_df = [cell_line for cell_line in cell_lines_for_disp.values() for _ in range(len(markers))]
    batch_df_raw = pd.DataFrame(index=[cell_lines_for_df, markers*len(cell_lines_for_disp.values())], 
                                columns=np.concatenate(([0], np.arange(350,1000, 20), [1000])))
    batch_df_raw['site_count'] = np.nan
    batch_df_norm = pd.DataFrame(index=[cell_lines_for_df, markers*len(cell_lines_for_disp.values())], 
                                columns=np.arange(0,1, 0.1))
    batch_df_norm['site_count'] = np.nan
    batch_df_processed = pd.DataFrame(index=[cell_lines_for_df, markers*len(cell_lines_for_disp.values())], 
                                columns=np.arange(0,1, 0.1))
    batch_df_processed['site_count'] = np.nan
    batch_df_raw, batch_df_norm = multiproc_calc_hists_per_batch_raw(images_raw, batch_df_raw, batch_df_norm, show_cond )
    batch_df_processed = multiproc_calc_hists_per_batch_proc(images_proc, batch_df_processed, show_cond)
    #return batch_df_raw, batch_df_norm, batch_df_processed
    plot_hists(batch_df_raw, batch_df_norm, batch_df_processed, batch)


def plot_n_valid_tiles_count(df, custom_palette,reps, batch_min=3, batch_max=9):
    if len(np.unique(df.batch))==1:
        g = sns.catplot(kind='box', data=df, y='cell_line', x='n_valid_tiles',height=12, hue='condition')#, palette=batch_palette,
                    #hue_order=batch_palette.keys(), legend=False)
        g.set_axis_labels('valid tiles count', 'cell line')

        plt.show()
    else:
        df['batch_rep'] = df.batch + " " + df.rep
        # Extract 7 colors from the palette
        colors_list = custom_palette

        batch_palette = {f'batch{i} {rep}':colors_list[i-batch_min] for i in range(batch_min,batch_max+1) for rep in reps}
        g = sns.catplot(kind='box', data=df, y='cell_line_cond', x='n_valid_tiles',height=12, hue='batch_rep', palette=batch_palette,
                        hue_order=batch_palette.keys(), legend=False)
        g.set_axis_labels('valid tiles count', 'cell line')
        rep_hatches = {'rep1': '', 'rep2': '//'}  # Use '' for rep1 (solid) and '//' for rep2 (dots)

        for ax in g.axes.flat:
            for rep in df['rep'].unique():
                if rep == 'rep1':
                    continue
                patches = ax.patches
                patches = [patch for patch in patches if type(patch) != matplotlib.patches.Rectangle]
                for patch in patches[1::len(df['rep'].unique())]:
                    hatch = rep_hatches[rep]
                    patch.set_hatch(hatch)
                    

        legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=batch_palette[key],ec='black', hatch=rep_hatches[key.split()[-1]]) for key in batch_palette]

        # Set the legend with the proxy artists
        g.axes.flat[-1].legend(legend_patches, batch_palette.keys(), title='Batch Rep', loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()


def plot_hm(df, split_by, rows, columns):
    splits = np.unique(df[split_by])
    if len(np.unique(df.batch))==1:
    # Get relevant sub-set of the data
        df_batch_side_a = df[df[split_by] == splits[0]]
        df_batch_side_b = df[df[split_by] == splits[1]]

        fig, axs = plt.subplots(ncols=len(splits), sharey=True, sharex=False)
        a = pd.crosstab(df_batch_side_a[rows], df_batch_side_a[columns], 
                        values=df_batch_side_a['whole_cells_count_in_valid_tiles_mean'], aggfunc=np.mean)
        aa = pd.crosstab(df_batch_side_b[rows], df_batch_side_b[columns], 
                            values=df_batch_side_b['whole_cells_count_in_valid_tiles_mean'], aggfunc=np.mean)
        # Create a heatmap with a separation line between reps
        vmin = 1
        vmax=3
        ax1 = sns.heatmap(a, annot=True, cmap="flare", linewidths=1, linecolor='gray', 
                        cbar=False, ax=axs[0], vmin=vmin, vmax=vmax,annot_kws={"fontsize": 12})
        ax2 = sns.heatmap(aa, annot=True, cmap="flare", linewidths=1, linecolor='gray', 
                        cbar=False, ax=axs[1], vmin=vmin, vmax=vmax, annot_kws={"fontsize": 12})

        plt.suptitle('Perturbations'  + "\n" + 'mean whole cells count in valid tiles', fontsize=20, color="navy")
        ax1.set_xlabel(splits[0], fontsize=24, color="navy")
        ax2.set_xlabel(splits[1], fontsize=24, color="navy")

        ax1.set_ylabel(rows.replace("_", " "), fontsize=24, color="navy")
        ax2.set_ylabel('')
        # Adjust the position of the colorbar
        cbar = ax1.figure.colorbar(ax1.collections[0])
        cbar.ax.tick_params(labelsize=16)
        ax1.axvline(a.shape[1], color='black', linewidth=2)
        ax2.axvline(0, color='black', linewidth=2)
        #ax2.axhline(-0.5, color='black', linewidth=2)
        fig.subplots_adjust(wspace=0)
        fig.show()

    else:
        batchs = np.sort(df['batch'].unique())
        for batch in batchs:
            # Get relevant sub-set of the data
            df_batch_side_a = df[(df['batch'] == batch) & (df[split_by] == splits[0])]
            df_batch_side_b = df[(df['batch'] == batch) & (df[split_by] == splits[1])]

            fig, axs = plt.subplots(figsize=(12, 8), ncols=len(splits), sharey=True, sharex=False)
            a = pd.crosstab(df_batch_side_a[rows], df_batch_side_a[columns], 
                            values=df_batch_side_a['whole_cells_count_in_valid_tiles_mean'], aggfunc=np.mean)
            aa = pd.crosstab(df_batch_side_b[rows], df_batch_side_b[columns], 
                                values=df_batch_side_b['whole_cells_count_in_valid_tiles_mean'], aggfunc=np.mean)
            
            #a = pd.concat([a, a_rep2], keys=['rep1', 'rep2'], axis=1)

            # Create a heatmap with a separation line between reps
            vmin = 1
            vmax=3
            ax1 = sns.heatmap(a, annot=True, cmap="flare", linewidths=1, linecolor='gray', 
                            cbar=False, ax=axs[0], vmin=vmin, vmax=vmax,annot_kws={"fontsize": 12})
            ax2 = sns.heatmap(aa, annot=True, cmap="flare", linewidths=1, linecolor='gray', 
                            cbar=False, ax=axs[1], vmin=vmin, vmax=vmax, annot_kws={"fontsize": 12})

            plt.suptitle(batch  + "\n" + 'mean whole cells count in valid tiles', fontsize=20, color="navy")
            ax1.set_xlabel(splits[0], fontsize=24, color="navy")
            ax2.set_xlabel(splits[1], fontsize=24, color="navy")

            ax1.set_ylabel(rows.replace("_", " "), fontsize=24, color="navy")
            ax2.set_ylabel('')
            # Adjust the position of the colorbar
            cbar = ax1.figure.colorbar(ax1.collections[0])
            cbar.ax.tick_params(labelsize=16)
            ax1.axvline(a.shape[1], color='black', linewidth=2)
            ax2.axvline(0, color='black', linewidth=2)
            #ax2.axhline(-0.5, color='black', linewidth=2)
            fig.subplots_adjust(wspace=0)
            plt.show()