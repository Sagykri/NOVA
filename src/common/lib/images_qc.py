import os
import pandas as pd
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
import re
import warnings


MOMAPS_HOME = '/home/labs/hornsteinlab/Collaboration/MOmaps/'
BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR_RAW = os.path.join(BASE_DIR,'input','images','raw','SpinningDisk')

INPUT_DIR_PROC = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')

def sample_and_calc_variance(INPUT_DIR, batch, sample_size_per_markers=200, num_markers=26,rep_count=2,cond_count=2):
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, batch)

    images = sample_images_all_markers_all_lines(INPUT_DIR_BATCH, sample_size_per_markers, num_markers, all_conds=True,
                                                 rep_count=rep_count, cond_count=cond_count)
    
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
    if pd.isna(array_string):
        return None
    # Remove square brackets and split the string into individual elements
    elements = array_string[1:-1].split()
    # Convert elements to integers and create the NumPy array
    array = np.array([int(elem) for elem in elements])
    return np.sum(array)

def convert_to_list(array_string):
    if pd.isna(array_string):
        return None
    string_list =  re.findall(r'\d+', array_string)
    int_list = [int(x) for x in string_list]
    return int_list

def log_files_qc(LOGS_PATH):
    files_pds = []

    # Go over all files under logs
    for batch_folder in os.listdir(LOGS_PATH):
        print(f"reading logs of {batch_folder}")
        for file in os.listdir(os.path.join(LOGS_PATH, batch_folder)):
            # Take only "cell_count_stats" CSV files
            if file.endswith(".csv") and file.startswith("cell_count_stats"):

                # Load each CSV
                df = pd.read_csv(os.path.join(LOGS_PATH,batch_folder,file), 
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

    if 'dox' in np.unique(df.condition) or 'HPBCD' in np.unique(df.condition) or 'LPS' in np.unique(df.condition):
        df['cell_line_cond'] = df.cell_line + " " + df.condition

    df['site_cell_count_sum'] = df['cells_counts'].apply(get_array_sum)
    df['site_whole_cells_counts_sum'] = df['whole_cells_counts'].apply(get_array_sum)
    df['cells_counts_list']=df['cells_counts'].apply(convert_to_list)

    #df['p_valid_tiles'] = df['n_valid_tiles']*100 / df['cells_counts_list'].apply(len) not relevant since total tiles=100!

    
    # for name, group in df.groupby(['filename','batch','cell_line','panel','condition','rep']):

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
    
def apply_color(value):
    # Check the conditions and return the corresponding color
    if (value == 100) or  (5000 < value):
        return color_light_green
    elif (80 < value < 100) or (3000 < value <=5000):
        return color_yellow
    elif (20 < value <= 80) or (1000 < value <=3000):
        return 'orange'
    elif (0 < value <= 20) or (100 < value <=1000):
        return 'red'
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

def custom_fmt(value):
    # Custom function to format the annotation text with a "/"
    return f'/{value:.0f}'

def plot_filtering_heatmap(filtered, extra_index, xlabel='', figsize=(5,5), second=None, vmin=0, vmax=100, 
                           show_sum=False):
    for batch, batch_data in filtered.groupby('batch'):
        p = batch_data.pivot_table(index=['rep', extra_index],
                                    columns='cell_line_cond',
                                    values='index')
        p = p.sort_values(by=extra_index)

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        hm = sns.heatmap(data=p, ax=ax,
                            yticklabels=p.index, cmap='RdYlGn',annot=True,
                            vmin=vmin, vmax=vmax, cbar=True,annot_kws={'fontsize': 5, 'ha':'right'},fmt=".0f",
                            cbar_kws = {'shrink': 0.2,})
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.xaxis.tick_top()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(batch)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', fontsize=8)
        # cbar_ax = fig.add_axes([0, 0.9, 0.2, 0.02])  # Adjust the position as needed
        # cbar = plt.colorbar(ax, cax=cbar_ax, orientation='vertical')
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(axis='y', labelsize=6)

        if second is not None:
            ax2 = ax.twinx()  # Create a twin Axes sharing the xaxis
            second_data = second[second.batch==batch]
            second_p = second_data.pivot_table(index=['rep', extra_index],
                                    columns='cell_line_cond',
                                    values='index')
            second_p = second_p.sort_values(by=extra_index)
            sns.heatmap(second_p, annot=False,
                         cbar=False, ax=ax2, alpha=0)
            for y, (rep, value) in enumerate(second_p.iterrows()):
                for x, val in enumerate(value):
                    if pd.isna(val):
                        continue
                    if val != p.iloc[y,x]:
                        ax2.annotate(f' ({val:.0f})', xy=(x+0.5, y+0.46),fontsize=5, c='white', va='center')

            # Customize the y-axis of the second heatmap
            ax2.set_yticks([])  # Hide the y-axis ticks
            ax2.set_ylabel('')  # Hide the y-axis label
        plt.show()
        
        if show_sum:
            p['Total'] = p.sum(axis=1)
            p.loc['Total'] = p.sum(axis=0)
            fig, axs = plt.subplots(ncols=2, figsize= (10,6), dpi=150)
            marker_total = p[['Total']].drop(index='Total')
            cell_line_total = pd.DataFrame(p.loc['Total']).drop(index='Total')
            sns.barplot(data=marker_total, y=marker_total.index, x='Total', ax=axs[0])
            axs[0].set_xlim(marker_total.min().min()-0.5*marker_total.min().min(), marker_total.max().max()+0.1* marker_total.max().max())
            axs[0].xaxis.set_major_locator(MultipleLocator(1000))
            axs[0].tick_params(axis='x', labelsize=10)
            axs[0].set_ylabel(batch)
            sns.barplot(data=cell_line_total, y=cell_line_total.index, x='Total', ax=axs[1])
            axs[1].set_xlim(cell_line_total.min().min()-0.5*cell_line_total.min().min(), cell_line_total.max().max()+0.1*cell_line_total.max().max())
            axs[1].xaxis.set_major_locator(MultipleLocator(10000))
            axs[1].tick_params(axis='x', labelsize=10)
            axs[1].set_ylabel('')

            for ax in axs:
                ax.set_xlabel(xlabel)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                plt.tight_layout()
            plt.show()

def add_empty_lines(df, batches, line_colors, panels):
    for batch in batches:
        for cell_line_cond in line_colors.keys():
            for panel in panels.columns:
                for rep in reps:
                    if df[(df.batch==batch)&
                        (df.cell_line_cond==cell_line_cond)&
                        (df.panel==f'panel{panel}')&
                        (df.rep==rep)].shape[0] ==0:
                            new_row = {'batch': batch, 'cell_line_cond': cell_line_cond,
                                    'panel': f'panel{panel}', 'rep':rep, 'index':0}
                            # Add the new row to the DataFrame
                            df = df.append(new_row, ignore_index=True)
    return df

def plot_filtering_table(filtered, extra_index, width=8, height=8):
    p = filtered.pivot_table(index=['batch', 'rep', extra_index],
                            columns='cell_line_cond',
                            values='index')
    # p=p.astype('Int64')
    p=p.applymap(lambda x: int(x) if not pd.isna(x) else x)

    p=p.sort_values(by=['batch',extra_index])
    color_p = p.applymap(apply_color)

    p = p.reset_index()
    p = p.set_index('batch')

    color_p = color_p.reset_index()
    color_p = color_p.set_index('batch')
    color_p['rep'] = 'white'
    color_p[extra_index] = 'white'
    fig, ax = plt.subplots(figsize=(width, height))
    table = ax.table(cellText=p.applymap(str).values,
            rowLabels=p.index,
            colLabels=p.columns,
            cellLoc='center',
            rowLoc='center',
            loc='center',
            cellColours=color_p.values,
            bbox=[0, 0, 3, 3],
            colWidths=[0.05]+ [0.05]+[0.1] * (len(p.columns)-1))
    plt.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    plt.show()  

def plot_table(df, file_name, plot_path, reps, expected_dapi, fig_height=8, fig_width=8, to_save=False):
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
    if to_save:
        fig.set_size_inches(5, 5)  # Example: width=10 inches, height=6 inches
        pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(plot_path, f'{file_name}.png'))

    # Displaying the plot with a smaller figure size in the notebook
    #plt.figure(figsize=(8, fig_height))  # Example: width=8 inches, height=4 inches
    fig.set_size_inches(fig_width, fig_height)
    plt.show()

def plot_table_diff(df, plot_path, file_name,fig_height=8, fig_width=8, to_save=False):
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
    if to_save:
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
    if not proc and 'deltaNLS' in root_dir:
        markers.remove('TDP43N')
        markers = [marker if marker not in ['TDP43B'] else 'TDP43' for marker in markers]         
    index_data = [[marker for marker in markers for _ in reps],reps * len(markers)]
    multi_index = pd.MultiIndex.from_arrays(index_data, names=['Marker', 'Rep'])
    for batch in batches: 
        print(batch)
        # Specify the root directory to validate
        batch_root_dir = os.path.join(root_dir, batch)
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
            print('No bad files are found.')
        else:
            print('Some files are bad:')
            for file in bad_files:
                print(file)

        title = f'{folder_type}_table_{batch}'
        print('Total Sites: ',batch_df.sum().sum())
        plot_table(batch_df, title, plot_path, reps, expected_dapi_raw, fig_height,fig_width)
        print('=' * 8)
        batch_dfs.append(batch_df)
    print('=' * 20)
    return batch_dfs
    
    
def plot_cell_count(df, order, custom_palette, y, title, norm=False):
    if np.unique(df.batch)[0]=="Perturbations":
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
        if no_batches>1:
            fig, axs = plt.subplots(nrows=1, ncols=no_batches, sharey=True, sharex=False, figsize=(15,6))
            fig.subplots_adjust(wspace=0)
            for i, (batch_name, batch) in enumerate(df.groupby('batch')):
                c = sns.barplot(data=batch, x='rep', hue='cell_line_cond', y=y, hue_order = order, 
                                ax=axs[i], palette=custom_palette, errorbar='sd')
                c.set_xlabel(batch_name, fontsize=12) 
                c.tick_params(axis='x', labelsize=10)
                if 0<i<no_batches-1: #middle plots
                    c.spines['left'].set_visible(False)
                    c.spines['right'].set_visible(False)
                    c.legend_.remove()
                    c.set_ylabel('')
                    # c.set_yticks([])
                    # c.set_yticklabels([])
                if i==no_batches-1:
                    c.spines['left'].set_visible(False)
                    c.set_ylabel('')
                    c.legend(title='Cell Line', loc='upper left', bbox_to_anchor=(1, 0.8), fontsize=14)

                if i==0:
                    c.spines['right'].set_visible(False)
                    c.legend_.remove()
                    c.set_ylabel(ylabel)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=no_batches, sharey=True, sharex=False, figsize=(12,6))
            for i, (batch_name, batch) in enumerate(df.groupby('batch')):
                c = sns.barplot(data=batch, x='rep', hue='cell_line_cond', y=y, hue_order = order, 
                                ax=ax, palette=custom_palette, errorbar='sd')
                c.set_xlabel(batch_name, fontsize=12) 
                c.tick_params(axis='x', labelsize=10)
                c.legend(title='Cell Line', loc='upper left', bbox_to_anchor=(1, 0.8), fontsize=14)
                c.set_ylabel(ylabel)
    # title = 'Cell Count Average per Site with STD'
    plt.suptitle(title)
    plt.show()

def plot_sites_count(df, expected, order, custom_palette, split_to_reps=False):
    title = 'Number of Raw images in each batch and cell line'
    if not split_to_reps:
        plt.figure(figsize=(12, 8))
        if np.unique(df.batch)[0]=='Perturbations':
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

def create_sublists_by_marker_cell_line(images, raw, n, cell_lines_for_disp):
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
        if 'SCNA' in cur_cell_line:
            continue
        cell_line_for_disp = cell_lines_for_disp[f'{cur_cell_line}_{cur_cond}']
        # Create a key from the combination of marker and cell line
        key = f"{marker}_{cell_line_for_disp}"

        # Check if the key already exists in the dictionary
        if key in sublists_dict:
            # If the key exists, append the file path to the existing sublist
            sublists_dict[key].append(file_path)
        else:
            # If the key does not exist, create a new sublist with the file path
            sublists_dict[key] = [file_path]
    

    def sub_sample_dict(original_dict, n):
        sub_sampled_dict = {}
        
        for key, value_list in original_dict.items():
            if n >= len(value_list):
                sub_sampled_dict[key] = value_list.copy()  # Include all elements
            else:
                sub_sampled_dict[key] = random.sample(value_list, n)
        return sub_sampled_dict
    
    sublists_dict = sub_sample_dict(sublists_dict, n)
    return sublists_dict

def multiproc_calc_hists_per_batch_raw(images_paths, batch_df_raw, batch_df_norm, n, cell_lines_for_disp):
    images = images_paths
    n_images  = len(images)
    #print("\n\nTotal of", n_images, "images were sampled for hist calculation.")
    sublists_dict = create_sublists_by_marker_cell_line(images, raw=True, n=n, cell_lines_for_disp=cell_lines_for_disp)
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

def multiproc_calc_hists_per_batch_proc(images_paths, batch_df_proc, n, cell_lines_for_disp):
    images = images_paths
    n_images  = len(images)
    #print("\n\nTotal of", n_images, "images were sampled for hist calculation.")
    sublists_dict = create_sublists_by_marker_cell_line(images, raw=False, n=n, cell_lines_for_disp=cell_lines_for_disp)
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

def plot_hist_sep_by_type(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num, ncols=3, nrows=3):
    for hist_df, name in zip([mean_hist_raw, mean_hist_rescale, mean_hist_proc], ['raw', 'rescaled','processed']):
        fig, axs = plt.subplots(figsize=(15, 8), ncols=ncols, nrows=nrows, sharey=True, dpi=200)
        fig.subplots_adjust(top=0.85) 
        plt.rcParams.update({'figure.autolayout': True})
        for j, (cell_line, cell_line_df) in enumerate(hist_df.drop(columns=['site_count']).groupby(level=[0])):
            df = cell_line_df.reset_index(level=0, drop=True).T
            x_ticks = [str(round(idx, 1)) for idx in df.index]
            x_ticks[-1] = x_ticks[-1] + "\n - 65000" if name=='raw' else x_ticks[-1] + "-1"

            # Generate positions for the bars using numpy.arange
            bar_positions = np.arange(df.shape[0])
            ax = axs[j//ncols, j%ncols]
            # Plot each column separately using a different color for each column
            for col in df.columns:
                ax.bar(bar_positions, df.loc[:,col], alpha=0.7, label=col)
            xticks_size = 8 if name=='rescaled' else 3
            ax.set_xticks(bar_positions,x_ticks,  fontsize=xticks_size)
            ax.set_xlabel('Intestiy value', fontsize=8)
            if j%ncols==0:
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

        plt.suptitle(f'{name} {batch_num}')
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
                
def plot_hists(batch_df_raw,batch_df_norm, batch_df_proc, batch_num, plot_sep_by_cell_line=False, ncols=3, nrows=3):
    mean_hist_raw = batch_df_raw.copy()
    mean_hist_raw[batch_df_raw.columns.difference(['site_count'])] = batch_df_raw.drop(columns=['site_count']).div(batch_df_raw['site_count'], axis=0).astype(int)
    
    mean_hist_rescale = batch_df_norm.copy()
    mean_hist_rescale[batch_df_norm.columns.difference(['site_count'])] = batch_df_norm.drop(columns=['site_count']).div(batch_df_norm['site_count'], axis=0).astype(int)

    mean_hist_proc = batch_df_proc.copy()
    mean_hist_proc[batch_df_proc.columns.difference(['site_count'])] = batch_df_proc.drop(columns=['site_count']).div(batch_df_proc['site_count'], axis=0).astype(int)
    # plot_hist_sep_by_type(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num, ncols, nrows)
    plot_hist_lines(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num, ncols, nrows)
    if plot_sep_by_cell_line:
        plot_hist_sep_by_cell_line(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num)


def plot_hist_lines(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num, ncols=7, nrows=4):
    for hist_df, name in zip([mean_hist_raw, mean_hist_rescale, mean_hist_proc], ['raw', 'rescaled','processed']):
        fig, axs = plt.subplots(figsize=(15, 8), ncols=ncols, nrows=nrows, sharey=True, dpi=200)
        fig.subplots_adjust(top=0.85) 
        plt.rcParams.update({'figure.autolayout': True})
        for j, (marker, marker_df) in enumerate(hist_df.drop(columns=['site_count']).groupby(level=[1])):
            df = marker_df.reset_index(level=1, drop=True).T
            x_ticks = [str(round(idx, 1)) for idx in df.index]
            x_ticks[-1] = x_ticks[-1] + "\n - 65000" if name=='raw' else x_ticks[-1] + "-1"

            # Generate positions for the bars using numpy.arange
            bar_positions = list(range(0, df.shape[0]))
            ax = axs[j//ncols, j%ncols]
            # Plot each column separately using a different color for each column
            for col in df.columns:
                ax.plot(bar_positions, df[col], label=col, linewidth=0.4)
            xticks_size = 4
            if name=='raw':
                bar_positions = bar_positions[0:-1:2] + [bar_positions[-1]]
                x_ticks = x_ticks[0:-1:2] + [x_ticks[-1]]
                xticks_size=2.5
            ax.set_xticks(bar_positions, x_ticks,  fontsize=xticks_size)
            ax.set_xlabel('Intestiy value', fontsize=8)
            if j%ncols==0:
                ax.set_ylabel('Count', fontsize=10)
                ax.tick_params(axis='y', labelsize=8)
            ax.set_title(f'{marker}', fontsize=8)
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

        plt.suptitle(f'{name} {batch_num}')
        plt.tight_layout()
        plt.show()

        
def run_calc_hist_new(batch, cell_lines_for_disp, markers, hist_sample=1, 
                      sample_size_per_markers=200, ncols=3, nrows=3, rep_count=2, cond_count=2, dnls=False):    
    INPUT_DIR_BATCH_RAW = os.path.join(INPUT_DIR_RAW, batch.replace('_16bit','').replace('_no_downsample',''))
    INPUT_DIR_BATCH_PROC = os.path.join(INPUT_DIR_PROC, batch.replace("_sort",""))

    images_raw = sample_images_all_markers_all_lines(INPUT_DIR_BATCH_RAW, sample_size_per_markers, _num_markers=len(markers),
                                                     raw=True, all_conds=False, rep_count=rep_count, cond_count=cond_count, exclude_DAPI=True)
    images_proc = sample_images_all_markers_all_lines(INPUT_DIR_BATCH_PROC, _sample_size_per_markers=sample_size_per_markers,#*2, 
                                                 _num_markers=len(markers), raw=False, all_conds=True)

    if dnls:
        raw_markers = markers.copy()
        raw_markers.remove('TDP43N')
        raw_markers = [marker if marker not in ['TDP43B'] else 'TDP43' for marker in raw_markers] 
    else:
        raw_markers = markers.copy()
    cell_lines_for_df_raw = [cell_line for cell_line in cell_lines_for_disp.values() for _ in range(len(raw_markers))]
    cell_lines_for_df = [cell_line for cell_line in cell_lines_for_disp.values() for _ in range(len(markers))]
    # print(raw_markers)
    batch_df_raw = pd.DataFrame(index=[cell_lines_for_df_raw, raw_markers*len(cell_lines_for_disp.values())], 
                                columns=np.concatenate(([0], np.arange(350,1000, 20), [1000])))
    batch_df_raw['site_count'] = np.nan
    batch_df_norm = pd.DataFrame(index=[cell_lines_for_df_raw, raw_markers*len(cell_lines_for_disp.values())], 
                                columns=np.arange(0,1, 0.1))
    batch_df_norm['site_count'] = np.nan
    batch_df_processed = pd.DataFrame(index=[cell_lines_for_df, markers*len(cell_lines_for_disp.values())], 
                                columns=np.arange(0,1, 0.1))
    batch_df_processed['site_count'] = np.nan
    batch_df_raw, batch_df_norm = multiproc_calc_hists_per_batch_raw(images_raw, batch_df_raw, batch_df_norm, hist_sample, cell_lines_for_disp )
    batch_df_processed =  multiproc_calc_hists_per_batch_proc(images_proc, batch_df_processed, hist_sample,cell_lines_for_disp)
    #return batch_df_raw, batch_df_norm, batch_df_processed
    plot_hists(batch_df_raw.dropna(), batch_df_norm.dropna(), batch_df_processed.dropna(), batch, ncols=ncols, nrows=nrows)

    #plot_hists(batch_df_processed, batch_df_processed, batch_df_processed, batch, ncols=ncols, nrows=nrows)


def plot_barplot(df, custom_palette, reps, title, y, x, batch_min=3, batch_max=9):
    df['batch_rep'] = df.batch + " " + df.rep
    colors_list = custom_palette
    batch_palette = {f'batch{i} {rep}':colors_list[i-batch_min] for i in range(batch_min,batch_max+1) for rep in reps}
    g = sns.barplot(df, y=y, x=x,hue='batch_rep', orient='h',palette=batch_palette,
                            hue_order=batch_palette.keys())
    g.set_ylabel('cell line')
    rep_hatches = {'rep1': '', 'rep2': '//'}  # Use '' for rep1 (solid) and '//' for rep2 (dots)

    for rep in df['rep'].unique():
        if rep == 'rep1':
            continue
        patches = g.patches
        for patch in patches[1::len(df['rep'].unique())]:
            hatch = rep_hatches[rep]
            patch.set_hatch(hatch)

    for patch in g.patches:
        patch.set_edgecolor('black')
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=batch_palette[key],ec='black', hatch=rep_hatches[key.split()[-1]]) for key in batch_palette]

    # Set the legend with the proxy artists
    g.legend(legend_patches, batch_palette.keys(), title='Batch Rep', loc='center left', bbox_to_anchor=(1, 0.5))
    g.set_title(title)
    plt.show()

def plot_count_plot(df, custom_palette, reps, title, batch_min=3, batch_max=9):
    df['batch_rep'] = df.batch + " " + df.rep
    colors_list = custom_palette
    batch_palette = {f'batch{i} {rep}':colors_list[i-batch_min] for i in range(batch_min,batch_max+1) for rep in reps}
    g = sns.countplot(df, y='cell_line_cond', hue='batch_rep',palette=batch_palette,
                            hue_order=batch_palette.keys())
    g.set_ylabel('cell line')
    rep_hatches = {'rep1': '', 'rep2': '//'}  # Use '' for rep1 (solid) and '//' for rep2 (dots)

    for rep in df['rep'].unique():
        if rep == 'rep1':
            continue
        patches = g.patches
        for patch in patches[1::len(df['rep'].unique())]:
            hatch = rep_hatches[rep]
            patch.set_hatch(hatch)

    for patch in g.patches:
        patch.set_edgecolor('black')
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=batch_palette[key],ec='black', hatch=rep_hatches[key.split()[-1]]) for key in batch_palette]

    # Set the legend with the proxy artists
    g.legend(legend_patches, batch_palette.keys(), title='Batch Rep', loc='center left', bbox_to_anchor=(1, 0.5))
    g.set_title(title)
    plt.show()

def plot_catplot(df, custom_palette, reps, x, x_title, batch_min=3, batch_max=9):
    if np.unique(df.batch)[0]=='Perturbations':
        g = sns.catplot(kind='box', data=df, y='cell_line', x=x,height=12, hue='condition')#, palette=batch_palette,
                    #hue_order=batch_palette.keys(), legend=False)
        g.set_axis_labels(x_title, 'cell line')

        plt.show()
    else:
        df['batch_rep'] = df.batch + " " + df.rep
        # Extract 7 colors from the palette
        colors_list = custom_palette

        batch_palette = {f'batch{i} {rep}':colors_list[i-batch_min] for i in range(batch_min,batch_max+1) for rep in reps}
        g = sns.catplot(kind='box', data=df, y='cell_line_cond', x=x,height=12, hue='batch_rep', palette=batch_palette,
                        hue_order=batch_palette.keys(), legend=False)
        g.set_axis_labels(x_title, 'cell line')
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

        fig, axs = plt.subplots(ncols=len(splits), sharey=True, sharex=False, figsize=(12,8))
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

def plot_hm_combine_batches(df,  batches, reps, rows, columns):
    fig, axs = plt.subplots(figsize=(24, 7), ncols=4, sharey=False, sharex=False,
                            gridspec_kw={'width_ratios': [0.8, 0.8, 0.8, 1]})
    for i, (batch, rep) in enumerate([(x, y) for x in batches for y in reps]):
        cur_df = df[(df['batch'] == batch) & (df['rep'] == rep)]
        a = pd.crosstab(cur_df[rows], cur_df[columns], 
                        values=cur_df['whole_cells_count_in_valid_tiles_mean'], aggfunc=np.mean)
        vmin = 1
        vmax = 2
        ytick_labels = [f'cell line {i}' for i in range(1, a.shape[0] + 1)]
        heatmap = sns.heatmap(a, annot=True, cmap="flare", linewidths=1, linecolor='gray', 
                        cbar=False, ax=axs[i], vmin=vmin, vmax=vmax,annot_kws={"fontsize": 12}, yticklabels=ytick_labels)

        heatmap.set_xlabel(f'{batch}\n{rep}', fontsize=24, color="navy")
        heatmap.axvline(a.shape[1], color='black', linewidth=2)
        heatmap.set_ylabel('')
        if i==0:
            heatmap.set_ylabel(rows.replace("_", " "), fontsize=24, color="navy")
            # axs[i].set_yticklabels(ytick_labels, fontsize=12, rotate=90)
            heatmap.set_yticklabels(ytick_labels, rotation=0)
            cbar = heatmap.figure.colorbar(heatmap.collections[0])
            cbar.ax.tick_params(labelsize=16)
            # axs[i].text(0.9, -0.3, batch,transform=axs[i].transAxes, fontsize=30)
        else:
            heatmap.axvline(0, color='black', linewidth=2)
            axs[i].set_yticklabels('')
            axs[i].set_yticks([])
        # if i==2:
        #     axs[i].text(0.8, -0.3, batch,transform=axs[i].transAxes, fontsize=30)

    fig.subplots_adjust(wspace=0)
    plt.suptitle('Mean of whole cells count in valid tiles', fontsize=20, color="navy")
    plt.show()

def show_site_survival_dapi_brenner(df_dapi, batches, line_colors, panels):
    dapi_filter_by_brenner = df_dapi.groupby(['batch','cell_line_cond','panel','rep']).index.count().reset_index()
    dapi_filter_by_brenner=add_empty_lines(dapi_filter_by_brenner, batches, line_colors, panels)
    dapi_filter_by_brenner.sort_values(by=['batch','cell_line_cond','panel','rep'], inplace=True)
    dapi_filter_by_brenner.reset_index(inplace=True, drop=True)
    plot_filtering_heatmap(dapi_filter_by_brenner, extra_index='panel',xlabel='% site survival Brenner on DAPI')
    return dapi_filter_by_brenner

def show_site_survival_dapi_cellpose(df_dapi, batches, dapi_filter_by_brenner, line_colors, panels):
    dapi_filter_by_cellpose = df_dapi[df_dapi.site_cell_count!=0]
    dapi_filter_by_cellpose = dapi_filter_by_cellpose.groupby(['batch','cell_line_cond','panel','rep']).index.count().reset_index()
    dapi_filter_by_cellpose=add_empty_lines(dapi_filter_by_cellpose, batches, line_colors, panels)
    dapi_filter_by_cellpose.sort_values(by=['batch','cell_line_cond','panel','rep'], inplace=True)
    dapi_filter_by_cellpose.reset_index(inplace=True, drop=True)
    assert(dapi_filter_by_cellpose.drop(columns='index') == dapi_filter_by_brenner.drop(columns='index')).all().all()
    dapi_filter_by_cellpose_per = dapi_filter_by_cellpose.copy()
    dapi_filter_by_cellpose_per['index'] = round(dapi_filter_by_cellpose_per['index']*100 / dapi_filter_by_brenner['index'])
    dapi_filter_by_cellpose_per.fillna(0, inplace=True)
    plot_filtering_heatmap(dapi_filter_by_cellpose_per, extra_index='panel', xlabel='% Site survival Cellpose', second=dapi_filter_by_cellpose)
    return dapi_filter_by_cellpose

def show_site_survival_dapi_tiling(df_dapi, batches, dapi_filter_by_cellpose, line_colors, panels):
    dapi_filter_by_tiling = df_dapi[(df_dapi.site_cell_count!=0) & (df_dapi.n_valid_tiles!=0)]
    dapi_filter_by_tiling = dapi_filter_by_tiling.groupby(['batch','cell_line_cond','panel','rep']).index.count().reset_index()
    dapi_filter_by_tiling=add_empty_lines(dapi_filter_by_tiling, batches, line_colors, panels)
    dapi_filter_by_tiling.sort_values(by=['batch','cell_line_cond','panel','rep'], inplace=True)
    dapi_filter_by_tiling.reset_index(inplace=True, drop=True)
    assert(dapi_filter_by_tiling.drop(columns='index') == dapi_filter_by_cellpose.drop(columns='index')).all().all()
    dapi_filter_by_tiling_per = dapi_filter_by_tiling.copy()
    dapi_filter_by_tiling_per['index'] = round(dapi_filter_by_tiling_per['index']*100 / dapi_filter_by_cellpose['index'])
    dapi_filter_by_tiling_per.fillna(0, inplace=True)
    plot_filtering_heatmap(dapi_filter_by_tiling_per, extra_index='panel', xlabel='% Site survival tiling', 
                       second=dapi_filter_by_tiling)
    return dapi_filter_by_tiling

def show_site_survival_target_brenner(df_dapi, df_target, dapi_filter_by_tiling):
    pass_dapi = df_dapi[(df_dapi.site_cell_count!=0) & (df_dapi.n_valid_tiles!=0)] # take only DAPI's that passed so far (Brenner & Cellpose & tiling)
    passs = pd.concat([pass_dapi,df_target])
    pass_target = pd.DataFrame(columns=['batch','rep','marker','panel']) # create empty df for results

    for marker in markers:
        if marker=='DAPI':
            continue
        # for each marker, find the DAPI sites that passed
        pass_target_cur = passs[passs.marker.str.contains(f'{marker}|DAPI', regex=True)] 
        # groupby all identifiers to group DAPI&marker sites, then count rows, later count only rows with count>1 (to ignore DAPI)
        site_pass = pass_target_cur.groupby(['site_num','batch','cell_line_cond','rep','panel']).index.count().reset_index() # for each site, count how many passes (includeing dapi)
        marker_pass = site_pass[site_pass['index']>1].groupby(['batch','cell_line_cond','rep','panel'])['index'].count().reset_index() # find how many targets passed and then add them all
        marker_pass['marker'] = marker # add marker info
        pass_target = pass_target.merge(marker_pass, how='outer') # save result

    pass_target_per = pass_target.copy() # calc percentages
    merge = pass_target.merge(dapi_filter_by_tiling[['batch','cell_line_cond','rep','index','panel']],
                    on=['batch', 'cell_line_cond', 'panel', 'rep'], suffixes=('_pass', '_dapi'))
    pass_target_per = pass_target_per.sort_values(by=['batch','cell_line_cond','rep','panel','marker']).reset_index()
    merge = merge.sort_values(by=['batch','cell_line_cond','rep','panel','marker']).reset_index()

    pass_target_per['index'] = round(merge['index_pass']*100 / merge['index_dapi'])
    plot_filtering_heatmap(pass_target_per.drop(columns=['level_0','panel']), extra_index='marker', 
                        xlabel = '% Site survival by Brenner on target channel', second=pass_target,
                        figsize=(6,8))
    return

def calc_total_sums(df_target, df_dapi, stats):
    dfs = []
    for marker in markers:
        if marker=='DAPI':
            continue
        cur_target = df_target[df_target.marker==marker]
        to_merge_target = cur_target[['batch','cell_line_cond','rep','site_num','panel']]
        to_merge_dapi = df_dapi[['batch','cell_line_cond','rep','site_num','panel'] + stats] 
        merge = to_merge_target.merge(to_merge_dapi,
                                    on= ['batch','cell_line_cond','rep','site_num','panel'], how='left')
        
        cur_sum = merge.groupby(['batch','cell_line_cond','rep','panel']).sum().reset_index()
        cur_sum['marker'] = marker
        dfs.append(cur_sum)
    total_sum = pd.concat(dfs)
    return total_sum

def show_total_sum_tables(total_sum):
    # show table for each batch
    for batch, batch_totals in total_sum.groupby('batch'):
        describe = pd.DataFrame()
        describe['n_valid_tiles'] = batch_totals[['n_valid_tiles']].describe()
        describe['% valid tiles'] = (batch_totals[['n_valid_tiles']]*100/10000).describe() # calc percentage of tiles out of posssible tiles (100 sites*100 tiles in a site)
        describe.loc['sum','n_valid_tiles'] = batch_totals['n_valid_tiles'].sum()
        describe['site_whole_cells_counts_sum'] = batch_totals[['site_whole_cells_counts_sum']].describe()
        describe.loc['sum','site_whole_cells_counts_sum'] = batch_totals['site_whole_cells_counts_sum'].sum()
        describe['site_cell_count'] = batch_totals[['site_cell_count']].describe()
        describe.loc['sum','site_cell_count'] = batch_totals['site_cell_count'].sum()
        describe.loc['expected_count'] = int(9 * 2 * 25)
        describe.index.name = batch
        display(HTML(describe.to_html()))
    # show table for all batches combined 
    describe = pd.DataFrame()
    describe['n valid tiles'] = total_sum[['n_valid_tiles']].describe() 
    describe['% valid tiles'] = (total_sum[['n_valid_tiles']]*100/10000).describe() # calc percentage of tiles out of posssible tiles (100 sites*100 tiles in a site)
    describe.loc['sum','n valid tiles'] = total_sum['n_valid_tiles'].sum()
    describe['site_whole_cells_counts_sum'] = total_sum[['site_whole_cells_counts_sum']].describe()
    describe.loc['sum','site_whole_cells_counts_sum'] = total_sum['site_whole_cells_counts_sum'].sum()
    describe['site_cell_count'] = total_sum[['site_cell_count']].describe()
    describe.loc['sum','site_cell_count'] = total_sum['site_cell_count'].sum()
    describe.loc['expected_count'] = int(9 * 2 * 25)
    describe.index.name = 'All batches'
    display(HTML(describe.to_html()))
    return