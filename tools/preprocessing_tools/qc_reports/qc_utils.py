import os
import pandas as pd
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import cv2
from src.preprocessing.preprocessing_utils import rescale_intensity
from IPython.display import display, HTML
from tools.preprocessing_tools.image_sampling_utils import sample_images_all_markers_all_lines
from multiprocessing import Pool
import matplotlib
import pathlib
import re
import warnings

def _calc_variance(img_path):

    # load the npy (16 tiles, 2 channels, 100x100)
    x = np.load(img_path)
    # take only protein channel
    x = x[:,:,:,0]

    tiles_var = np.var(x, axis=(1,2)).mean()
    return tiles_var 

def _multiproc_calc_variance(images_paths):
    
    images = images_paths
    n_images  = len(images)
    print("\n\nTotal of", n_images, "images were sampled.")
    
    vars = []
    with Pool() as mp_pool:    
        
        for mean_var in mp_pool.map(_calc_variance, ([img_path for img_path in images])):
            vars.append(mean_var) 
        
        # if wish to run sequentially, and not multiproc
        # for img_path in images:
        #     mean_var = _calc_variance(img_path)
        #     vars.append(mean_var) 
        
        mp_pool.close()
        mp_pool.join()
    
    print(f"Variance: {np.mean(vars)}")
    return np.mean(vars)

def sample_and_calc_variance(INPUT_DIR, batch, sample_size_per_markers=200, num_markers=26,rep_count=2,cond_count=2):
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, batch)

    images = sample_images_all_markers_all_lines(INPUT_DIR_BATCH, sample_size_per_markers, num_markers, all_conds=True,
                                                 rep_count=rep_count, cond_count=cond_count)
    
    variance = _multiproc_calc_variance(images_paths=images)
    
    return variance

def validate_files_proc(path, batch_df, bad_files, marker_info, cell_lines_for_disp, check_antibody=True):
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
            len_rep = len([file for file in all_files_of_marker if f'{rep}_' in file])
            batch_df.loc[(cur_marker, rep), cell_line_for_disp] = len_rep

    else:
        cur_antybodies = ['DAPI','ch1']
        for rep in batch_df.index.get_level_values(1):
            len_rep = len([file for file in all_files_of_marker if f'{rep}_' in file])
            batch_df.loc[(cur_marker, rep), cell_line_for_disp] = len_rep


    for file in all_files_of_marker:
        try:
            size = os.path.getsize(os.path.join(path, file))
            if size < 100000: #size in bytes
                bad_files.append(f'{path}, {file} small size ({size/1000} kB)')
        except:
            bad_files.append(f'{path}, {file} cannot read')
        if check_antibody:
            good_file = False
            if cur_marker!='DAPI':
                for i, antibody in enumerate(cur_antybodies):
                    if f'panel{cur_panels[i]}' in file and antibody in file and cur_cell_line in file:
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

def validate_files_raw(path, batch_df, bad_files, marker_info,cell_lines_for_disp, check_antibody=True):
    path_split = path.split('/')
    cur_marker = path_split[-1]
    cur_cond = path_split[-3]
    cur_cell_line = path_split[-5]
    all_files_of_marker_rep = os.listdir(path)
    cell_line_for_disp = cell_lines_for_disp[f'{cur_cell_line}_{cur_cond}']
    cur_rep = path_split[-2]
    if cur_marker !='DAPI':
        cur_antybodies = marker_info.loc[cur_marker, 'Antibody']
    else:
        cur_antybodies = ['DAPI','ch1']
    if pd.isna(batch_df.loc[cur_marker,cur_rep][cell_line_for_disp]):
        batch_df.loc[(cur_marker, cur_rep), cell_line_for_disp] = len(all_files_of_marker_rep)
    else:
        batch_df.loc[(cur_marker, cur_rep), cell_line_for_disp] += len(all_files_of_marker_rep)
    for file in all_files_of_marker_rep:
        file_ext = os.path.splitext(file)[1]
        if file_ext != '.tiff' and file_ext!='.tif':
            bad_files.append(f'{path}, {file}, ext is {file_ext}')
            continue
        try:
            size = os.path.getsize(os.path.join(path, file))
            if size < 100000: #size in bytes
                bad_files.append(f'{path}, {file} small size ({size/1000} kB)')
                continue
        except:
            bad_files.append(f'{path}, {file} cannot read')
            continue
        if check_antibody:
            good_file = False
            for antibody in cur_antybodies:
                if antibody in file:
                    good_file = True
                    break
            if not good_file:
                bad_files.append(f'{path}, {file}')
    return bad_files, batch_df
                 
def validate_folder_structure(root_dir, folder_structure, missing_paths, bad_files, batch_df,
                               marker_info, cell_lines_for_disp, proc=False, check_antibody=True):
    for name, content in folder_structure.items():
        path = os.path.join(root_dir, name)

        if not os.path.exists(path):
            missing_paths.append(path)
            continue

        if isinstance(content, dict):
            validate_folder_structure(path, content, missing_paths, bad_files, batch_df, marker_info, cell_lines_for_disp, 
                                      proc=proc, check_antibody=check_antibody)
        else: # end of recursion of folders, need to check files
            if proc:
                bad_files, batch_df = validate_files_proc(path, batch_df, bad_files, marker_info, cell_lines_for_disp, check_antibody)
            else:
                bad_files, batch_df = validate_files_raw(path, batch_df, bad_files, marker_info, cell_lines_for_disp, check_antibody)

                
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

def log_files_qc(LOGS_PATH, batches=None, only_wt_cond = True, filename_split='_',site_location=-1):
    files_pds = []

    # Go over all files under logs
    for batch_folder in os.listdir(LOGS_PATH):
        if batches is not None:
            skip_batch = True
            for batch in batches:
                if batch in batch_folder:
                    skip_batch = False
            if skip_batch:
                continue
        print(f"reading logs of {batch_folder}")
        for file in os.listdir(os.path.join(LOGS_PATH, batch_folder)):
            # Take only "cell_count_stats" CSV files
            if file.endswith(".csv") and file.startswith("cell_count_stats"):

                # Load each CSV
                df = pd.read_csv(os.path.join(LOGS_PATH,batch_folder,file), 
                                index_col=None, 
                                header=0, 
                                )

                # Combine them to a single dataframe
                if (not df.empty):
                    files_pds.append(df)

    print("\nTotal of", len(files_pds), "files were read.")
    all_df = pd.concat(files_pds, axis=0).reset_index()
    print("Before dup handeling ", all_df.shape)
    
    #handle duplicates
    all_df['site_num'] = all_df.filename.str.split(filename_split).str[site_location]
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
    df.loc[df.cell_line=='SCNA', 'cell_line'] = 'SNCA'
    if only_wt_cond:
        df['cell_line_cond'] = df['cell_line']
        df.loc[df.cell_line=='WT', 'cell_line_cond'] = df.cell_line + " " + df.condition
    else:
        df['cell_line_cond'] = df.cell_line + " " + df.condition

    df['site_cell_count_sum'] = df['cells_counts'].apply(get_array_sum)
    df['site_whole_cells_counts_sum'] = df['whole_cells_counts'].apply(get_array_sum)
    df['cells_counts_list']=df['cells_counts'].apply(convert_to_list)
    
    print(f'\nPAY ATTENTION!!!! df.site_num: {df.site_num[:1].values[0]}, can be defined using filename_split & site_location')
    return df.sort_values(by=['batch'])

def create_folder_structure(folder_type, markers,cell_lines_to_cond, reps, panels, cell_lines_to_reps = None):
    folder_structure = {}
    
    for cell_line in cell_lines_to_cond.keys():
        if cell_lines_to_reps is not None:
            reps = cell_lines_to_reps[cell_line]
        if folder_type == 'processed':
            folder_structure[cell_line] = {cond : {marker: {marker} for marker in markers} 
                                                    for cond in cell_lines_to_cond[cell_line]}

        elif folder_type == 'raw':
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
    
def apply_color(value, expected_count=100):
    # Check the conditions and return the corresponding color
    if (value == expected_count) or  (5000 < value):
        return color_light_green
    elif (0.8*expected_count < value < expected_count) or (3000 < value <=5000):
        return color_yellow
    elif (0.2*expected_count < value <= 0.8*expected_count) or (1000 < value <=3000):
        return 'orange'
    elif (0 < value <= 0.2*expected_count) or (100 < value <=1000):
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
                           show_sum=False, fmt=".0f"):
    for batch, batch_data in filtered.groupby('batch'):
        p = batch_data.pivot_table(index=['rep', extra_index],
                                    columns='cell_line_cond',
                                    values='index')
        p = p.sort_values(by=[extra_index,'rep'])

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        annot=True
        if second is not None:
            annot = p.apply(lambda col: col.map(lambda x: f"{x:.0f}%"))

        hm = sns.heatmap(data=p, ax=ax,
                            yticklabels=p.index, cmap='RdYlGn',annot=annot,
                            vmin=vmin, vmax=vmax, cbar=True,
                            annot_kws={'fontsize': 5, 'ha':'right','color':'black'},
                            fmt=fmt,
                            cbar_kws = {'shrink': 0.2,})
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.xaxis.tick_top()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(batch)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', fontsize=8)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(axis='y', labelsize=6)

        if second is not None:
            ax2 = ax.twinx()  # Create a twin Axes sharing the xaxis
            second_data = second[second.batch==batch]
            second_p = second_data.pivot_table(index=['rep', extra_index],
                                    columns='cell_line_cond',
                                    values='index')
            second_p = second_p.sort_values(by=[extra_index,'rep'])
            sns.heatmap(second_p, annot=False,
                         cbar=False, ax=ax2, alpha=0)
            for y, (rep, value) in enumerate(second_p.iterrows()):
                for x, val in enumerate(value):
                    if pd.isna(val):
                        continue
                    if val != p.iloc[y,x]:
                        ax2.annotate(f' ({val:.0f})', xy=(x+0.5, y+0.46),fontsize=5, c='black', va='center')

            # Customize the y-axis of the second heatmap
            ax2.set_yticks([])  # Hide the y-axis ticks
            ax2.set_ylabel('')  # Hide the y-axis label
        plt.show()
        
        if show_sum:
            p['Total'] = p.sum(axis=1)
            p.loc['Total'] = p.sum(axis=0)
            fig, axs = plt.subplots(ncols=2, figsize= (10,6), dpi=150)
            marker_total = p[['Total']].drop(index='Total')
            marker_total.index = [f"{idx[0]}, {idx[1]}" for idx in marker_total.index]

            cell_line_total = pd.DataFrame(p.loc['Total']).drop(index='Total')
            sns.barplot(data=marker_total.reset_index(), y='index', x='Total', ax=axs[0], palette='husl',
                        hue='index',legend=False)
            axs[0].set_xlim(marker_total.min().min()-0.5*marker_total.min().min(), marker_total.max().max()+0.1* marker_total.max().max())
            axs[0].tick_params(axis='x', labelsize=10)
            axs[0].set_ylabel(batch)
            sns.barplot(data=cell_line_total, y=cell_line_total.index, x='Total', ax=axs[1], palette='tab10',
                        hue=cell_line_total.index, legend=False)
            axs[1].set_xlim(cell_line_total.min().min()-0.5*cell_line_total.min().min(), cell_line_total.max().max()+0.1*cell_line_total.max().max())
            axs[1].tick_params(axis='x', labelsize=10)
            axs[1].set_ylabel('')

            for ax in axs:
                ax.set_xlabel(xlabel)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                plt.tight_layout()
            plt.show()

def add_empty_lines(df, batches, line_colors, panels, reps, to_ignore=None, markers=None):
    has_marker = 'marker' in df.columns
    use_markers = markers if markers is not None else (df['marker'].unique() if has_marker else [None])

    for batch in batches:
        for cell_line_cond in line_colors.keys():
            for panel in panels.columns:
                for rep in reps:
                    for marker in use_markers:
                        filters = (
                            (df.batch == batch) &
                            (df.cell_line_cond == cell_line_cond) &
                            (df.panel == f'panel{panel}') &
                            (df.rep == rep)
                        )
                        if marker is not None:
                            filters &= (df['marker'] == marker)
                        matches = df[filters]
                        if matches.shape[0] == 0:
                            # Check if this exact combination should be ignored
                            should_ignore = False
                            if to_ignore:
                                should_ignore = all([
                                    (k == 'batch' and batch in to_ignore[k]) or
                                    (k == 'cell_line_cond' and cell_line_cond in to_ignore[k]) or
                                    (k == 'panel' and f'panel{panel}' in to_ignore[k]) or
                                    (k == 'rep' and rep in to_ignore[k]) or
                                    (k == 'marker' and marker is not None and marker in to_ignore[k])
                                    for k in to_ignore
                                ])
                            value = np.nan if should_ignore else 0

                            new_row = {
                                'batch': batch,
                                'cell_line_cond': cell_line_cond,
                                'panel': f'panel{panel}',
                                'rep': rep,
                                'index': value
                            }
                            if marker is not None:
                                new_row['marker'] = marker
                            # new_row_df = pd.DataFrame([new_row])
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def plot_filtering_table(filtered, extra_index, width=8, height=8):
    p = filtered.pivot_table(index=['batch', 'rep', extra_index],
                            columns='cell_line_cond',
                            values='index')
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
    table = ax.table(cellText=p.apply(lambda x: x.map(str)).values,
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

def plot_table(df, file_name, plot_path, reps, expected_dapi, fig_height=8, fig_width=8, to_save=False, expected_count=100):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colored_df_without_DAPI = df.drop('DAPI', level=0).apply(lambda x: x.map(lambda v: apply_color(v, expected_count)) if x.name else x)
    dapi_index_data = [['DAPI']*len(reps),reps]
    
    dapi = df.loc['DAPI']
    dapi  = dapi.set_index(pd.MultiIndex.from_arrays(dapi_index_data))
    colored_df_DAPI = dapi.apply(lambda col: col.map(lambda x: apply_color_dapi(x, expected_dapi)))

    colored_df = pd.concat([colored_df_without_DAPI, colored_df_DAPI])
    colored_df['Rep'] = 'white'
    colored_df = colored_df[ ['Rep']+ [col for col in colored_df.columns if col != 'Rep']]
    df_reset = df.reset_index(level=1)
    df_dapi = df_reset.loc['DAPI']
    df_reset = df_reset.drop(index='DAPI')
    df_reset = pd.concat([df_reset, df_dapi])
    col_labels = [col.replace("_", "\n") for col in df_reset.columns]
    table = ax.table(cellText=df_reset.apply(lambda x: x.map(str)).values,
             rowLabels=df_reset.index,
             colLabels=col_labels,
             cellLoc='center',
             rowLoc='center',
             loc='center',
             cellColours=colored_df.values,
             bbox=[0, 0, 3, 3],
             colWidths=[0.05]+ [0.1] * (len(df_reset.columns)-1))
    plt.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    if to_save:
        fig.set_size_inches(5, 5)  # Example: width=10 inches, height=6 inches
        pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(plot_path, f'{file_name}.png'))

    fig.set_size_inches(fig_width, fig_height)
    plt.show()

def plot_table_diff(df, plot_path, file_name,fig_height=8, fig_width=8, to_save=False):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colored_df = df.apply(lambda x: x.map(apply_color_diff))

    colored_df = colored_df.reset_index(level=1)
    colored_df['Rep'] = 'white'
    df_reset = df.reset_index(level=1)
    col_labels = [col.replace("_", "\n") for col in df_reset.columns]

    table = ax.table(cellText=df_reset.apply(lambda x: x.map(str)).values,
             rowLabels=df_reset.index,
             colLabels=col_labels,
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
                                    batches=[f'batch{i}' for i in range(3,10)],
                                    cell_lines_to_reps = None, expected_count=100, check_antibody=True):
    folder_type = 'processed' if proc else 'raw'
    folder_structure = create_folder_structure(folder_type, markers,cell_lines_to_cond, reps, panels, cell_lines_to_reps)
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
                                                                       cell_lines_for_disp, proc=proc,check_antibody=check_antibody)
        if len(missing_paths) == 0:
            print("Folder structure is valid.")
        else:
            print(f"Folder structure is invalid. Missing {len(missing_paths)} paths:")
            for path in missing_paths:
                print(path)
        if len(bad_files) == 0:
            print('No bad files are found.')
        else:
            print(f'{len(bad_files)} files are bad:')
            for file in bad_files[:3]:
                print(file)

        title = f'{folder_type}_table_{batch}'
        print('Total Sites: ',batch_df.sum().sum())
        plot_table(batch_df, title, plot_path, reps, expected_dapi_raw, fig_height,fig_width, expected_count=expected_count)
        print('=' * 8)
        batch_dfs.append(batch_df)
    print('=' * 20)
    return batch_dfs
    
    
def plot_cell_count(df, order, custom_palette, y, title, norm=False, figsize=(15,6)):
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
            fig, axs = plt.subplots(nrows=1, ncols=no_batches, sharey=False, sharex=False,figsize=figsize)
            fig.subplots_adjust(wspace=0)
            max_y_value = (max(df.groupby(['batch','rep','cell_line_cond'])[y].std()+df.groupby(['batch','rep','cell_line_cond'])[y].mean()))
            min_y_value = (min(-df.groupby(['batch','rep','cell_line_cond'])[y].std()+df.groupby(['batch','rep','cell_line_cond'])[y].mean()))

            for i, (batch_name, batch) in enumerate(df.groupby('batch')):
                batch = batch.sort_values(by='rep')
                c = sns.barplot(data=batch, x='rep', hue='cell_line_cond', y=y, hue_order = order, 
                                ax=axs[i], palette=custom_palette, errorbar='sd', err_kws={'linewidth': 1})
                c.set_xlabel(batch_name, fontsize=12) 
                c.tick_params(axis='x', labelsize=10)
                axs[i].set_ylim(min_y_value*1.3, max_y_value*1.1)
                if 0<i<no_batches-1: #middle plots
                    c.legend_.remove()
                    c.set_ylabel('')
                    axs[i].set_yticks([])
                    axs[i].set_yticklabels([])
                if i==no_batches-1:
                    c.set_ylabel('')
                    c.legend(title='Cell Line', loc='upper left', bbox_to_anchor=(1, 0.8), fontsize=14)
                    axs[i].set_yticks([])
                    axs[i].set_yticklabels([])
                if i==0:
                    c.legend_.remove()
                    c.set_ylabel(ylabel)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=no_batches, sharey=True, sharex=False, figsize=figsize)
            for i, (batch_name, batch) in enumerate(df.groupby('batch')):
                batch = batch.sort_values(by='rep')
                c = sns.barplot(data=batch, x='rep', hue='cell_line_cond', y=y, hue_order = order, 
                                ax=ax, palette=custom_palette, errorbar='sd')
                c.set_xlabel(batch_name, fontsize=12) 
                c.tick_params(axis='x', labelsize=10)
                c.legend(title='Cell Line', loc='upper left', bbox_to_anchor=(1, 0.8), fontsize=14)
                c.set_ylabel(ylabel)
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
        if f'{cur_cell_line}_{cur_cond}' not in cell_lines_for_disp:
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
    sublists_dict = create_sublists_by_marker_cell_line(images, raw=True, n=n, cell_lines_for_disp=cell_lines_for_disp)
    with Pool() as mp_pool:    
        
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
    sublists_dict = create_sublists_by_marker_cell_line(images, raw=False, n=n, cell_lines_for_disp=cell_lines_for_disp)
    with Pool() as mp_pool:    
        
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
            ax.set_xlabel('Intensity value', fontsize=8)
            if j%ncols==0:
                ax.set_ylabel('Count', fontsize=10)
                ax.tick_params(axis='y', labelsize=8)
            ax.set_title(f'{cell_line}', fontsize=8)
            ax.grid(False)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', ncol=1, fontsize=8, bbox_to_anchor=(1.1,0.5))
        # Create a ScalarMappable object for the entire figure
        sm = plt.cm.ScalarMappable(cmap='gray')
        sm.set_array([])  # Dummy array to satisfy the ScalarMappable

        # Add shared colorbar for the entire figure
        cbar_ax = fig.add_axes([1.11, 0.5, 0.02, 0.2])  # Adjust the position as needed
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_ticks([0, 1])  # Assuming the range of values is from 0 to 1
        cbar.set_label('Intensity Colorbar', fontsize=10)
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
                axs[i].set_xlabel('Intensity value', fontsize=18,labelpad=20)
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
        cbar.set_label('Intensity Colorbar', fontsize=10)
        plt.suptitle(f'{cell_line} batch {batch_num}', fontsize=20)
        handles, labels = axs[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', ncol=1, fontsize=8, bbox_to_anchor=(1.1,0.5))
        plt.tight_layout()
        plt.show()
                
def plot_hists(batch_df_raw,batch_df_norm, batch_df_proc, batch_num, plot_sep_by_cell_line=False, ncols=3, nrows=3, figsize=(15, 8)):
    mean_hist_raw = batch_df_raw.copy()
    mean_hist_raw[batch_df_raw.columns.difference(['site_count'])] = batch_df_raw.drop(columns=['site_count']).div(batch_df_raw['site_count'], axis=0).astype(int)
    
    mean_hist_rescale = batch_df_norm.copy()
    mean_hist_rescale[batch_df_norm.columns.difference(['site_count'])] = batch_df_norm.drop(columns=['site_count']).div(batch_df_norm['site_count'], axis=0).astype(int)

    mean_hist_proc = batch_df_proc.copy()
    mean_hist_proc[batch_df_proc.columns.difference(['site_count'])] = batch_df_proc.drop(columns=['site_count']).div(batch_df_proc['site_count'], axis=0).astype(int)
    plot_hist_lines(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num, ncols, nrows, figsize=figsize)
    if plot_sep_by_cell_line:
        plot_hist_sep_by_cell_line(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num)


def plot_hist_lines(mean_hist_raw, mean_hist_rescale, mean_hist_proc, batch_num, ncols=7, nrows=4, figsize=(15, 8)):
    for hist_df, name in zip([mean_hist_raw, mean_hist_rescale, mean_hist_proc], ['raw', 'rescaled','processed']):
        fig, axs = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows, sharey=True, dpi=200)
        if nrows==1:
            axs = axs.reshape(1,-1)
        if ncols==1:
            axs = axs.reshape(-1,1)
        fig.subplots_adjust(top=0.85) 
        plt.rcParams.update({'figure.autolayout': True})
        for j, (marker, marker_df) in enumerate(hist_df.drop(columns=['site_count']).groupby(level=1)):
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
            ax.set_xlabel('Intensity value', fontsize=8)
            if j%ncols==0:
                ax.set_ylabel('Count', fontsize=10)
                ax.tick_params(axis='y', labelsize=8)
            ax.set_title(f'{marker}', fontsize=8)
            ax.grid(False)
        handles, labels = ax.get_legend_handles_labels()
        if (j+1)//ncols < nrows or (j+1)%ncols <nrows:
            for new_j in range(j+1, ncols*nrows):
                ax = axs[new_j//ncols, new_j%ncols]
                ax.axis('off')
                ax.set_xticks([])
                ax.tick_params(axis='y', which='both', left=False, right=False)  # Hide y-ticks
                ax.set_ylabel('') 
        fig.legend(handles, labels, loc='center right', ncol=1, fontsize=8, bbox_to_anchor=(1.4,0.5))
        # Create a ScalarMappable object for the entire figure
        sm = plt.cm.ScalarMappable(cmap='gray')
        sm.set_array([])  # Dummy array to satisfy the ScalarMappable

        plt.suptitle(f'{name} {batch_num}')
        fig.tight_layout()
        plt.show()

        
def run_calc_hist_new(batch, cell_lines_for_disp, markers, input_dir_raw, input_dir_proc, hist_sample=1, 
                      sample_size_per_markers=200, ncols=3, nrows=3, rep_count=2, cond_count=2, dnls=False,
                      figsize=(15, 8)):    
    input_dir_batch_raw = os.path.join(input_dir_raw, batch.replace('_16bit','').replace('_no_downsample',''))
    input_dir_batch_proc = os.path.join(input_dir_proc, batch.replace("_sort",""))

    images_raw = sample_images_all_markers_all_lines(input_dir_batch_raw, sample_size_per_markers, _num_markers=len(markers), markers_to_include=markers,
                                                     raw=True, all_conds=False, rep_count=rep_count, cond_count=cond_count, exclude_DAPI=True)
    images_proc = sample_images_all_markers_all_lines(input_dir_batch_proc, _sample_size_per_markers=sample_size_per_markers,#*2, 
                                                 _num_markers=len(markers), raw=False, all_conds=True, markers_to_include=markers)

    if dnls:
        raw_markers = markers.copy()
        raw_markers.remove('TDP43N')
        raw_markers = [marker if marker not in ['TDP43B'] else 'TDP43' for marker in raw_markers] 
    else:
        raw_markers = markers.copy()
    cell_lines_for_df_raw = [cell_line for cell_line in cell_lines_for_disp.values() for _ in range(len(raw_markers))]
    cell_lines_for_df = [cell_line for cell_line in cell_lines_for_disp.values() for _ in range(len(markers))]
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
    plot_hists(batch_df_raw.dropna(), batch_df_norm.dropna(), batch_df_processed.dropna(), batch, ncols=ncols, nrows=nrows, figsize=figsize)

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

def plot_catplot(df, custom_palette, reps, x, x_title, y='cell_line_cond', y_title='cell line',hue='batch_rep', 
                 batch_min=3, batch_max=9, height = 12, aspect=1, batches=None):
    if np.unique(df.batch)[0]=='Perturbations':
        g = sns.catplot(kind='box', data=df, y='cell_line', x=x,height=12, hue='condition')
        g.set_axis_labels(x_title, 'cell line')

        plt.show()
    else:
        df.loc[:, 'batch_rep'] = df['batch'] + " " + df['rep']
        colors_list = custom_palette
        if hue == 'batch_rep':
            if batches is None:
                palette = {f'batch{i} {rep}':colors_list[i-batch_min] for i in range(batch_min,batch_max+1) for rep in reps}
            else:
                batches = [int(batch.replace('batch','')) for batch in batches]
                palette = {f'batch{i} {rep}':colors_list[i-1] for i in batches for rep in reps}
            hue_order=palette.keys()
        else:
            palette=custom_palette
            hue_order = None
        df=df.sort_values(by=hue)
        g = sns.catplot(kind='box', data=df, y=y, x=x,height=height, aspect=aspect, hue=hue, palette=palette,
                        hue_order=hue_order)
        g.set_axis_labels(x_title, y_title)

        if hue == 'batch_rep' and len(reps)<3: # not really working for more than 2 reps currently!!!
            g._legend.remove()
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
            legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=palette[key],ec='black', hatch=rep_hatches[key.split()[-1]]) for key in palette]

            # Set the legend with the proxy artists
            g.axes.flat[-1].legend(legend_patches, palette.keys(), title='Batch Rep', loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()


def plot_hm_of_mean_cell_count_per_tile(df, split_by, rows, columns, value='cells_count_in_valid_tiles_mean', figsize=(12, 8), vmin=1, vmax=4):
    
    if len(np.unique(df.batch))==1:
        if split_by is not None:
            splits = np.unique(df[split_by])
            # Get relevant sub-set of the data
            df_batch_side_a = df[df[split_by] == splits[0]]
            df_batch_side_b = df[df[split_by] == splits[1]]

            fig, axs = plt.subplots(ncols=len(splits), sharey=False, sharex=False, figsize=(12,8))
            a = pd.crosstab(df_batch_side_a[rows], df_batch_side_a[columns],
                            values=df_batch_side_a[value], aggfunc="mean")
            aa = pd.crosstab(df_batch_side_b[rows], df_batch_side_b[columns], 
                                values=df_batch_side_b[value], aggfunc="mean")
            # Create a heatmap with a separation line between reps
            ax1 = sns.heatmap(a, annot=True, cmap="flare", linewidths=1, linecolor='gray', 
                            cbar=False, ax=axs[0], vmin=vmin, vmax=vmax,annot_kws={"fontsize": 12})
            ax2 = sns.heatmap(aa, annot=True, cmap="flare", linewidths=1, linecolor='gray', 
                            cbar=False, ax=axs[1], vmin=vmin, vmax=vmax, annot_kws={"fontsize": 12})

            plt.suptitle(value.replace('_',' '), fontsize=20, color="navy")
            ax1.set_xlabel(splits[0], fontsize=24, color="navy")
            ax2.set_xlabel(splits[1], fontsize=24, color="navy")

            ax1.set_ylabel(rows.replace("_", " "), fontsize=24, color="navy")
            ax2.set_ylabel('')
            # Adjust the position of the colorbar
            cbar = ax1.figure.colorbar(ax1.collections[0])
            cbar.ax.tick_params(labelsize=16)
            ax1.axvline(a.shape[1], color='black', linewidth=2)
            ax2.axvline(0, color='black', linewidth=2)
            fig.subplots_adjust(wspace=0)
            fig.show()
        else:
            fig, ax = plt.subplots(ncols=1, sharey=False, sharex=False, figsize=(12,8))
            a = pd.crosstab(df[rows], df[columns],
                            values=df[value], aggfunc="mean")
            # Create a heatmap with a separation line between reps
            ax1 = sns.heatmap(a, annot=True, cmap="flare", linewidths=1, linecolor='gray', 
                            cbar=False, ax=ax, vmin=vmin, vmax=vmax,annot_kws={"fontsize": 12})

            plt.suptitle(value.replace('_',' '), fontsize=20, color="navy")

            ax1.set_ylabel(rows.replace("_", " "), fontsize=24, color="navy")
            # Adjust the position of the colorbar
            cbar = ax1.figure.colorbar(ax1.collections[0])
            cbar.ax.tick_params(labelsize=16)
            ax1.axvline(a.shape[1], color='black', linewidth=2)
            fig.subplots_adjust(wspace=0)
            fig.show()

    else:
        splits = np.unique(df[split_by])
        batchs = np.sort(df['batch'].unique())
        for batch in batchs:
            fig, axs = plt.subplots(figsize=figsize, ncols=len(splits), sharey=False, sharex=False)
            if len(splits) == 1:
                axs = [axs]  # Ensure axs is iterable
            for i, split_val in enumerate(splits):
                df_batch = df[(df['batch'] == batch) & (df[split_by] == split_val)]
                pivot_table = pd.crosstab(df_batch[rows], df_batch[columns],
                                        values=df_batch[value], aggfunc="mean")
                ax = axs[i]
                sns.heatmap(pivot_table, annot=True, cmap="flare", linewidths=1, linecolor='gray',
                            cbar=False, ax=ax, vmin=vmin, vmax=vmax, annot_kws={"fontsize": 12})

                ax.set_xlabel(str(split_val), color="navy")
                if i == 0:
                    ax.set_ylabel(rows.replace("_", " "), color="navy")
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                else:
                    ax.set_ylabel('')
                    ax.set_yticks([])

                # Add vertical line for visual separation
                ax.axvline(pivot_table.shape[1], color='black', linewidth=2)
    
            plt.suptitle(batch  + "\n" + value.replace('_',' '), color="navy")
            fig.subplots_adjust(wspace=0)
            fig.tight_layout()
            plt.show()

def plot_hm_combine_batches(df,  batches, reps, rows, columns, vmin=1, vmax=4):
    ncols = len(batches)*len(reps)
    fig, axs = plt.subplots(figsize=(6*ncols, 7), ncols=ncols, sharey=False, sharex=False,
                            gridspec_kw={'width_ratios': [0.8]*(ncols-1) + [1]})
    for i, (batch, rep) in enumerate([(x, y) for x in batches for y in reps]):
        cur_df = df[(df['batch'] == batch) & (df['rep'] == rep)]
        a = pd.crosstab(cur_df[rows], cur_df[columns], 
                        values=cur_df['cells_count_in_valid_tiles_mean'], aggfunc="mean")
        ytick_labels = [f'cell line {i}' for i in range(1, a.shape[0] + 1)]
        heatmap = sns.heatmap(a, annot=True, cmap="flare", linewidths=1, linecolor='gray', 
                        cbar=False, ax=axs[i], vmin=vmin, vmax=vmax,annot_kws={"fontsize": 12}, yticklabels=ytick_labels)

        heatmap.set_xlabel(f'{batch}\n{rep}', fontsize=24, color="navy")
        heatmap.axvline(a.shape[1], color='black', linewidth=2)
        heatmap.set_ylabel('')
        if i==0:
            heatmap.set_ylabel(rows.replace("_", " "), fontsize=24, color="navy")
            heatmap.set_yticklabels(ytick_labels, rotation=0)

        else:
            heatmap.axvline(0, color='black', linewidth=2)
            axs[i].set_yticklabels('')
            axs[i].set_yticks([])

    fig.subplots_adjust(wspace=0)
    plt.suptitle('Mean of cells count in valid tiles', fontsize=20, color="navy")
    plt.show()

def show_site_survival_dapi_brenner(df_dapi, batches, line_colors, panels, reps, figsize=(5,5), vmax=100,to_ignore=None):
    dapi_filter_by_brenner = df_dapi.groupby(['batch','cell_line_cond','panel','rep']).index.count().reset_index()
    dapi_filter_by_brenner=add_empty_lines(dapi_filter_by_brenner, batches, line_colors, panels, reps, to_ignore)
    dapi_filter_by_brenner.sort_values(by=['batch','cell_line_cond','panel','rep'], inplace=True)
    dapi_filter_by_brenner.reset_index(inplace=True, drop=True)
    plot_filtering_heatmap(dapi_filter_by_brenner, extra_index='panel',
                           xlabel='% site survival Brenner on DAPI', figsize=figsize, 
                           vmax=vmax)
    return dapi_filter_by_brenner

def show_site_survival_dapi_cellpose(df_dapi, batches, dapi_filter_by_brenner, line_colors, panels, reps, figsize=(5,5), to_ignore=None):
    dapi_filter_by_cellpose = df_dapi[df_dapi.site_cell_count!=0]
    dapi_filter_by_cellpose = dapi_filter_by_cellpose.groupby(['batch','cell_line_cond','panel','rep']).index.count().reset_index()
    dapi_filter_by_cellpose=add_empty_lines(dapi_filter_by_cellpose, batches, line_colors, panels, reps,to_ignore=to_ignore)
    dapi_filter_by_cellpose.sort_values(by=['batch','cell_line_cond','panel','rep'], inplace=True)
    dapi_filter_by_cellpose.reset_index(inplace=True, drop=True)
    assert(dapi_filter_by_cellpose.drop(columns='index') == dapi_filter_by_brenner.drop(columns='index')).all().all()
    dapi_filter_by_cellpose_per = dapi_filter_by_cellpose.copy()
    dapi_filter_by_cellpose_per['index'] = round(dapi_filter_by_cellpose_per['index']*100 / np.maximum(dapi_filter_by_brenner['index'],1))
    plot_filtering_heatmap(dapi_filter_by_cellpose_per, extra_index='panel', xlabel='% Site survival Cellpose', 
                           second=dapi_filter_by_cellpose, figsize=figsize, fmt="")
    return dapi_filter_by_cellpose

def show_site_survival_dapi_tiling(df_dapi, batches, dapi_filter_by_cellpose, line_colors, panels, reps, figsize=(5,5),to_ignore=None):
    dapi_filter_by_tiling = df_dapi[(df_dapi.site_cell_count!=0) & (df_dapi.n_valid_tiles!=0)]
    dapi_filter_by_tiling = dapi_filter_by_tiling.groupby(['batch','cell_line_cond','panel','rep']).index.count().reset_index()
    dapi_filter_by_tiling=add_empty_lines(dapi_filter_by_tiling, batches, line_colors, panels, reps, to_ignore=to_ignore)
    dapi_filter_by_tiling.sort_values(by=['batch','cell_line_cond','panel','rep'], inplace=True)
    dapi_filter_by_tiling.reset_index(inplace=True, drop=True)
    assert(dapi_filter_by_tiling.drop(columns='index') == dapi_filter_by_cellpose.drop(columns='index')).all().all()
    dapi_filter_by_tiling_per = dapi_filter_by_tiling.copy()
    dapi_filter_by_tiling_per['index'] = round(dapi_filter_by_tiling_per['index']*100 / np.maximum(dapi_filter_by_cellpose['index'],1))
    plot_filtering_heatmap(dapi_filter_by_tiling_per, extra_index='panel', xlabel='% Site survival tiling', 
                       second=dapi_filter_by_tiling, figsize=figsize, fmt="")
    return dapi_filter_by_tiling

def show_site_survival_target_brenner(df_dapi, df_target, dapi_filter_by_tiling, markers, figsize=(6,8) ):
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
                        figsize=figsize, fmt="")
    return

def calc_total_sums(df_target, df_dapi, stats, markers):
    dfs = []
    for marker in markers:
        if marker=='DAPI':
            merge = df_dapi[['batch','cell_line_cond','rep','site_num','panel'] + stats]
        else:
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

def show_total_valid_tiles_per_marker_and_batch(total_sum, vmin=None, vmax=None):
    total_per_batch = total_sum.groupby(['marker','batch']).n_valid_tiles.sum().reset_index()
    total_per_batch = total_per_batch.pivot(index='marker', columns='batch', values='n_valid_tiles')
    total_per_batch = total_per_batch.drop(index='DAPI')
    fig = plt.figure(figsize=(6,8),dpi=100)
    hm = sns.heatmap(total_per_batch, annot=True, fmt="", cmap='coolwarm_r', vmin=vmin, vmax=vmax)
    hm.set_yticks([i + 0.5 for i in range(len(total_per_batch.index))])
    hm.set_yticklabels(total_per_batch.index)
    hm.set_title('Total Valid Tiles')
    plt.show()