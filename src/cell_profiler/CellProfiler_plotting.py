## Plot UMAPs from CellProfiler output

#Packages
from datetime import datetime
import pandas as pd 
import numpy as np
import logging
import os
import seaborn as sns
import sys

import matplotlib.backends.backend_pdf as bpdf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import umap     #installation of umap-learn is required
import umap.plot

from random import shuffle
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score
import sklearn.cluster as cluster

# Global paths
BATCH_TO_RUN = 'batch3' 

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
sys.path.insert(1, BASE_DIR)

INPUT_DIR = os.path.join(BASE_DIR, 'outputs','cell_profiler', 'deltaNLS_sort')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN, 'combined')
OUTPUT_DIR = os.path.join(INPUT_DIR, BATCH_TO_RUN, 'plots')

LOG_DIR_PATH = os.path.join(BASE_DIR, 'outputs','cell_profiler', 'logs')
    
    
def set_logging(log_file_path, level=logging.INFO, format=' INFO: %(message)s'):
    formatter = '%(asctime)s %(levelname)-8s %(message)s'
    handlers = [logging.FileHandler(log_file_path + '.log'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=formatter, handlers=handlers, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(__doc__)


def load_data_and_plot_UMAPs(input_path, stress = True):
     with os.scandir(input_path) as input_data_folder:
        # We loop on all markers, plot single marker UMAPs (AKA UMAP0), 
        # but also keep the marker data for later UMAP1 (of all markers in one plot)
        all_markers_df = []
        
        for features_file in input_data_folder:
            
            file = os.path.basename(features_file.path)
            if file == 'all_markers_all.csv':
                continue
            
            if (file == f'all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'):
                continue
            
            if file == 'FUS_FUS-lines-only_preprocessed.csv':
                continue
            
            if (file == f'stress_all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'):
                continue
            
            if file == f'UMAP2_dataframe_{BATCH_TO_RUN}.csv':
                continue
            
            if file == f'UMAP2_stress-dataframe_{BATCH_TO_RUN}.csv':
                continue
            
            # To support UMAP2
            marker = file.replace("_all.csv","")
            
            if (marker == 'all_markers'):
                continue
            
            # Load CP features
            logging.info(f'Reading csv: {features_file}')
            df = pd.read_csv(features_file)
            df['marker'] = marker
            
            # Create an empty dataframe
            new_df = []

            groups = df.groupby(['replicate', 'treatment', 'cell_line'])
            for name, group in groups:
                
                logging.info(f"\n\nGroup name:{name} {group.shape}")
                
                group_df = group[group['object_type']!='PrimaryObject3']
                mask1, mask2, mask3 = group_df['object_type']=='PrimaryObject1', group_df['object_type']=='PrimaryObject2', group_df['object_type']=='SecondaryObject'
                A, B, C = group_df[mask1], group_df[mask2], group_df[mask3]
                
                # Add suffix to column names
                A.columns = [col+'_PrimaryObject1' if col not in ['object_type', 'replicate', 'treatment', 'cell_line', 'Parent_Nucleus', 'marker'] else col for col in A.columns]
                B.columns = [col+'_PrimaryObject2' if col not in ['object_type', 'replicate', 'treatment', 'cell_line', 'Parent_Nucleus', 'marker'] else col for col in B.columns]
                C.columns = [col+'_SecondaryObject' if col not in ['object_type', 'replicate', 'treatment', 'cell_line', 'Parent_Nucleus', 'marker'] else col for col in C.columns]
                
                # Combine all object types into one row per condition per image
                for (_, a), (_, b), (_, c) in zip(A.iterrows(), B.iterrows(), C.iterrows()):
                    single_row = pd.concat([a, b, c], axis=0).to_frame().T
                    new_df.append(pd.DataFrame(single_row))
                    
            # concat list of dataframe into a single dataframe
            df = pd.concat(new_df)
            
            # Remove duplicated columns (cell_line, rep, treatment)
            df = df.T.drop_duplicates().T
            logging.info(f"\n\nFinished concatinating features: {df.shape}")
           
            df = preprocessing(df)
            
            #plot_umap0(df, marker)
            
            all_markers_df.append(df)
            
            #print(f"After preprocessing: {df.shape}")
            """
            from 358 to 352 columns while only 4 should be dropped?
            """
        
        #Combine markers
        df_all = pd.concat(all_markers_df)

        if stress:
            df_all.to_csv(os.path.join(INPUT_DIR_BATCH, f'stress_all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'))
        else:
            df_all.to_csv(os.path.join(INPUT_DIR_BATCH, f'all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'))
        
        # Plot UMAP1
        #plot_umap1(df_all)
        
        return None
                  
            
def preprocessing(features_df, use_condition='deltaNLS', exclude_cols=['Parent_Nucleus', 'object_type', 'Unnamed: 0_PrimaryObject1', 'Unnamed: 0_PrimaryObject2', 'Unnamed: 0_SecondaryObject']):
    """
    Prepare the data for clustering
    - Included desired condition
    - Remove columns (in load_data_and_plot_UMAPs, all columns converted to data type 'float', so manually write columns to remove)
    - Handle misssing values
    """
    df = features_df.copy()
    # Check if the columns you wish to exclude, really exist in the df
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    
    # Drop SCNA line because too many missing values
    #df = df[df['cell_line'].isin(['WT', 'TDP43', 'TBK1', 'OPTN', 'FUSRevertant', 'FUSHomozygous', 'FUSHeterozygous'])]
    
    # Microglia test:
    #df = df[df['cell_line'].isin(['WT', 'TDP43', 'TBK1', 'FUSHomozygous'])]

    # Take the desired condition and corresponding columns
    if use_condition == 'stress':
        df = df.loc[df['cell_line'] == 'WT'] #shortcut now because only WT has stress / no stress
    elif use_condition == 'LPS':
        df = df.loc[df['treatment'] == use_condition]
        df.drop('treatment', axis=1, inplace=True)
    elif use_condition == 'Untreated':
        df = df.loc[df['treatment'] == use_condition]
        df.drop('treatment', axis=1, inplace=True)
    elif use_condition == 'deltaNLS':
        df = df.loc[df['cell_line'] == 'TDP43']
    
    df.drop(exclude_cols, axis=1, inplace=True)
        
    # Data was converted to type 'object', so convert back to float to support UMAP
    cols_to_convert = [col for col in df.columns if col not in ['replicate', 'treatment', 'cell_line', 'marker']]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric)
    
    # Four features have misssing values, replace them with -1 (Neighbors_AngleBetweenNeighbors_Adjacent,Neighbors_SecondClosestDistance_Adjacent, Neighbors_FirstClosestDistance_Adjacent, AreaShape_FormFactor
    df = df.fillna(value=-1)
    return df


def plot_umap0(df, marker, 
               color_map = {'WT': 1, 'TDP43': 2, 'TBK1': 3, 'OPTN': 4, 'FUSRevertant': 5, 'FUSHomozygous': 6, 'FUSHeterozygous': 7},
               stress = True, customized = False):
    """
    MICROGLIA TEST: change color_map:
    color_map = {'WT': 1, 'TDP43': 2, 'TBK1': 3, 'FUSHomozygous': 4},

    Remember to change the use_condition variable in preprocessing as well when plotting stress
    """
    
    logging.info(f"Starting plot_umap0() of marker {marker}...\n")
    #logging.info('%s %s', "\n", df.value_counts(subset=['replicate', 'cell_line', 'marker']))
    
    from src.common.lib.metrics import calc_ari_with_kmeans
    from src.common.lib.umap_plotting import __get_metrics_figure
    from matplotlib.gridspec import GridSpec
    
    if stress:
        # keep relevant columns only
        temp_df = df.drop(['replicate', 'marker', 'cell_line'], axis=1, inplace=False)
        temp_df.set_index('treatment', inplace=True)
        # get true labels
        true_labels = temp_df.index
        # calculate first 2 UMAP components
        umaps = umap.UMAP(random_state=42, n_components=2, n_jobs=1).fit_transform(temp_df)
        
        # calculate clustering scores
        #scores = calc_clustering_validation_metric(umaps, true_labels, metrics=['ARI'])

        # UMAP plotting
        color_map = {'Untreated':'#52C5D5', 'stress':'#F7810F'} #cyan and orange
        # colors = true_labels.map(color_map)
        df_new = pd.DataFrame(umaps, columns = ['UMAP1', 'UMAP2'], index = true_labels).reset_index()

        fig = plt.figure()
        gs = GridSpec(2,1,height_ratios=[20,1])
        ax = fig.add_subplot(gs[0])
        sns.scatterplot(data=df_new, x='UMAP1', y='UMAP2', ax = ax,
                             hue='treatment', palette = color_map, s = 30)
        gs_bottom = fig.add_subplot(gs[1])
        __get_metrics_figure(umaps, np.array(true_labels), ax=gs_bottom)
        ax.legend().set_title('')
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        # ax.set_xlabel("UMAP1",fontsize=15)
        # ax.set_ylabel("UMAP2",fontsize=15)
        ax.tick_params(left=False, bottom=False)
        ax.set_title(f'{BATCH_TO_RUN}_{marker}')
        fig.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'UMAP0_{marker}_stress_ARI.png'))
        plt.clf()
        
        logging.info(f'UMAP0 with ARI of {marker} saved')
        
        """
        for name, score in scores.items():
            plt.annotate(f'{name}: {score}',  # Your string
            # The point that we'll place the text in relation to 
            xy=(0.5, 0), 
            # Interpret the x as axes coords, and the y as figure coords
            xycoords=('axes fraction', 'figure fraction'),
            # The distance from the point that the text will be at
            xytext=(0, 5),  
            # Interpret `xytext` as an offset in points...
            textcoords='offset points',
            # Any other text parameters we'd like
            size=14, ha='center', va='bottom')
        """
        
    else:
        # UMAP0 - population: all cell lines and a single marker, color by: cell line
        umap_df1 = df.drop(['replicate', 'marker'], axis=1, inplace=False)
        umap_df1.set_index('cell_line', inplace=True)
        indices1 = umap_df1.index
        embedding1 = umap.UMAP(random_state=42, n_jobs=1).fit_transform(umap_df1)
        
        # UMAP0 - population: all cell lines and a single marker, color by: rep (well effect)
        umap_df2 = df.drop(['marker', 'cell_line'], axis=1, inplace=False)
        umap_df2.set_index(['replicate'], inplace=True)
        indices2 = umap_df2.index
        embedding2 = umap.UMAP(random_state=42, n_jobs=1).fit_transform(umap_df2)
        
        # Allow for specific plots; adjust name manually
        if customized:
            pdf_file = os.path.join(OUTPUT_DIR, f'UMAP0_FUS_FUS-lines-only.pdf')
        else:
            pdf_file = os.path.join(OUTPUT_DIR, f'UMAP0_{marker}.pdf')
        
        pdf_pages = bpdf.PdfPages(pdf_file)

        # Iterate over the UMAP embeddings and create a scatter plot for each dataframe
        for idx, mapper in zip([indices1, indices2], [embedding1, embedding2]):
            # Use the same coloring for each cell line for the different marker UMAPs
            if 'WT' in idx:
                colors = idx.map(color_map)
            else:
                unique_indices = idx.unique()
                color_map = {index: color for index, color in zip(unique_indices, range(len(unique_indices)))}
                colors = idx.map(color_map)
            scatter = plt.scatter(mapper[:,0], mapper[:,1], c=colors, alpha=0.5, cmap='Spectral')
            legend_handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(color)), label=index) for index, color in color_map.items()]
            legend_handles.sort(key=lambda patch: patch.get_label())
            plt.legend(handles=legend_handles, bbox_to_anchor=(0, 0, 1, 1))
            plt.title(f'{marker}', fontsize=22)
            plt.ylabel("UMAP 2")
            plt.xlabel("UMAP 1")
            pdf_pages.savefig()
            
            logging.info(f'UMAP0 of {marker} saved')
            plt.clf()
        
        pdf_pages.close()
    
    return None


def plot_umap1(df_all):
    
    logging.info(f"\nStarting plot_umap1() ...")
    logging.info('%s %s', "\n", df_all.value_counts(subset=['replicate', 'cell_line', 'marker']))

    umap_df = df_all.drop(['replicate'], axis=1, inplace=False)
    
    # UMAP1 - population: all cell lines and all markers, color by: marker and cell line 
    umap_df1 = umap_df.set_index(['marker', 'cell_line'], inplace=False)
    umap_df1.index = umap_df1.index.map('_'.join)
    indices1 = umap_df1.index
    embedding1 = umap.UMAP(random_state=42, n_jobs=1).fit_transform(umap_df1)

    # UMAP1 - population: all cell lines and all markers, color by: marker 
    umap_df2 = umap_df.set_index(['marker'], inplace=False)
    umap_df2.drop(['cell_line'], axis=1, inplace=True)
    indices2 = umap_df2.index
    embedding2 = umap.UMAP(random_state=42, n_jobs=1).fit_transform(umap_df2)
    
    # UMAP1 - population: all cell lines and all markers, color by: cell line 
    umap_df3 = umap_df.set_index(['cell_line'], inplace=False)
    umap_df3.drop(['marker'], axis=1, inplace=True)
    indices3 = umap_df3.index
    embedding3 = umap.UMAP(random_state=42, n_jobs=1).fit_transform(umap_df3)
    
    pdf_file = os.path.join(OUTPUT_DIR, f'UMAP1_{BATCH_TO_RUN}.pdf')
    pdf_pages = bpdf.PdfPages(pdf_file)
  
    # Iterate over the UMAP embeddings and create a scatter plot for each dataframe
    for idx, mapper in zip([indices1, indices2, indices3], [embedding1, embedding2, embedding3]):
        unique_indices = idx.unique()
        color_map = {index: color for index, color in zip(unique_indices, range(len(unique_indices)))}
        colors = idx.map(color_map)
        scatter = plt.scatter(mapper[:,0], mapper[:,1], s=5, c=colors, alpha=0.5, cmap='Spectral')
        legend_handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(color)), label=index) for index, color in color_map.items()]
        legend_handles.sort(key=lambda patch: patch.get_label())
        plt.legend(handles=legend_handles, bbox_to_anchor=(0, 0, 1, 1), fontsize = 'x-small')
        plt.ylabel("UMAP 2")
        plt.xlabel("UMAP 1")
        pdf_pages.savefig()
        plt.clf()
    
    pdf_pages.close()
    
    logging.info(f'UMAP1 of {BATCH_TO_RUN} saved')

    return None


def load_CP_features(input_path, stress = True):
    # load CP features (all)
    if stress:
        df = pd.read_csv(os.path.join(input_path,f'stress_all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'))
    else:
        df = pd.read_csv(os.path.join(input_path,f'all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'))
    print("\n\nXXXX df.shape", df.shape)   
    # remove Unnamed column
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

def remove_SCNA_cell_line(df):
    # Drop SCNA line because too many missing values
    df = df[df['cell_line'].isin(['WT', 'TDP43', 'TBK1', 'OPTN', 'FUSRevertant', 'FUSHomozygous', 'FUSHeterozygous'])]  
    return df

def keep_intersected_markers(df):
    # Remove markers that don't have data in all conditions 
    groups = df.groupby(['replicate', 'cell_line'])
    # List of lists, where each list holds the markers names that appear in this group 
    markers_lists = []
    for name, group in groups:
        # Keep only markers that appear in this group
        markers_lists.append(group['marker'].unique())
    # Get the intersection of marker names 
    markers = set.intersection(*map(set,markers_lists))
    df = df[df['marker'].isin(markers)]
    
    # NANCYS TEST
    # [ 'CD41', 'TDP43', 'NONO', 'Phalloidin', 'TOMM20', 'FUS', 'PML', 'FMRP', 'NCL', 'PSD95', 'NEMO', 'CLTC', 'mitotracker', 'GM130', 'PEX14', 'SQSTM1', 'DCP1A']
    #df = df[df['marker'].isin(['NCL', 'FUS', 'NONO', 'CD41', 'TOMM20'])] 
    
    logging.info(f"\n\n XXXX df.shape: {df.shape}, \nTotal of {len(markers)} intersecting markers:{markers}")   
    return df

    
def plot_umap2(input_path, stress = False):
    """
    Receives the dataframe that contains image measurements concatenated by object type, with one row per rep - marker - cell line combination
    = df_all from load_data_and_plot_UMAPs(input_path), which is saved under INPUT_DIR_BATCH = input_path
    = after pre-processing, so columns to exclude already excluded 
    """
    
    logging.info(f"\n\nStarting to concatenate markers and plot UMAP2 from Cell Profiler output of batch: {INPUT_DIR_BATCH}")
    
    df_all = load_CP_features(input_path)

    if stress == False:
        df_all = remove_SCNA_cell_line(df_all)
        df_all = keep_intersected_markers(df_all)
    else:
        df_all = keep_intersected_markers(df_all)
    
    
    ####################################################################
    # Group by and create a nested dict with all indices to match
    tmp_d = defaultdict(list)
    
    if stress:
        groups = df_all.groupby(['replicate', 'treatment'])
    else:
        groups = df_all.groupby(['replicate', 'cell_line'])
    
    for c_name, group in groups:
        tmp_d[c_name] = {}
        marker_groups = group.groupby(['marker'])
        for marker_name, marker_group in marker_groups:
            res = marker_group.index.tolist()
            shuffle(res) #inplace
            tmp_d[c_name][marker_name] = []
            tmp_d[c_name][marker_name] = res
            print(c_name, marker_name, "the len of res is ", len(res))
    
    ####################################################################
    # Build a new dataframe, after matching marker features (within a cell line condition)
    final_df_list = []
    # for cell line
    for k,v in tmp_d.items():
        cell_df_list = []
        # concat a row of each marker  
        for marker, indexes in v.items():
            # get sub-df: all indexes of this marker 
            marker_df = df_all.loc[indexes]
            # change column name to allow concat
            marker_df.columns = [col+f'_{marker}' if col not in ['replicate', 'cell_line', 'marker', 'treatment'] else col for col in marker_df.columns]
            marker_df = marker_df.reset_index(drop=True, inplace=False) # reset_index() is important, to remove NaN rows
            print("marker_df", marker, marker_df.shape)
            # append to list of sub-dfs
            cell_df_list.append(marker_df) 
            
        # concat columns (stack columns one after the another)
        cell_df = pd.concat(cell_df_list, axis=1)
        print("cell_df", k, cell_df.shape) #some cell lines don't have data in all markers, so number of columns is different 
        
        # workaround - removing rows with Nan values (many markers don't have 100 rows)
        cell_df = cell_df.dropna(axis=0)
        print("cell_df after NA removal", k, cell_df.shape)
        
        # remove duplicate columns (replicate, cell_line)
        cell_df = cell_df.loc[:,~cell_df.columns.duplicated()]
        print(cell_df.shape, "duplicated columns after removal", cell_df.columns[cell_df.columns.duplicated()])
        final_df_list.append(cell_df)     
        
    # concat dataframe of cell lines (stack rows)
    final_df = pd.concat(final_df_list, axis=0)

    #final_df = final_df.reset_index(drop=True, inplace=False)
    print("final_df", final_df.shape)
    if stress:
        final_df.to_csv(os.path.join(INPUT_DIR_BATCH, f'UMAP2_stress-dataframe_{BATCH_TO_RUN}.csv'))
    else:
        final_df.to_csv(os.path.join(INPUT_DIR_BATCH, f'UMAP2_dataframe_{BATCH_TO_RUN}.csv'))
   
    ####################################################################
    # Plotting the UMAP2
    umap_df = final_df.drop(['replicate'], axis=1, inplace=False)
    umap_df.drop(['marker'], axis=1, inplace=True)
    
    # workaround - removing columns with Nan values 
    umap_df = umap_df.dropna(axis=1)
    
    if stress:
        umap_df = umap_df.drop(['cell_line'], axis=1, inplace=False)
        umap_df.set_index('treatment', inplace=True)
        indices = umap_df.index
        embedding = umap.UMAP(random_state=42, n_jobs=1).fit_transform(umap_df)
        
        color_map = {'Untreated':'cyan', 'stress':'orange'}
        colors = indices.map(color_map)
        scatter = plt.scatter(embedding[:,0], embedding[:,1], s=20, c=colors)
        legend_handles = [mpatches.Patch(color=color, label=index) for index, color in color_map.items()]
        legend_handles.sort(key=lambda patch: patch.get_label())
        plt.legend(handles=legend_handles, bbox_to_anchor=(0, 0, 1, 1))
        plt.ylabel("UMAP 2")
        plt.xlabel("UMAP 1")
        plt.savefig(os.path.join(OUTPUT_DIR, f'UMAP2_stress.pdf'))
        plt.clf()
        logging.info(f'UMAP2 of of {BATCH_TO_RUN} stress treatment saved')
    else:
        umap_df = umap_df.set_index('cell_line', inplace=False)
        indices = umap_df.index
        embedding = umap.UMAP(random_state=42, n_jobs=1).fit_transform(umap_df)

        color_map = {'WT': 1, 'TDP43': 2, 'TBK1': 3, 'OPTN': 4, 'FUSRevertant': 5, 'FUSHomozygous': 6, 'FUSHeterozygous': 7}
        colors = indices.map(color_map)
        scatter = plt.scatter(embedding[:,0], embedding[:,1], s=15, c=colors, cmap='Spectral', alpha=0.7) 
        legend_handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(color)), label=index) for index, color in color_map.items()]
        legend_handles.sort(key=lambda patch: patch.get_label())
        plt.legend(handles=legend_handles, bbox_to_anchor=(0, 0, 1, 1), fontsize = 'x-small')
        plt.ylabel("UMAP 2")
        plt.xlabel("UMAP 1")
        plt.savefig(os.path.join(OUTPUT_DIR, f'UMAP2_{BATCH_TO_RUN}.pdf'))
        plt.clf()
        logging.info(f'UMAP2 of {BATCH_TO_RUN} saved')
            
        # for marker, marker_group in marker_groups:
        #     logging.info(f"\n\Marker group name:{marker} {marker_group.shape}")
        #     selected = marker_group.sample(1)
        #     tmp_d[].append(selected)
        #     print(df_all.shape)
            
        #     print(df_all.shape)
        #     print("tmp_list", tmp_list)
        
        
    logging.info(f"\n\nFinished plotting UMAP2 of: {BATCH_TO_RUN}")


def check_na(df):
    count = df.isna().sum()
    new_df = (pd.concat([count.rename('missing_count'),
                        count.div(len(df))
                            .rename('missing_ratio')],axis = 1)
                .loc[count.ne(0)]).sort_values(by='missing_ratio', ascending=False)
    print(new_df[new_df['missing_count']>0.01])


def customized_plot():
    """
    To plot UMAP0 of FUS marker in only WT and FUS lines. 
    """
    df = pd.read_csv(os.path.join(INPUT_DIR_BATCH, 'FUS_all.csv'))
    marker = 'FUS'
    df['marker'] = marker
    
    df = preprocessing(df)
    
    options = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
    df_umap = df[df['cell_line'].isin(options)]
    df_umap.drop('Unnamed: 0', axis = 1, inplace=True)
    df_umap.to_csv(os.path.join(INPUT_DIR_BATCH, f'FUS_FUS-lines-only_preprocessed.csv'))
    plot_umap0(df_umap, marker, color_map = {'WT': 1, 'FUSRevertant': 2, 'FUSHomozygous': 3, 'FUSHeterozygous': 4})
    
    return None
            
def main():
    logging.info(f"\n\nStarting to plot marker UMAPs from Cell Profiler output of batch: {INPUT_DIR_BATCH}")
    load_data_and_plot_UMAPs(INPUT_DIR_BATCH, stress = False)
    #plot_umap2(INPUT_DIR_BATCH)
    #customized_plot()
    logging.info(f"\n\nFinished plotting marker UMAPs from Cell Profiler output of batch: {INPUT_DIR_BATCH}")


if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    main()    

    # TBD - support analysis across batches  
    
    # UMAP1 - color only one marker at a time, save in one PDF
    
    # UMAP2 - deal with missing values?   
    # Remove correlated features 

