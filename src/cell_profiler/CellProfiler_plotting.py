## Plot UMAPs from CellProfiler output

#Packages
import pandas as pd 
import os
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import umap     #installation of umap-learn is required
import umap.plot
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.backends.backend_pdf as bpdf

# Global paths
BATCH_TO_RUN = 'batch6' 

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR, 'outputs','cell_profiler')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN, 'combined')
OUTPUT_DIR = os.path.join(INPUT_DIR, BATCH_TO_RUN, 'plots')

LOG_DIR_PATH = os.path.join(INPUT_DIR, 'logs')
    
    
def set_logging(log_file_path, level=logging.INFO, format=' INFO: %(message)s'):
    formatter = '%(asctime)s %(levelname)-8s %(message)s'
    handlers = [logging.FileHandler(log_file_path + '.log'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=formatter, handlers=handlers, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(__doc__)


def load_data_and_plot_UMAPs(input_path):
     with os.scandir(input_path) as input_data_folder:
        # We loop on all markers, plot single marker UMAPs (AKA UMAP0), 
        # but also keep the marker data for later UMAP1 (of all markers in one plot)
        all_markers_df = []
        
        for features_file in input_data_folder:
            
            file = os.path.basename(features_file.path)
            if (file == f'all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'):
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
            
            plot_umap0(df, marker)
            
            all_markers_df.append(df)
            
            #print(f"After preprocessing: {df.shape}")
            """
            from 358 to 352 columns while only 4 should be dropped?
            """
        
        # Combine markers
        #df_all = pd.concat(all_markers_df)
        #df_all.to_csv(os.path.join(INPUT_DIR_BATCH, f'all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'))
        
        # Plot UMAP1
        #plot_umap1(df_all)
        
        return None
                  
            
def preprocessing(features_df, use_condition='stress', exclude_cols=['Parent_Nucleus', 'object_type', 'Unnamed: 0_PrimaryObject1', 'Unnamed: 0_PrimaryObject2', 'Unnamed: 0_SecondaryObject']):
    """
    Prepare the data for clustering
    - Included desired condition
    - Remove columns (in load_data_and_plot_UMAPs, all columns converted to data type 'float', so manually write columns to remove)
    - Handle misssing values
    """
    df = features_df.copy()
    # Check if the columns you wish to exclude, really exist in the df
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    
    # Take the desired condition and corresponding columns
    if use_condition == 'stress':
        df = df.loc[df['cell_line'] == 'WT'] #shortcut now because only WT has stress / no stress
        df.drop(exclude_cols, axis = 1, inplace=True)
    elif use_condition == 'Untreated':
        df = df.loc[df['treatment'] == use_condition]
        df.drop(exclude_cols, axis=1, inplace=True)
        df.drop('treatment', axis=1, inplace=True)
    
    # Data was converted to type 'object', so convert back to float to support UMAP
    cols_to_convert = [col for col in df.columns if col not in ['replicate', 'treatment', 'cell_line', 'marker']]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric)
    
    # Four features have misssing values, replace them with -1 (Neighbors_AngleBetweenNeighbors_Adjacent,Neighbors_SecondClosestDistance_Adjacent, Neighbors_FirstClosestDistance_Adjacent, AreaShape_FormFactor
    df = df.fillna(value=-1)
    return df


def plot_umap0(df, marker, 
               color_map = {'WT': 1, 'TDP43': 2, 'TBK1': 3, 'SCNA': 4, 'OPTN': 5, 'FUSRevertant': 6, 'FUSHomozygous': 7, 'FUSHeterozygous': 8},
               stress = True):
    
    logging.info(f"Starting plot_umap0() of marker {marker}...\n")
    logging.info('%s %s', "\n", df.value_counts(subset=['replicate', 'cell_line', 'marker']))
    
    if stress:
        umap_df = df.drop(['replicate', 'marker', 'cell_line'], axis=1, inplace=False)
        umap_df.set_index('treatment', inplace=True)
        indices = umap_df.index
        embedding = umap.UMAP(random_state=42, n_jobs=1).fit_transform(umap_df)
        
        color_map = {'Untreated':'cyan', 'stress':'orange'}
        colors = indices.map(color_map)
        scatter = plt.scatter(embedding[:,0], embedding[:,1], c=colors)
        legend_handles = [mpatches.Patch(color=color, label=index) for index, color in color_map.items()]
        legend_handles.sort(key=lambda patch: patch.get_label())
        plt.legend(handles=legend_handles, bbox_to_anchor=(0, 0, 1, 1))
        plt.title(f'{marker}', fontsize=22)
        plt.ylabel("UMAP 2")
        plt.xlabel("UMAP 1")
        plt.savefig(os.path.join(OUTPUT_DIR, f'UMAP0_{marker}_stress.pdf'))
        plt.clf()
        logging.info(f'UMAP0 of {marker} saved')
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
            scatter = plt.scatter(mapper[:,0], mapper[:,1], c=colors, cmap='Spectral')
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
        scatter = plt.scatter(mapper[:,0], mapper[:,1], s=5, c=colors, cmap='Spectral')
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


def plot_umap2(input_path):
    """
    Receives the dataframe that contains image measurements concatenated by object type, with one row per rep - marker - cell line combination
    = df_all from load_data_and_plot_UMAPs(input_path), which is saved under INPUT_DIR_BATCH = input_path
    = after pre-processing, so columns to exclude already excluded 
    """
    from random import shuffle
    from collections import defaultdict
    logging.info(f"\n\nStarting to concatenate markers and plot UMAP2 from Cell Profiler output of batch: {INPUT_DIR_BATCH}")
    
    # load CP features (all)
    df_all = pd.read_csv(os.path.join(input_path,f'all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'))
    print("\n\nXXXX df_all.shape", df_all.shape)   
    # remove Unnamed column
    df_all.drop('Unnamed: 0', axis=1, inplace=True)
    
    ########### REMOVE THIS FOR TEST
    # df_all = df_all[['AreaShape_BoundingBoxMaximum_Y_PrimaryObject1',
    #                 'AreaShape_BoundingBoxMinimum_X_PrimaryObject1',
    #                 'AreaShape_BoundingBoxMinimum_Y_PrimaryObject1',
    #                 'AreaShape_Center_X_PrimaryObject1',
    #                 'AreaShape_Center_Y_PrimaryObject1', 'replicate', 'cell_line', 'marker']]
    # print("\n\nXXXX df_all.shape", df_all.shape)   
    
    df_all = df_all[df_all['marker'].isin(['ANXA11', 'CD41', 'CLTC', 'Calreticulin', 'DCP1A', 'FMRP', 'FUS', 'GM130', 'KIF5A', 'LAMP1', 'NCL', 'NEMO', 'NONO',
'PEX14', 'PML', 'PSD95', 'PURA', 'Phalloidin' , 'SQSTM1', 'TDP43', 'TIA1', 'TOMM20', 'mitotracker'])]
    print("\n\nXXXX df_all.shape", df_all.shape)   
    ########### REMOVE THIS FOR TEST
    



    
    ####################################################################
    # Group by and create a nested dict with all indecies to match
    tmp_d = defaultdict(list)
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
            marker_df.columns = [col+f'_{marker}' if col not in ['replicate', 'cell_line', 'marker'] else col for col in marker_df.columns]
            marker_df = marker_df.reset_index(drop=True, inplace=False) # reset_index() is important, to remove NaN rows
            print("marker_df", marker, marker_df.shape)
            # append to list of sub-dfs
            cell_df_list.append(marker_df) 
            
        # concat columns (stack columns one after the another)
        cell_df = pd.concat(cell_df_list, axis=1)
        print("cell_df", k, cell_df.shape)
        
        #print(cell_df.reset_index(drop=True, inplace=False).head(50))
        #cell_df = cell_df.reset_index(drop=True, inplace=False)
        final_df_list.append(cell_df.reset_index(drop=True))
        
        # Weli, I had a tough fight with this error "pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects"
        # that we get in line "final_df = pd.concat(final_df_list, join="inner", axis=0, ignore_index=True)""
        # conclusion: it is because we try to stack rows, but each df has different columns and number of rows. tried to add "inner" and other stuff, but didn't work.
        
    
    #final_df = final_df.reset_index(drop=True, inplace=False)
    # concat dataframe of cell lines (stack rows)
    final_df = pd.concat(final_df_list)
    print("final_df", final_df.shape)
    
    
    
    # UMAP code - not final
    umap_df = final_df.drop(['replicate'], axis=1, inplace=False)
    
    
    # workaround - removing columns with Nan values 
    umap_df = umap_df.dropna(axis=1)
    
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
    
    pdf_file = os.path.join(OUTPUT_DIR, f'UMAP2_{BATCH_TO_RUN}.pdf')
    pdf_pages = bpdf.PdfPages(pdf_file)
  
    # Iterate over the UMAP embeddings and create a scatter plot for each dataframe
    for idx, mapper in zip([indices2, indices3], [embedding2, embedding3]):
        unique_indices = idx.unique()
        color_map = {index: color for index, color in zip(unique_indices, range(len(unique_indices)))}
        colors = idx.map(color_map)
        scatter = plt.scatter(mapper[:,0], mapper[:,1], s=5, c=colors, cmap='Spectral')
        legend_handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(color)), label=index) for index, color in color_map.items()]
        legend_handles.sort(key=lambda patch: patch.get_label())
        plt.legend(handles=legend_handles, bbox_to_anchor=(0, 0, 1, 1), fontsize = 'x-small')
        plt.ylabel("UMAP 2")
        plt.xlabel("UMAP 1")
        pdf_pages.savefig()
        plt.clf()
    
    pdf_pages.close()
    
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
            
def main():
    logging.info(f"\n\nStarting to plot marker UMAPs from Cell Profiler output of batch: {INPUT_DIR_BATCH}")
    load_data_and_plot_UMAPs(INPUT_DIR_BATCH)
    #plot_umap2(INPUT_DIR_BATCH)
    logging.info(f"\n\nFinished plotting marker UMAPs from Cell Profiler output of batch: {INPUT_DIR_BATCH}")


if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    main()    

    # TBD - support analysis of 1) across batches 2) WT untreated vs. stress
    # TBD - check all column names for potential confounders 
    
    # UMAP0 - population: all cell lines and a single marker, color by: cell line and batch (batch effect) - TBD
    # UMAP0 - population: WT cell line and a single marker, color by: treatment and rep (well effect) - TBD
    
    # UMAP1 - color only one marker at a time, save in one PDF
    
    # UMAP2 - population: all cell lines and all markers concatenated (so smaller population)   
    # Remove correlated features 

