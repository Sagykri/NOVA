import pandas as pd 
import os
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import umap     #installation of umap-learn is required
import umap.plot
import scipy
from scipy import stats
import math

import CellProfiler_plotting as cp

# Global paths
BATCH_TO_RUN = 'batch7' 

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR, 'outputs','cell_profiler')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN, 'combined')
OUTPUT_DIR = os.path.join(INPUT_DIR, BATCH_TO_RUN, 'plots')

LOG_DIR_PATH = os.path.join(INPUT_DIR, 'logs')

def get_data(input_path, marker):
    df = pd.read_csv(os.path.join(input_path,f'all_markers_concatenated-by-object-type_{BATCH_TO_RUN}.csv'))
    df = df.loc[df['marker'] == marker]
    return df

def correlate_dimensions(df, stress = False):
    marker = df['marker'].unique()[0]
    
    if stress:
        umap_df = df.drop(['cell_line', 'replicate', 'marker', 'Unnamed: 0'], axis=1, inplace=False)
        umap_df.set_index('treatment', inplace=True)
    else:
        umap_df = df.drop(['replicate', 'marker', 'Unnamed: 0'], axis=1, inplace=False)
        # If taking specific cell lines:
        #umap_df = umap_df.loc[(umap_df['cell_line'] == 'WT') | (umap_df['cell_line'] == 'FUSHeterozygous') | (umap_df['cell_line'] == 'FUSHomozygous') | (umap_df['cell_line'] == 'FUSRevertant')]
        umap_df = umap_df.loc[(umap_df['cell_line'] == 'WT') | (umap_df['cell_line'] == 'FUSHeterozygous')]

        umap_df.set_index('cell_line', inplace=True)
    
    print(f'number of features: {len(umap_df.columns)}')
    #indices = umap_df.index
    embedding = umap.UMAP(random_state=42, n_jobs=1).fit_transform(umap_df)

    corr_df = pd.DataFrame()
    corr_df['feature'] = umap_df.keys()
    corr_df = corr_df.assign(pearson_1 = None)
    corr_df = corr_df.assign(pearson_2 = None)
    corr_df = corr_df.assign(spearman_1 = None)
    corr_df = corr_df.assign(spearman_2 = None)
    corr_df = corr_df.assign(combined_pearson = None)
    corr_df = corr_df.assign(combined_spearman = None)
    
    for feature in umap_df.keys():
        pearson_a = scipy.stats.pearsonr(umap_df[feature], embedding[:,0]).statistic
        pearson_b = scipy.stats.pearsonr(umap_df[feature], embedding[:,1]).statistic
        combined_pearson = math.sqrt(pow(pearson_a,2) + pow(pearson_b,2))
        spearman_a = scipy.stats.spearmanr(umap_df[feature], embedding[:,0])[0]
        spearman_b = scipy.stats.spearmanr(umap_df[feature], embedding[:,1])[0]
        combined_spearman = math.sqrt(pow(spearman_a,2) + pow(spearman_b,2))
        
        corr_df.loc[corr_df['feature'] == feature, 'pearson_1'] = pearson_a
        corr_df.loc[corr_df['feature'] == feature, 'pearson_2'] = pearson_b
        corr_df.loc[corr_df['feature'] == feature, 'spearman_1'] = spearman_a
        corr_df.loc[corr_df['feature'] == feature, 'spearman_2'] = spearman_b
        corr_df.loc[corr_df['feature'] == feature, 'combined_pearson'] = combined_pearson       
        corr_df.loc[corr_df['feature'] == feature, 'combined_spearman'] = combined_spearman
    
    corr_df = corr_df.sort_values(by = 'combined_spearman', ascending = False)
    if stress:
        corr_df.to_csv(os.path.join(OUTPUT_DIR, f'feature-correlation_{marker}_stress_{BATCH_TO_RUN}.csv'))
    else:
        #corr_df.to_csv(os.path.join(OUTPUT_DIR, f'feature-correlation_{marker}_lines_{BATCH_TO_RUN}.csv'))
        corr_df.to_csv(os.path.join(OUTPUT_DIR, f'feature-correlation_{marker}_WT_FUSHetero_{BATCH_TO_RUN}.csv'))

    # Boxplots of top 10 and bottom 10 features
    #corr_df.head(10)
    #corr_df.tail(10)
    
def main():
    logging.info(f"\n\nStarting to test feature contribution of batch: {INPUT_DIR_BATCH}")
    
    #for marker in ['G3BP1', 'FMRP', 'TOMM20', 'mitotracker', 'PURA', 'PML', 'TDP43']:
    #    correlate_dimensions(get_data(INPUT_DIR_BATCH, marker))
    correlate_dimensions(get_data(INPUT_DIR_BATCH, 'ANXA11'))

if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    cp.set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    main()  