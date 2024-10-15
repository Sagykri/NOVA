import pandas as pd
import os

BATCH_TO_RUN = 'batch9' 

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR, 'outputs','cell_profiler')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)

# data = pd.read_csv(os.path.join(INPUT_DIR_BATCH, 'combined', 'stress_all_markers_concatenated-by-object-type_batch9.csv'))

# print(data.head())
# print(data.keys())

x = pd.read_csv(os.path.join(INPUT_DIR_BATCH, 'combined', 'CD41_all.csv'))
print(x.head())
print(x.keys())