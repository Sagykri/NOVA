import json
import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging
import datetime

from src.common.lib.utils import get_if_exists, load_config_file
from src.common.lib.embeddings_utils import load_embeddings
from src.common.lib.plotting import plot_umap0, plot_umap1, plot_umap2
from src.common.lib.utils import handle_log

from src.Analysis.analyzer_umap0 import AnalyzerUMAP0
from src.Analysis.analyzer_umap1 import AnalyzerUMAP1
from src.Analysis.analyzer_umap2 import AnalyzerUMAP2

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig

def generate_umaps():
    if len(sys.argv) < 4:
        raise ValueError("Invalid arguments. Must supply trainer config and data config and UMAP type! ('umap0','umap1','umap2').")
    
    config_path_trainer = sys.argv[1]
    config_trainer = load_config_file(config_path_trainer, 'data')
    model_output_folder = config_trainer.OUTPUT_FOLDER_PATH #TODO: change this to the right name
    handle_log(model_output_folder)

    config_path_data = sys.argv[2]
    config_data = load_config_file(config_path_data, 'data')
    
    train_batches = get_if_exists(config_data, 'TRAIN_BATCHES', None)
    if train_batches:
        logging.info(f'config_data.TRAIN_BATCHES: {train_batches}')

    embeddings, labels = load_embeddings(model_output_folder, config_data, train_batches)
    
    umap_type = sys.argv[3]
    if umap_type=='umap0':
        logging.info("[Generate UMAPs 0 (vit)]")
        u = AnalyzerUMAP0(config_trainer, config_data)
        u.calculate(embeddings, labels)
        output_folder_path = os.path.join(model_output_folder, 'figures', u.experiment_type,'UMAP', umap_type)
        plot_umap0(u.features, config_data, output_folder_path)

    elif umap_type=='umap1':
        logging.info("[Generate UMAP 1 (vit)]")
        u = AnalyzerUMAP1(config_trainer, config_data)
        u.calculate(embeddings, labels)
        output_folder_path = os.path.join(model_output_folder, 'figures', u.experiment_type,'UMAP', umap_type)
        plot_umap1(u.features, config_data, output_folder_path)

    elif umap_type=='umap2':
        logging.info("[Generate SM (umap2, vit)]")
        u = AnalyzerUMAP2(config_trainer, config_data)
        u.calculate(embeddings, labels)
        output_folder_path = os.path.join(model_output_folder, 'figures', u.experiment_type,'UMAP', umap_type)
        plot_umap2(u.features, config_data, output_folder_path)
        

# def __generate_umap1(config_data, output_folder_path, embeddings, labels):
        
#     title = f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}_{'_'.join(config_data.REPS)}"
#     __now = datetime.datetime.now()
   
#     saveroot = os.path.join(output_folder_path,\
#                             'UMAPs',\
#                             'UMAP1')
#     if not os.path.exists(saveroot):
#         os.makedirs(saveroot, exist_ok=True)
        
#     with open(os.path.join(saveroot, 'config.json'), 'w') as json_file:
#         json.dump(config_data.__dict__, json_file, indent=4)
        
#     savepath = os.path.join(saveroot,
#                                 f'{__now.strftime("%d%m%y_%H%M%S_%f")}_{title}')    
#     map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
#     if map_labels_function is not None:
#         map_labels_function = eval(map_labels_function)(config_data)

#     ordered_marker_names = get_if_exists(config_data, 'ORDERED_MARKER_NAMES', None)
#     if ordered_marker_names is not None:
#         ordered_names = [config_data.UMAP_MAPPINGS[marker]['alias'] for marker in ordered_marker_names]
#     else:
#         ordered_names = None
    
#     umap_embeddings = compute_umap_embeddings(embeddings)
#     label_data = map_labels_function(labels) if map_labels_function is not None else labels
#     logging.info(f'label_data uniqie: {np.unique(label_data)}')
#     plot_umap_embeddings(umap_embeddings, label_data, config_data, savepath, 
#                         ordered_names = ordered_names, show_ari=False)
#     logging.info(f"UMAP saved successfully to {savepath}")

# def __generate_umap0(config_data, output_folder_path, embeddings, labels, add_title=None):
    
#     markers = np.unique([m.split('_')[0] if '_' in m else m for m in np.unique(labels.reshape(-1,))]) 
#     logging.info(f"Detected markers: {markers}")
#     # __now = datetime.datetime.now()
#     title = f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}_{'_'.join(config_data.REPS)}" #_{__now.strftime('%d%m%y_%H%M%S_%f')}"
#     if add_title is not None:
#         title = f"{title}_{add_title}"
    
#     saveroot = os.path.join(output_folder_path,\
#                                 'UMAPs',\
#                                     f'{title}')
#     if not os.path.exists(saveroot):
#         os.makedirs(saveroot, exist_ok=True)
            
#     with open(os.path.join(saveroot, 'config.json'), 'w') as json_file:
#         json.dump(config_data.__dict__, json_file, indent=4)    
    
#     for c in markers:
#         logging.info(f"Marker: {c}")
#         logging.info(f"[{c}] Selecting indexes of marker")
#         c_indexes = np.where(np.char.startswith(labels.astype(str), f"{c}_"))[0]
#         logging.info(f"[{c}] {len(c_indexes)} indexes have been selected")

#         if len(c_indexes) == 0:
#             logging.info(f"[{c}] Not exists in embedding. Skipping to the next one")
#             continue

#         embeddings_c, labels_c = np.copy(embeddings[c_indexes]), np.copy(labels[c_indexes].reshape(-1,))
        
#         logging.info(f"[{c}] Plot umap...")
#         savepath = os.path.join(saveroot, f'{c}') # NANCY
        
#         map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
#         if map_labels_function is not None:
#             map_labels_function = eval(map_labels_function)(config_data)
#         label_data = map_labels_function(labels_c) if map_labels_function is not None else labels_c
#         if np.unique(label_data).shape[0]>10:
#             show_ari=False
#         else:
#             show_ari=True
                
#         umap_embeddings = compute_umap_embeddings(embeddings_c)
#         cell_line_cond_high = get_if_exists(config_data, 'CELL_LINE_COND_HIGH', None)
#         plot_umap_embeddings(umap_embeddings, label_data, config_data, 
#                              savepath=savepath, show_ari=show_ari, title=c,
#                              cell_line_cond_high=cell_line_cond_high)
#         logging.info(f"[{c}] UMAP saved successfully to {savepath}")
        

# def __generate_sm(config_data, output_folder_path, embeddings, labels, add_title=None):
#     title = f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}_{'_'.join(config_data.REPS)}"
#     if add_title is not None:
#         title = f"{title}_{add_title}"
#     __now = datetime.datetime.now()
    
#     saveroot = os.path.join(output_folder_path,\
#                             'UMAPs',\
#                             'SM_UMAPs',title)
#     if not os.path.exists(saveroot):
#         os.makedirs(saveroot, exist_ok=True)
        
#     with open(os.path.join(saveroot, 'config.json'), 'w') as json_file:
#         json.dump(config_data.__dict__, json_file, indent=4)    

#     savepath = os.path.join(saveroot, f'{__now.strftime("%d%m%y_%H%M%S_%f")}_{title}') 

#     logging.info(f"[Before concat] Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
    
#     df = __embeddings_to_df(embeddings, labels, config_data,  vq_type='vqindhist1')
#     embeddings, label_data, unique_groups = __get_multiplexed_embeddings(df, random_state=config_data.SEED)
#     logging.info(f"[After concat] Embeddings shape: {embeddings.shape}, Labels shape: {label_data.shape}")
#     label_data = label_data.reshape(-1)
#     map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
#     if map_labels_function is not None:
#         logging.info("Applyging map_labels_function from the config on the unique_groups")
#         logging.info(f"unique groups before function: {unique_groups}")

#         map_labels_function = eval(map_labels_function)(config_data)
#         unique_groups = map_labels_function(unique_groups)    
#         logging.info(f"unique groups after function: {unique_groups}")

#         label_data = map_labels_function(label_data)

#     logging.info('[SM] computing umap')
#     umap_embeddings = compute_umap_embeddings(embeddings)
#     plot_umap_embeddings(umap_embeddings, label_data, config_data, savepath, show_ari=False,)
#                         #unique_groups=unique_groups)

# def compute_umap_embeddings(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
#     reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
#     umap_embeddings = reducer.fit_transform(embeddings)
#     return umap_embeddings


if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        generate_umaps()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
