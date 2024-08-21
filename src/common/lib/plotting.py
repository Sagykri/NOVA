import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import logging
import os
from src.common.lib.metrics import get_metrics_figure
from src.common.lib.utils import get_if_exists
import json

def plot_umap_embeddings(umap_embeddings, label_data, config_data, savepath = None,
                         title='UMAP projection of Embeddings', outliers_fraction=0.1,
                        dpi=300, figsize=(6,5), ordered_names=None, show_ari=True,
                        unique_groups=None, cell_line_cond_high = None):
    
    name_color_dict =  config_data.UMAP_MAPPINGS
    name_key=config_data.UMAP_MAPPINGS_ALIAS_KEY
    color_key=config_data.UMAP_MAPPINGS_COLOR_KEY
    s = config_data.SIZE
    alpha = config_data.ALPHA
    if unique_groups is None:
        unique_groups = np.unique(label_data)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2,1,height_ratios=[20,1])

    ax = fig.add_subplot(gs[0])
    for i, gp in enumerate(unique_groups):
        logging.info(f'adding {gp}')
        ind = label_data == gp
        ind = ind.reshape(-1,)
        if name_color_dict is not None:
            if cell_line_cond_high is not None:
                color=False
                for cl in cell_line_cond_high:
                    if cl in gp:
                        color=True
                if not color:
                    _c = np.array([*['gray'] * sum(ind)])
                if color:
                    _c = np.array([*[name_color_dict[gp][color_key]] * sum(ind)])
            else:
                _c = np.array([*[name_color_dict[gp][color_key]] * sum(ind)])
        else:
            _c = np.array([*[plt.get_cmap('tab20')(i)] * sum(ind)])
        ax.scatter(
            umap_embeddings[ind, 0],
            umap_embeddings[ind, 1],
            s=s,
            alpha=alpha,
            c=_c,
            marker = 'o',
            label=gp if name_color_dict is None else name_color_dict[gp][name_key],
        )
        logging.info(f'adding label {gp if name_color_dict is None else name_color_dict[gp][name_key]}')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    hndls, names = ax.get_legend_handles_labels()
    
    if ordered_names is not None:
        logging.info('ordering legend labels!')
        logging.info(f"names before: {names}")
        logging.info(f"hndls before : {hndls}")
        hndls = [handle for name, handle in sorted(zip(names, hndls), key=lambda x: ordered_names.index(x[0]))]
        names = ordered_names #sorted(ordered_names)

        logging.info(f"names after: {names}")
        logging.info(f"hndls after : {hndls}")

    leg = ax.legend(
    hndls,
    names,
    prop={'size': 6},
    bbox_to_anchor=(1, 1),
    loc='upper left',
    ncol=1 + len(names) // 25,
    #ncol=1, #NOAM: for umap1
    frameon=False,
)
    for ll in leg.legendHandles:
        ll.set_alpha(1)
        ll.set_sizes([max(6, s)]) # SAGY
    ax.set_xlabel('UMAP1') # Nancy for figure 2A - remove axis label - comment this out
    ax.set_ylabel('UMAP2') # Nancy for figure 2A - remove axis label - comment this out
    ax.set_title(title)
    
    ax.set_xticklabels([]) 
    ax.set_yticklabels([]) 
    ax.set_xticks([]) 
    ax.set_yticks([]) 
        
    if show_ari:
        gs_bottom = fig.add_subplot(gs[1])
        ax, scores = get_metrics_figure(umap_embeddings, label_data, ax=gs_bottom, outliers_fraction=outliers_fraction)
    fig.tight_layout()
    
    if savepath:
        __savepath_parent = os.path.dirname(savepath)
        if not os.path.exists(__savepath_parent):
            os.makedirs(__savepath_parent, exist_ok=True)

        logging.info(f"Saving umap to {savepath}")#SAGY
        # fig.savefig(f"{savepath}.eps", dpi=dpi, format='eps')
        fig.savefig(f"{savepath}.png", dpi=dpi)
        return
    plt.show()
    return

def plot_umap0(features, config_data, output_folder_path):
    umap_embeddings, labels = features[:,:2], features[:,2]
    markers = np.unique([m.split('_')[0] if '_' in m else m for m in np.unique(labels.reshape(-1,))]) 
    logging.info(f"Detected markers: {markers}")
    title = f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}_{'_'.join(config_data.REPS)}" #_{__now.strftime('%d%m%y_%H%M%S_%f')}"

    saveroot = output_folder_path
    if not os.path.exists(saveroot):
        os.makedirs(saveroot, exist_ok=True)
            
    with open(os.path.join(saveroot, 'config.json'), 'w') as json_file:
        json.dump(config_data.__dict__, json_file, indent=4)    
    
    for c in markers:
        logging.info(f"Marker: {c}")
        logging.info(f"[{c}] Selecting indexes of marker")
        c_indexes = np.where(np.char.startswith(labels.astype(str), f"{c}_"))[0]
        logging.info(f"[{c}] {len(c_indexes)} indexes have been selected")

        if len(c_indexes) == 0:
            logging.info(f"[{c}] Not exists in embedding. Skipping to the next one")
            continue

        umap_embeddings_c, labels_c = np.copy(umap_embeddings[c_indexes]), np.copy(labels[c_indexes].reshape(-1,))
        
        logging.info(f"[{c}] Plot umap...")
        savepath = os.path.join(saveroot, title, f'{c}') # NANCY
                
        map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
        if map_labels_function is not None:
            map_labels_function = eval(map_labels_function)(config_data)
        label_data = map_labels_function(labels_c) if map_labels_function is not None else labels_c
        if np.unique(label_data).shape[0]>10:
            show_ari=False
        else:
            show_ari=True
                
        cell_line_cond_high = get_if_exists(config_data, 'CELL_LINE_COND_HIGH', None)
        plot_umap_embeddings(umap_embeddings_c, label_data, config_data, 
                             savepath=savepath, show_ari=show_ari, title=c,
                             cell_line_cond_high=cell_line_cond_high)
        
        logging.info(f"[{c}] UMAP saved successfully to {savepath}")
        


def plot_umap1(features, config_data, output_folder_path):
    umap_embeddings, labels = features[:,:2], features[:,2]
    title = f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}_{'_'.join(config_data.REPS)}"
   
    saveroot = output_folder_path
    if not os.path.exists(saveroot):
        os.makedirs(saveroot, exist_ok=True)
        
    with open(os.path.join(saveroot, 'config.json'), 'w') as json_file:
        json.dump(config_data.__dict__, json_file, indent=4)
        
    savepath = os.path.join(saveroot, title)    
    map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
    if map_labels_function is not None:
        map_labels_function = eval(map_labels_function)(config_data)

    ordered_marker_names = get_if_exists(config_data, 'ORDERED_MARKER_NAMES', None)
    if ordered_marker_names is not None:
        ordered_names = [config_data.UMAP_MAPPINGS[marker]['alias'] for marker in ordered_marker_names]
    else:
        ordered_names = None
    
    label_data = map_labels_function(labels) if map_labels_function is not None else labels
    logging.info(f'label_data unique: {np.unique(label_data)}')
    plot_umap_embeddings(umap_embeddings, label_data, config_data, savepath, 
                        ordered_names = ordered_names, show_ari=False)
    logging.info(f"UMAP saved successfully to {savepath}")


def plot_umap2(features, config_data, output_folder_path):
    umap_embeddings, labels = features[:,:2], features[:,2]
    title = f"{'_'.join([os.path.basename(f) for f in config_data.INPUT_FOLDERS])}_{'_'.join(config_data.REPS)}"
    
    saveroot = os.path.join(output_folder_path)
    if not os.path.exists(saveroot):
        os.makedirs(saveroot, exist_ok=True)
        
    with open(os.path.join(saveroot, 'config.json'), 'w') as json_file:
        json.dump(config_data.__dict__, json_file, indent=4)    

    savepath = os.path.join(saveroot, title) 

    map_labels_function = get_if_exists(config_data, 'MAP_LABELS_FUNCTION', None)
    if map_labels_function is not None:
        map_labels_function = eval(map_labels_function)(config_data)
        label_data = map_labels_function(labels)

    plot_umap_embeddings(umap_embeddings, label_data, config_data, savepath, show_ari=False,)
                        #unique_groups=unique_groups)
    

def plot_marker_ranking():
    pass

# def plot_heatmap()

# def plot_bubble_plot()