"""
Version: 5
"""


import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.common.lib.model import Model

from figures.V5.configs.config import FigureV5Config
# from model import load_data, get_analytics
from src.common.lib.cytoself_custom import plot_umap, plot_feature_spectrum_from_image
from src.common.lib.synthetic_multiplexing import multiplex, plot_boostrapping, calc_bootstrapping
from src.common.lib.metrics import plot_metrics, calc_reconstruction_error
from src.common.lib.utils import get_colors_dict
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns

import os
import common.configs.model_config as model_config
# from src.common.configs.model_config import METRICS_FOLDER, METRICS_MATCH_PATH, METRICS_RANDOM_PATH, \
#     MARKERS, MICROGLIA_MARKERS, COMBINED_MARKERS

def __set_params(config:FigureV5Config):
    # TODO: Get rid of this function (and these global variables) and change the functions in this file to use the config vars directly instead

    input_folders = config.neuroself_config.INPUT_FOLDERS
    global input_folders_cytoself
    input_folders_cytoself = config.cytoself_config.INPUT_FOLDERS
    global input_folders_microglia
    input_folders_microglia = config.imgself_config.INPUT_FOLDERS
    global input_folders_combined 
    input_folders_combined = input_folders + input_folders_microglia
    
    global input_folders_pertrubations 
    input_folders_pertrubations = ["./data/processed/Perturbations"]
    global input_folders_pertrubations_spd 
    input_folders_pertrubations_spd = ["./data/processed/SpinningDisk/Perturbations"]
    global input_folders_pertrubations_spd2 
    input_folders_pertrubations_spd2 = ["./data/processed/spd2/SpinningDisk/Perturbations"]

    global cytoself_model_path
    cytoself_model_path = config.PRETRAINED_MODEL_PATH
    global neuroself_model_path 
    neuroself_model_path = config.neuroself_config.MODEL_PATH
    global imgself_model_path 
    imgself_model_path = config.imgself_config.MODEL_PATH
    global combined_model_path 
    combined_model_path = os.path.join("./model_outputs", "comb_model_trying", "model_ep0050_default_fixed.h5")
    global output_dir 
    output_dir = config.output_folder

    global groups_terms_condition
    groups_terms_condition = config.neuroself_config.GROUPS_TERMS_CONDITION
    global groups_terms_line 
    groups_terms_line = config.neuroself_config.GROUPS_TERMS_LINE
    global groups_terms_line_microglia 
    groups_terms_line_microglia = config.imgself_config.GROUPS_TERMS_LINE
    global groups_terms_type 
    groups_terms_type = ["_WT_microglia", "_WT_neurons", "_TDP43_microglia", "_TDP43_neurons",
                        "_FUS_microglia", "_FUS_neurons", "_OPTN_microglia", "_OPTN_neurons"]

    global self_config
    self_config = config
    
    global cytoself_config
    cytoself_config = config.cytoself_config
    
    global neuroself_config
    neuroself_config = config.neuroself_config
    
    global imgself_config
    imgself_config = config.imgself_config

def get_figures(config: FigureV5Config):
    """Get figures

    Args:
        figs ({fig_id: [panel_id1, panel_id2, ...], fig_id2: [..], ...}): Which figures (/panels) to return
    """
    
    __set_params(config)
    
    figs = config.figures
    
    for f in figs:
        panels = figs[f]
        figs_mapping[f](panels)


def __plot_umap_per_marker(images, labels, markers, model_path, fig_id='unknown', panel_id='unknown', savefig=False,figsize=None,cmap=None,
                           output_filename_prefix='', calc_metrics=True, is_comb=False, alpha=1.0, s=25, use_colors_dict=True):
    labels_s = pd.Series(labels.reshape(-1, ))
    for m in markers:
        logging.info(f"[{fig_id} {panel_id} {m}] Marker: {m}")
        if is_comb:
            reps = {"SCNA": "SNCA", "NCL": "Nucleolin", "phalloidin": "Phalloidin"}
            if m in ["SCNA", "NCL", "phalloidin"]:
                markers_indexes = labels_s[labels_s.str.contains('|'.join([m, reps[m]]), regex=True)].index
            elif m in ["SNCA", "Nucleolin", "Phalloidin"]:
                continue
            else:
                markers_indexes = labels_s[labels_s.str.contains(f"^{m}_", regex=True)].index
        else:
            markers_regex = '|'.join(list(map(lambda x: f"^{x}$|^{x}_", [m])))
            markers_indexes = labels_s[labels_s.str.contains(markers_regex, regex=True)].index

        images_subset = images[markers_indexes]
        labels_subset = labels[markers_indexes]
        colors_dict = get_colors_dict(labels_subset) if use_colors_dict else None

        analytics = get_analytics(images_subset, labels_subset, model_path=model_path)
        umap_vec, l1 = plot_umap(analytics, s=s, savefig=savefig, to_annot=False, titles=m, alpha=alpha,cmap=cmap,
                                 output_filename=f"{output_filename_prefix}{m}", colors_dict=colors_dict,figsize=figsize)
        if calc_metrics:
            logging.info("On UMAP vectors")
            plot_metrics(umap_vec[1], labels_subset, n_clusters=len(np.unique(labels_subset)))
            logging.info("On embedded vectors")
            plot_metrics(analytics.model.embvec[1].reshape(analytics.model.embvec[1].shape[0], -1), labels_subset,
                         n_clusters=len(np.unique(labels_subset)))

def __get_save_path(fig_id, panel_id):
    return os.path.join(output_dir, f"{fig_id}_{panel_id}")

def __run_panels(panels_mapping, panels_ids=None):
    # If None -> run all panels
    if panels_ids is None or panels_ids == []:
        for p in panels_mapping:
            p()
        return

    for p in panels_ids:
        panels_mapping[p]()


def __get_figure1(panels_ids=None):
    fig_id = "Fig1"

    logging.info(f"[{fig_id}] init")
    
    cytoself_model = Model(cytoself_config)

    def get_panel_c():
        panel_id = "C"

        markers = ['DAPI', 'Phalloidin', 'DCP1A', 'G3BP1']
        cytoself_model.markers = markers
        logging.info(f"[{fig_id} {panel_id}] Loading data")
        # images, labels, labels_changepoints, markers_order = load_data(input_folders_cytoself, condition_l=True,
        #                                                                markers=markers, type_l=False,
        #                                                                split_by_set=False)
        images, labels, labels_changepoints, markers_order = cytoself_model.load_with_datamanager()

        # Load data
        colors_dict = get_colors_dict(labels, cytoself_model.conf.COLORS_MAPPING)

        logging.info(f"[{fig_id} {panel_id}] Processing markers")
        for m in markers:
            logging.info(f"[{fig_id} {panel_id}] Processing markers - {m}")
            labels_s = pd.Series(labels.reshape(-1, ))
            markers_indexes = labels_s[labels_s.str.contains(m)].index
            images_subset = images[markers_indexes]
            labels_subset = labels[markers_indexes]

            # Load model
            logging.info(f"[{fig_id} {panel_id}] Processing markers - {m} (get analytics_cytoself)")
            # analytics_cytoself = get_analytics(images_subset, labels_subset, model_path=cytoself_model_path)
            analytics_cytoself = cytoself_model.load_analytics()

            savepath = __get_save_path(fig_id, panel_id)
            logging.info(f"[{fig_id} {panel_id}] Processing markers - {m} (plot umap)")
            umap_vec, _ = plot_umap(analytics_cytoself, colors_dict=colors_dict,
                                    to_annot=False, savepath=savepath+f"_{m}_umap.pdf")

            logging.info(f"[{fig_id} {panel_id}] Processing markers - {m} (plot mertrics)")

            logging.info("On UMAP vectors")
            plot_metrics(umap_vec[1], labels_subset, n_clusters=len(groups_terms_condition), savepath=savepath+f"_{m}_metrics_umap.pdf")

            logging.info("On embedded vectors")
            plot_metrics(analytics_cytoself.model.embvec[1].reshape(analytics_cytoself.model.embvec[1].shape[0], -1),
                         labels_subset,
                         n_clusters=len(groups_terms_condition),
                         savepath=savepath+f"_{m}_metrics_embvec.pdf")

    def get_panel_e_and_f():
        panel_id = "E+F"
        logging.info(f"[{fig_id} {panel_id}] init")

        markers = ["Phalloidin", "DCP1A"]
        cytoself_model.markers = markers
        # Load data
        logging.info(f"[{fig_id} {panel_id}] Loading data")
        # images, labels, labels_changepoints, markers_order = load_data(input_folders_cytoself, markers=markers,
        #                                                                condition_l=True, type_l=False,
        #                                                                split_by_set=False)
        images, labels, labels_changepoints, markers_order = cytoself_model.load_with_datamanager()


        # Load model
        logging.info(f"[{fig_id} {panel_id}] get analytics_cytoself")
        # analytics_cytoself = get_analytics(images, labels, model_path=cytoself_model_path)
        cytoself_model.load_analytics()


        # E - Synthetic multiplexing
        panel_id = "E"
        savepath = __get_save_path(fig_id, panel_id)
        logging.info(f"[{fig_id} {panel_id}] init")
        logging.info(f"[{fig_id} {panel_id}] Calculating synthetic multiplexing")
        _, _ = multiplex(cytoself_model, images, labels, \
                         groups_terms=groups_terms_condition, markers=markers, \
                         markers_order=markers_order, annot_font_size1=15, match=False, legend_inside=False,
                         savepath1 = savepath + "_umap1.pdf",
                         savepath1 = savepath + "_umap2.pdf")

        # F - Actual Multiplexing
        panel_id = "F"
        savepath = __get_save_path(fig_id, panel_id)
        logging.info(f"[{fig_id} {panel_id}] init")
        logging.info(f"[{fig_id} {panel_id}] Calculating actual multiplexing")
        _, _ = multiplex(cytoself_model, images, labels, \
                         groups_terms=groups_terms_condition, markers=markers, \
                         labels_changepoints=labels_changepoints, \
                         markers_order=markers_order, annot_font_size1=15, match=True, legend_inside=False,
                         savepath1 = savepath + "_umap1.pdf",
                         savepath1 = savepath + "_umap2.pdf")

    # TODO: TBD
    def get_panel_g():
        raise NotImplementedError()
        panel_id = "G"
        markers = ["Phalloidin", "DCP1A"]
        cytoself_model.markers = markers
        # Load data
        logging.info(f"[{fig_id} {panel_id}] Loading data")
        # images, labels, labels_changepoints, markers_order = load_data(input_folders_cytoself, markers=markers,
        #                                                                condition_l=True, type_l=False,
        #                                                                split_by_set=False)
        images, labels, labels_changepoints, markers_order = cytoself_model.load_with_datamanager()


        # logging.info(f"[{fig_id} {panel_id}] init")
        # markers = np.unique([m[:m.find("_")] for m in labels])

        # Load model
        logging.info(f"[{fig_id} {panel_id}] get analytics_cytoself")
        # analytics_cytoself = get_analytics(images, labels, model_path=cytoself_model_path)
        analytics_cytoself = cytoself_model.load_analytics()


        logging.info(f"[{fig_id} {panel_id}] Calculating ARI (bootstrapping)")
        calc_bootstrapping(analytics_cytoself, images, labels, markers=markers,
                           markers_order=markers_order, groups_terms=groups_terms_condition,
                           save_folder=METRICS_FOLDER)

        logging.info(f"[{fig_id} {panel_id}] Plotting ARI (bootstrapping)")
        plot_boostrapping(metrics_random_path=METRICS_RANDOM_PATH, metrics_match_path=METRICS_MATCH_PATH)

    def get_panel_h():
        panel_id = "H"
        # markers = ["Phalloidin", "DCP1A"]
        markers = ['DAPI', 'Phalloidin', 'DCP1A', 'G3BP1']
        cytoself_model.markers = markers
        # Load data
        logging.info(f"[{fig_id} {panel_id}] Loading data")
        # images, labels, labels_changepoints, markers_order = load_data(input_folders_cytoself, markers=markers,
        #                                                                condition_l=True, type_l=False,
        #                                                                split_by_set=False)
        images, labels, labels_changepoints, markers_order = cytoself_model.load_with_datamanager()


        logging.info(f"[{fig_id} {panel_id}] Processing markers")

        # Load model
        logging.info(f"[{fig_id} {panel_id}] Processing markers - (get analytics_cytoself)")
        # analytics_cytoself = get_analytics(images, labels, model_path=cytoself_model_path)
        analytics_cytoself = cytoself_model.load_analytics()

        # 1D (Feature spectrum)
        logging.info(f"[{fig_id} {panel_id}] Processing markers - (feature spectrum)")
        analytics_cytoself.plot_clustermaps()

        labels_s = pd.Series(labels.reshape(-1, ))
        for obj in [('unstressed', model_config.COLORS_MAPPING[model_config.TERM_UNSTRESSED]),
                    ('stressed', model_config.COLORS_MAPPING[model_config.TERM_STRESSED])]:
            cond, color = obj[0], obj[1]
            logging.info(f"[{fig_id} {panel_id}] Processing markers - (feature spectrum; {cond})")

            images_subset_cond = images[labels_s[labels_s.str.contains(f"_{cond}")].index]

            plot_feature_spectrum_from_image(
                analytics=analytics_cytoself,
                data=images_subset_cond,
                savepath=output_dir,
                filename=f"{fig_id}_{panel_id}_ALL_{cond}_Feature_spectrum",
                color=color,
                figsize=(15, 1),
            )

    panels_mapping = {"c": get_panel_c, 'e+f': get_panel_e_and_f,
                      "g": get_panel_g, 'h': get_panel_h}

    __run_panels(panels_mapping, panels_ids)


def __get_figure2(panels_ids=None):
    fig_id = "Fig2"

    logging.info(f"[{fig_id}] init")

    logging.info(f"[{fig_id}] Loading data")
    images, labels, labels_changepoints, markers_order = load_data(input_folders, condition_l=True, type_l=False,
                                                                   cell_lines_include=["WT"],
                                                                   split_by_set=False,
                                                                   #    set_type="test",
                                                                   markers=MARKERS,
                                                                   split_by_set_include=[("WT", "unstressed")])

    markers = ["G3BP1", "TIA1", "DAPI"]

    def get_panel_b():
        panel_id = 'B'

        logging.info(f"[{fig_id} [{panel_id}]] Plot umap")
        __plot_umap_per_marker(images, labels, markers, cytoself_model_path,
                               fig_id=fig_id, panel_id=panel_id,
                               savefig=True, output_filename_prefix=f"MODEL18_{fig_id}_{panel_id}_",
                               calc_metrics=True)

    def get_panel_c():
        panel_id = 'C'

        logging.info(f"[{fig_id} [{panel_id}]] Plot umap")
        __plot_umap_per_marker(images, labels, markers, neuroself_model_path,
                               fig_id=fig_id, panel_id=panel_id,
                               savefig=True, output_filename_prefix=f"MODEL18_{fig_id}_{panel_id}_",
                               calc_metrics=True)

    def get_panel_d():
        panel_id = "D"
        n_images = 20
        # Take only specific (selected) images 
        #marker_image_mapping = {"G3BP1": 3, "TIA1": 3, "DAPI": 1}
        logging.info(f"[{fig_id} {panel_id}] init")

        labels_s = pd.Series(labels.reshape(-1, ))

        for m in markers:
            logging.info(f"[{fig_id} {panel_id} {m}] Marker: {m}")
            markers_regex = '|'.join(list(map(lambda x: f"^{x}$|^{x}_", [m])))
            markers_indexes = labels_s[labels_s.str.contains(markers_regex)].index
            markers_indexes_subset = markers_indexes[:min(n_images, len(markers_indexes))]
            #markers_indexes_subset = np.random.choice(markers_indexes,
            #                                          size=min(n_images, len(markers_indexes)),
            #                                          replace=False)
            # Take only specific (selected) images                                          
            #markers_indexes_subset = [markers_indexes_subset[marker_image_mapping[m]]]
            images_subset = images[markers_indexes_subset]
            labels_subset = labels[markers_indexes_subset]

            # Load models
            logging.info(f"[{fig_id} {panel_id} {m}] get analytics_cytoself")
            analytics_cytoself = get_analytics(images_subset, labels_subset, model_path=cytoself_model_path)
            logging.info(f"[{fig_id} {panel_id} {m}] get analytics_neuroself")
            analytics_neuroself = get_analytics(images_subset, labels_subset, model_path=neuroself_model_path)

            # Generate reconsturcted images and calculate reconstruction error (MSE)
            for i in range(len(images_subset)):
                logging.info(f"[{fig_id} {panel_id} {m}] Generate reconstructed images (cytoself)")
                rec_error = calc_reconstruction_error(analytics_cytoself, images_indexes=[i], show=True,
                                                      entire_data=images, 
                                                      cmap_original = 'Greys_r', cmap_reconstructed = 'rainbow', enhance_contrast=False,
                                                      embvecs=analytics_cytoself.model.embvec, reset_embvec=False)
                logging.info(f"[{fig_id} {panel_id} {m} (cytoself)] {rec_error}")

                logging.info(f"[{fig_id} {panel_id} {m}] Generate reconstructed images (neuroself)")
                rec_error = calc_reconstruction_error(analytics_neuroself, images_indexes=[i], show=True,
                                                      entire_data=images,
                                                      cmap_original = 'Greys_r', cmap_reconstructed = 'rainbow', enhance_contrast=False,
                                                      embvecs=analytics_neuroself.model.embvec, reset_embvec=False)
                logging.info(f"[{fig_id} {panel_id} {m} (neuroself)] {rec_error}")

            ##########################

    # Map and run the panels
    panels_mapping = {"b": get_panel_b, 'c': get_panel_c, 'd': get_panel_d}

    __run_panels(panels_mapping, panels_ids)
    ############################


def __get_figure3(panels_ids=None):
    fig_id = "Fig3"

    logging.info(f"[{fig_id}] init")

    logging.info(f"[{fig_id}] Loading data")
    images, labels, labels_changepoints, markers_order = load_data(input_folders, condition_l=True, type_l=False,
                                                                   cell_lines_include=["WT"],
                                                                   split_by_set=False,
                                                                   # set_type="test",
                                                                   markers=MARKERS,
                                                                   split_by_set_include=[("WT", "unstressed")])

    # def get_panel_b_old():
    #     panel_id = "B"

    #     logging.info(f"[{fig_id} {panel_id}] init")

    #     logging.info(f"[{fig_id} {panel_id}] get analytics_neuroself")
    #     analytics_neuroself = get_analytics(images, labels, model_path=neuroself_model_path)

    #     logging.info(f"[{fig_id} {panel_id}] Calculating embvec")
    #     analytics_neuroself.model.calc_embvec(analytics_neuroself.data_manager.test_data)
    #     X = analytics_neuroself.model.embvec[1]
    #     X = X.reshape(X.shape[0], -1)
    #     logging.info(X.shape)

    #     # Setting clusters_centers as the median of all images of specific marker+cond
    #     logging.info(f"[{fig_id} {panel_id}] Calculating the median of each set of marker+cond combination")
    #     labels_s = pd.Series(labels.reshape(-1, ))

    #     labels_unique = labels_s.unique()
    #     clusters_median = np.zeros((len(labels_unique), X.shape[1]))
    #     for j, l in enumerate(labels_unique):
    #         ii = labels_s[labels_s.str.startswith(l)].index.tolist()
    #         current_x = X[ii, :].copy()
    #         clusters_median[j] = np.median(current_x, axis=0)

    #     # Calc dists
    #     logging.info(f"[{fig_id} {panel_id}] Calculating euclidean distances")
    #     dists = euclidean_distances(clusters_median)
    #     dists_df = pd.DataFrame(dists, index=labels_unique, columns=labels_unique)

    #     # Take only dists between the same marker in different conditions
    #     logging.info(f"[{fig_id} {panel_id}] Taking only the distances between conditions for each marker")
    #     dists_df_copy = dists_df.copy()
    #     markers = np.unique([m.split("_")[0] for m in dists_df_copy.index.to_list()])
    #     distances_for_marker = {m: 0 for m in markers}
    #     dists_df_copy = dists_df_copy.loc[
    #         dists_df_copy.columns.str.contains(config.TERM_UNSTRESSED), dists_df_copy.columns.str.contains(
    #             config.TERM_STRESSED)]

    #     for ind in dists_df_copy.index:
    #         m, cond = ind.split('_')
    #         logging.info(m, cond)
    #         cols_with_current_marker = dists_df_copy.columns.str.contains(f"{m}_")
    #         distances_for_marker[m] = dists_df_copy.loc[ind, cols_with_current_marker]

    #     df_ranking = pd.DataFrame.from_dict(distances_for_marker, orient='index')
    #     df_ranking_indexes = df_ranking.index.tolist()
    #     df_ranking = df_ranking.to_numpy().diagonal()
    #     df_ranking = pd.Series(df_ranking, index=df_ranking_indexes)
    #     df_ranking = df_ranking.sort_values(ascending=False)

    #     df_ranking.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_ranking_conds_based.csv"))
    #     logging.info(f"[{fig_id} {panel_id}] {df_ranking}")

    def get_panel_b():
        panel_id = 'B'

        logging.info(f"[{fig_id} {panel_id}] Plot UMAPs")

        __plot_umap_per_marker(images, labels, MARKERS, neuroself_model_path, fig_id=fig_id,
                               panel_id=panel_id, savefig=True, output_filename_prefix=f'MODEL18_{fig_id}_{panel_id}_',
                               calc_metrics=True)

        # logging.info(f"[{fig_id} {panel_id}] Arrange UMAPs")
        # arrange_plots('./umaps', nrows=5, ncols=6, file_name_contains='MODEL18_')

    def get_panel_c():
        panel_id = 'C'

        logging.info(f"[{fig_id} {panel_id}] Calc distances")
        logging.info(f"[{fig_id} {panel_id}] !!!! CHANGED from neuroelf to the combined model!")
        dists = __calc_dist_conds(fig_id, panel_id, images, labels, combined_model_path)
        logging.info(f"[{fig_id} {panel_id}] {dists}")
        dists.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_combined_rankings.csv"))

        return dists

    def get_panel_d():
        panel_id = "D"

        # logging.info(f"[{fig_id}] Loading data")
        # images, labels, labels_changepoints, markers_order = load_data(input_folders, condition_l=True, type_l=False,
        #                                                               cell_lines_include=["WT"],
        #                                                               split_by_set=False,
        #                                                               markers=MARKERS,
        #                                                               split_by_set_include=[("WT", "unstressed")])

        # Load models
        logging.info(f"[{fig_id} {panel_id}] get analytics_neuroself")
        analytics_neuroself = get_analytics(images, labels, model_path=neuroself_model_path)

        # Plot UMAPs
        logging.info(f"[{fig_id} {panel_id}] Plot multiplexed UMAPs")
        colors_dict = get_colors_dict(labels)
        _, _ = multiplex(analytics_neuroself, images, labels, \
                         groups_terms=groups_terms_condition, markers=MARKERS, \
                         # s1=5,
                         # s2=20,
                         colors_dict1=colors_dict,
                         markers_order=markers_order,
                         annot_font_size1=9,
                         show1=False,
                         savefig=True)

    ##########################
    # Map and run the panels
    panels_mapping = {"b": get_panel_b, 'c': get_panel_c, 'd': get_panel_d}

    __run_panels(panels_mapping, panels_ids)
    ############################


def __get_figure4(panels_ids=None):
    raise NotImplemented


def __get_figure4_old(panels_ids=None):
    fig_id = "Fig4"

    logging.info(f"[{fig_id}] Init")

    # Load data
    logging.info(f"[{fig_id}] Load data")
    images, labels, labels_changepoints, markers_order = load_data(input_folders, markers=MARKERS, \
                                                                   condition_l=False, type_l=True, \
                                                                   split_by_set=True,
                                                                   set_type='test',
                                                                   conds_include=["unstressed"],
                                                                   split_by_set_include=[("WT", "unstressed")])

    # def get_panel_a_d():
    #     panel_id = "A+D"
    #     logging.info(f"[{fig_id} {panel_id}] init")

    #     # Load models
    #     logging.info(f"[{fig_id} {panel_id}] get analytics_neuroself")
    #     analytics_neuroself = get_analytics(images, labels, model_path=neuroself_model_path)

    #     # Plot UMAPs
    #     logging.info(f"[{fig_id} {panel_id}] Plot multiplexed UMAPs")
    #     _, _ = multiplex(analytics_neuroself, images, labels,\
    #               filename="unstressed_4", groups_terms=groups_terms_line, markers=MARKERS,\
    #               first_umap_figsize=(15,15),
    #               s1=5, cmap1=["tab20", "tab20b", "tab20c", "tab10", "Set1", "Set2", "Set3", "Accent", "Dark2", "Pastel1", "Pastel2", "Paired", "tab20", "tab20b", "tab20c", "tab10", "Set1", "Set2", "Set3", "Accent", "Dark2", "Pastel1", "Pastel2", "Paired"],
    #               markers_order=markers_order, savefig=False, annot_font_size1=9)

    def get_panel_b():
        panel_id = "B"

        logging.info(f"[{fig_id} {panel_id}] get analytics_neuroself")
        analytics_neuroself = get_analytics(images, labels, model_path=neuroself_model_path)
        logging.info(f"[{fig_id} {panel_id}] Calculating embvec")
        analytics_neuroself.model.calc_embvec(analytics_neuroself.data_manager.test_data)

        logging.info(f"[{fig_id} {panel_id}] Init")
        X = analytics_neuroself.model.embvec[1]
        X = X.reshape(X.shape[0], -1)
        logging.info(X.shape)

        # Setting clusters_centers as the median of all images of specific marker+cond
        logging.info(f"[{fig_id} {panel_id}] Calculating the median of each set of marker+cond combination")
        labels_s = pd.Series(labels.reshape(-1, ))

        labels_unique = labels_s.unique()
        clusters_median = np.zeros((len(labels_unique), X.shape[1]))
        for j, l in enumerate(labels_unique):
            ii = labels_s[labels_s.str.startswith(l)].index.tolist()
            current_x = X[ii, :].copy()
            clusters_median[j] = np.median(current_x, axis=0)

        # Calc dists
        logging.info(f"[{fig_id} {panel_id}] Calculating euclidean distances")
        dists = euclidean_distances(clusters_median)
        dists_df = pd.DataFrame(dists, index=labels_unique, columns=labels_unique)

        logging.info(f"[{fig_id} {panel_id}] Normalize values (min-max)")
        dists_df_cols, dists_df_ind = dists_df.columns, dists_df.index
        # To numpy for scaling accross all columns!
        dists_df_np = dists_df.to_numpy()
        dists_df_np_scaled = (dists_df_np - dists_df_np.min()) / (dists_df_np.max() - dists_df_np.min())
        # Back to dataframe after scaling
        dists_df_scaled = pd.DataFrame(dists_df_np_scaled, columns=dists_df_cols, index=dists_df_ind)

        logging.info(f"[{fig_id} {panel_id}] Taking only the distances between cell lines for each marker")
        markers = np.unique([m.split("_")[0] for m in dists_df.index.to_list()])

        plt.figure(figsize=(20, 20))
        sns.clustermap(dists_df_scaled, yticklabels=True, xticklabels=True)
        plt.show()

        for m in markers:
            logging.info(f"[{fig_id} {panel_id}] {m}")
            cols_with_current_marker = dists_df_scaled.columns.str.contains(f"{m}_")
            d = dists_df_scaled.loc[cols_with_current_marker, cols_with_current_marker]
            d = d.sort_index(axis=0).sort_index(axis=1)
            sns.heatmap(d)
            plt.title(m)
            plt.show()

    def get_panel_b2():
        panel_id = "B"

        logging.info(f"[{fig_id} {panel_id}] Init")

        logging.info(f"[{fig_id} {panel_id}] get analytics_neuroself")
        analytics_neuroself = get_analytics(images, labels, model_path=neuroself_model_path)
        logging.info(f"[{fig_id} {panel_id}] Calculating embvec")
        analytics_neuroself.model.calc_embvec(analytics_neuroself.data_manager.test_data)

        X = analytics_neuroself.model.embvec[1]
        X = X.reshape(X.shape[0], -1)
        logging.info(X.shape)

        # Setting clusters_centers as the median of all images of specific marker+cond
        logging.info(f"[{fig_id} {panel_id}] Calculating the median of each set of marker+cond combination")
        labels_s = pd.Series(labels.reshape(-1, ))

        labels_unique = labels_s.unique()
        clusters_median = np.zeros((len(labels_unique), X.shape[1]))
        for j, l in enumerate(labels_unique):
            ii = labels_s[labels_s.str.startswith(l)].index.tolist()
            current_x = X[ii, :].copy()
            clusters_median[j] = np.median(current_x, axis=0)

        # Calc dists
        logging.info(f"[{fig_id} {panel_id}] Calculating euclidean distances")
        dists = euclidean_distances(clusters_median)
        dists_df = pd.DataFrame(dists, index=labels_unique, columns=labels_unique)

        logging.info(f"[{fig_id} {panel_id}] Taking only the distances between cell lines for each marker")
        dists_df_copy = dists_df.copy()
        markers = np.unique([m.split("_")[0] for m in dists_df_copy.index.to_list()])
        distances_for_marker_WT = {m: None for m in markers}
        distances_for_marker_ALS = distances_for_marker_WT.copy()

        # ind = np.unique([t.split('_')[0] for t in dists_df_copy.index.to_list()])
        # col = np.unique([t.split('_')[1] for t in dists_df_copy.columns.to_list()])
        # distances_for_marker_ALS_sep = pd.DataFrame(index=ind, columns=col)

        for ind in dists_df_copy.index:
            m, cell_line = ind.split('_')
            logging.info(f"[{fig_id} {panel_id}] {m} {cell_line}")
            cols_with_current_cellline = dists_df_copy.columns.str.contains(f"{m}_{cell_line}")
            cols_with_current_marker = dists_df_copy.columns.str.contains(f"{m}_")
            if cell_line == "WT":
                vals = dists_df_copy.loc[cols_with_current_marker & cols_with_current_cellline,
                                         cols_with_current_marker & ~cols_with_current_cellline]
                distances_for_marker_WT[m] = vals.mean(axis=1).values[0]
            else:
                if distances_for_marker_ALS[m] is not None:
                    continue
                cols_with_WT = dists_df_copy.columns.str.contains(f"{m}_WT")
                vals = dists_df_copy.loc[cols_with_current_marker & ~cols_with_WT,
                                         cols_with_current_marker & ~cols_with_WT]
                distances_for_marker_ALS[m] = np.sum(np.triu(vals.to_numpy())) * 1.0 / (vals.shape[0])
                # distances_for_marker_ALS_sep.loc[m, cell_line] = dists_df_copy.loc[cols_with_WT, cols_with_current_cellline]

        df_WT = pd.DataFrame.from_dict(distances_for_marker_WT, orient='index')
        df_ALS = pd.DataFrame.from_dict(distances_for_marker_ALS, orient='index')

        merged = pd.merge(df_WT, df_ALS, left_index=True, right_index=True)
        merged.columns = ["WT", "ALS"]

        # merged = pd.merge(merged, distances_for_marker_ALS_sep, left_index=True, right_index=True)
        # merged.columns = merged.columns + distances_for_marker_ALS_sep.columns

        logging.info(f"[{fig_id} {panel_id}] Sort values")
        merged.sort_values(by=["WT", "ALS"], ascending=[False, True])

        # Plot
        logging.info(f"[{fig_id} {panel_id}] Normalize values (min-max)")
        merged_norm = pd.DataFrame(
            [(merged[col] - merged[col].min()) / (merged[col].max() - merged[col].min()) for col in merged.columns]).T

        logging.info(f"[{fig_id} {panel_id}] Plot (combined)")
        plt.scatter(merged_norm.loc[:, 'WT'], merged_norm.loc[:, 'ALS'])
        for i in range(merged_norm.shape[0]):
            plt.annotate(merged_norm.index[i], (merged_norm.iloc[i]['WT'], merged_norm.iloc[i]['ALS']))
        plt.xlabel('WT')
        plt.ylabel('ALS')
        plt.show()

    ##########################
    # Map and run the panels
    panels_mapping = {'b': get_panel_b, 'b2': get_panel_b2}

    __run_panels(panels_mapping, panels_ids)
    ############################


def __get_figure5(panels_ids=None):
    fig_id = "Fig5"

    logging.info(f"[{fig_id}] Init")

    __get_figure_5_6(fig_id, input_folders, MARKERS, neuroself_model_path, groups_terms_line,
                     model_name="Neuroself", panels_ids=panels_ids)


def __get_figure6(panels_ids=None):
    fig_id = "Fig6"

    logging.info(f"[{fig_id}] Init")
    __get_figure_5_6(fig_id, input_folders=input_folders_microglia,
                     markers=MICROGLIA_MARKERS,
                     model_path=imgself_model_path,
                     groups_terms_line=groups_terms_line_microglia,
                     model_name="Imgself", panels_ids=panels_ids)


def __calc_dist_conds(fig_id, panel_id, images, labels, model_path):
    logging.info(f"[{fig_id} {panel_id}] get analytics_neuroself")
    analytics_neuroself = get_analytics(images, labels, model_path=model_path)
    logging.info(f"[{fig_id} {panel_id}] Calculating embvec")
    analytics_neuroself.model.calc_embvec(analytics_neuroself.data_manager.test_data)

    logging.info(f"[{fig_id} {panel_id}] Init")
    X = analytics_neuroself.model.embvec[1]
    X = X.reshape(X.shape[0], -1)
    logging.info(X.shape)

    # Setting clusters_centers as the median of all images of specific marker+cond
    logging.info(f"[{fig_id} {panel_id}] Calculating the median of each set of marker+cond combination")
    labels_s = pd.Series(labels.reshape(-1, ))

    labels_unique = labels_s.unique()
    clusters_median = np.zeros((len(labels_unique), X.shape[1]))
    for j, l in enumerate(labels_unique):
        ii = labels_s[labels_s.str.startswith(l)].index.tolist()
        current_x = X[ii, :].copy()
        clusters_median[j] = np.median(current_x, axis=0)

    # Calc dists
    logging.info(f"[{fig_id} {panel_id}] Calculating euclidean distances")
    dists = euclidean_distances(clusters_median)
    dists_df = pd.DataFrame(dists, index=labels_unique, columns=labels_unique)

    logging.info(f"[{fig_id} {panel_id}] Saving euclidean distances before normalization")
    dists_df.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_Neuroself_dists_before_norm.csv"))

    unstressed_ind = np.where(dists_df.index.str.contains(f"_unstressed"))[0]
    stressed_ind = np.where(dists_df.columns.str.contains(f"_stressed"))[0]

    dists_df = dists_df.iloc[unstressed_ind, stressed_ind]
    markers = [m.split("_")[0] for m in dists_df.index.to_list()]
    dists_diag = [dists_df.loc[f"{m}_unstressed", f"{m}_stressed"] for m in markers]

    # Normalize (min max)
    logging.info(f"[{fig_id} {panel_id}] Normalize values (min-max)")
    dists_diag = (dists_diag - min(dists_diag)) / (max(dists_diag) - min(dists_diag))
    rankings_norm = pd.Series(dists_diag, index=markers).sort_values(ascending=False)

    return rankings_norm


def __calc_dist(fig_id, panel_id, images, labels, model_path, model_name):
    logging.info(f"[{fig_id} {panel_id}] get analytics_neuroself")
    analytics_neuroself = get_analytics(images, labels, model_path=model_path)
    logging.info(f"[{fig_id} {panel_id}] Calculating embvec")
    analytics_neuroself.model.calc_embvec(analytics_neuroself.data_manager.test_data)

    logging.info(f"[{fig_id} {panel_id}] Init")
    X = analytics_neuroself.model.embvec[1]
    X = X.reshape(X.shape[0], -1)
    logging.info(X.shape)

    # Setting clusters_centers as the median of all images of specific marker+cond
    logging.info(f"[{fig_id} {panel_id}] Calculating the median of each set of marker+cond combination")
    labels_s = pd.Series(labels.reshape(-1, ))

    labels_unique = labels_s.unique()
    clusters_median = np.zeros((len(labels_unique), X.shape[1]))
    for j, l in enumerate(labels_unique):
        ii = labels_s[labels_s.str.startswith(l)].index.tolist()
        current_x = X[ii, :].copy()
        clusters_median[j] = np.median(current_x, axis=0)

    # Calc dists
    logging.info(f"[{fig_id} {panel_id}] Calculating euclidean distances")
    dists = euclidean_distances(clusters_median)
    dists_df = pd.DataFrame(dists, index=labels_unique, columns=labels_unique)
    markers = np.unique([m.split("_")[0] for m in dists_df.index.to_list()])

    dists_df.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_dists_before_norm.csv"))

    logging.info(f"[{fig_id} {panel_id}] Normalize values (min-max)")
    dists_df_cols, dists_df_ind = dists_df.columns, dists_df.index
    dists_df_scaled = dists_df.copy()
    indexes = {}
    for m in markers:
        indexes[m] = np.where(dists_df_scaled.columns.str.startswith(f"{m}_"))[0]
    for m in indexes:
        markers_indexes = indexes[m]
        subset = dists_df_scaled.iloc[markers_indexes, markers_indexes]
        # min().min() because we taking the minimum of the entire dataframe (all columns/rows)
        dists_df_scaled.iloc[markers_indexes, markers_indexes] = (subset - subset.min().min()) / (
                subset.max().max() - subset.min().min())

    return dists_df_scaled


def __calc_dist_combined(fig_id, panel_id, images, labels, model_path, model_name):
    logging.info(f"[{fig_id} {panel_id}] get analytics_neuroself")
    analytics_neuroself = get_analytics(images, labels, model_path=model_path)
    logging.info(f"[{fig_id} {panel_id}] Calculating embvec")
    analytics_neuroself.model.calc_embvec(analytics_neuroself.data_manager.test_data)

    logging.info(f"[{fig_id} {panel_id}] Init")
    X = analytics_neuroself.model.embvec[1]
    X = X.reshape(X.shape[0], -1)
    logging.info(X.shape)

    # Setting clusters_centers as the median of all images of specific marker+cond
    logging.info(f"[{fig_id} {panel_id}] Calculating the median of each set of marker+cond combination")
    labels_s = pd.Series(labels.reshape(-1, ))

    labels_unique = labels_s.unique()
    clusters_median = np.zeros((len(labels_unique), X.shape[1]))
    for j, l in enumerate(labels_unique):
        ii = labels_s[labels_s.str.startswith(l)].index.tolist()
        current_x = X[ii, :].copy()
        clusters_median[j] = np.median(current_x, axis=0)

    # Calc dists
    logging.info(f"[{fig_id} {panel_id}] Calculating euclidean distances")
    dists = euclidean_distances(clusters_median)
    dists_df = pd.DataFrame(dists, index=labels_unique, columns=labels_unique)
    markers = np.unique([m.split("_")[0] for m in dists_df.index.to_list()])

    dists_df.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_dists_before_norm.csv"))

    logging.info(f"[{fig_id} {panel_id}] Normalize values (min-max)")
    dists_df_cols, dists_df_ind = dists_df.columns, dists_df.index
    dists_df_scaled_microglia = dists_df.iloc[np.where(dists_df.columns.str.contains(f"microglia"))[0],
                                              np.where(dists_df.columns.str.contains(f"microglia"))[0]]
    dists_df_scaled_neurons = dists_df.iloc[np.where(dists_df.columns.str.contains(f"neurons"))[0],
                                            np.where(dists_df.columns.str.contains(f"neurons"))[0]]
    indexes_microglia, indexes_neurons = {}, {}
    for m in markers:
        indexes_microglia[m] = np.where(dists_df_scaled_microglia.columns.str.startswith(f"{m}_"))[0]
        indexes_neurons[m] = np.where(dists_df_scaled_neurons.columns.str.startswith(f"{m}_"))[0]
    for m in indexes_microglia:
        markers_indexes_microglia = indexes_microglia[m]
        subset = dists_df_scaled_microglia.iloc[markers_indexes_microglia, markers_indexes_microglia]
        # min().min() because we taking the minimum of the entire dataframe (all columns/rows)
        dists_df_scaled_microglia.iloc[markers_indexes_microglia, markers_indexes_microglia] = \
            (subset - subset.min().min()) / (subset.max().max() - subset.min().min())
    for m in indexes_neurons:
        markers_indexes_neurons = indexes_neurons[m]
        subset = dists_df_scaled_neurons.iloc[markers_indexes_neurons, markers_indexes_neurons]
        # min().min() because we taking the minimum of the entire dataframe (all columns/rows)
        dists_df_scaled_neurons.iloc[markers_indexes_neurons, markers_indexes_neurons] = \
            (subset - subset.min().min()) / (subset.max().max() - subset.min().min())

    return dists_df_scaled_microglia, dists_df_scaled_neurons


def __heatmap_dists(fig_id, panel_id, dists):
    markers = np.unique([m.split("_")[0] for m in dists.index.to_list()])

    plt.figure(figsize=(20, 20))
    sns.clustermap(dists, yticklabels=True, xticklabels=True)
    plt.show()

    import re
    extract_cell_line_name = lambda x: re.sub(".*_", "", x)

    for m in markers:
        logging.info(f"[{fig_id} {panel_id}] {m}")
        cols_with_current_marker = dists.columns.str.startswith(f"{m}_")
        d = dists.loc[cols_with_current_marker, cols_with_current_marker]
        d = d.sort_index(axis=0).sort_index(axis=1)
        d = d.rename(columns=extract_cell_line_name, index=extract_cell_line_name)
        sns.heatmap(d)
        plt.title(m)
        plt.savefig(f"./output/{fig_id}_{panel_id}_{m}_heatmap_dists.pdf", bbox_inches='tight')
        plt.show()


def __get_markers_ranking_per_cell_line(fig_id, panel_id, dists, lines, model_name=""):
    markers = np.unique([m.split("_")[0] for m in dists.index.to_list()])

    wt_hits = dists.index.str.contains(model_config.TERM_WT)
    for line in lines:
        line_hits = dists.columns.str.contains(line)
        rankings = dists.loc[wt_hits, line_hits]
        rankings = rankings.loc[sorted(rankings.index), sorted(rankings.columns)]
        rankings = pd.Series(np.diag(rankings), index=rankings.index).sort_values(ascending=False)
        rankings.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_ranking_{line}.csv"), index=True)
        logging.info(line)
        logging.info(rankings)
        logging.info('----')


def __plot_markers_ranking(fig_id, panel_id, dists, model_name="", plot_title=None):
    markers = np.unique([m.split("_")[0] for m in dists.index.to_list()])

    distances_for_marker_WT = {m: None for m in markers}
    distances_for_marker_ALS = distances_for_marker_WT.copy()

    for ind in dists.index:
        m, cell_line = ind.split('_')
        logging.info(f"[{fig_id} {panel_id}] {m} {cell_line}")
        cols_with_current_cellline = dists.columns.str.startswith(f"{m}_{cell_line}")
        cols_with_current_marker = dists.columns.str.startswith(f"{m}_")
        if cell_line == "WT":
            vals = dists.loc[cols_with_current_marker & cols_with_current_cellline,
                             cols_with_current_marker & ~cols_with_current_cellline]
            distances_for_marker_WT[m] = vals.mean(axis=1).values[0]
        else:
            if distances_for_marker_ALS[m] is not None:
                continue
            cols_with_WT = dists.columns.str.startswith(f"{m}_WT")
            vals = dists.loc[cols_with_current_marker & ~cols_with_WT,
                             cols_with_current_marker & ~cols_with_WT]
            distances_for_marker_ALS[m] = np.sum(np.triu(vals.to_numpy())) * 1.0 / (
                    (vals.shape[0] - 1) * vals.shape[0] / 2)

    df_WT = pd.DataFrame.from_dict(distances_for_marker_WT, orient='index')
    df_ALS = pd.DataFrame.from_dict(distances_for_marker_ALS, orient='index')

    merged = pd.merge(df_WT, df_ALS, left_index=True, right_index=True)
    merged.columns = ["WT", "ALS"]

    merged.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_WT.csv"))

    # Plot
    logging.info(f"[{fig_id} {panel_id}] Plot")
    plt.scatter(merged.loc[:, 'WT'], merged.loc[:, 'ALS'])
    for i in range(merged.shape[0]):
        plt.annotate(merged.index[i], (merged.iloc[i]['WT'], merged.iloc[i]['ALS']))
    plt.xlabel('WT')
    plt.ylabel('ALS')
    if plot_title:
        plt.title(plot_title)
    plt.savefig(f"./output/{fig_id}_{panel_id}_{model_name}_ranking.pdf", bbox_inches='tight')
    plt.show()


def __get_markers_ranking_per_type(dists, fig_id, panel_id, model_name):
    markers = np.unique([m.split("_")[0] for m in dists.index.to_list()])
    distances_for_marker_WT = {m: None for m in markers}
    distances_for_marker_ALS = distances_for_marker_WT.copy()
    for ind in dists.index:
        m, cell_line, cell_type = ind.split('_')
        logging.info(f"[{fig_id} {panel_id}] {m} {cell_line} {cell_type}")
        cols_with_current_cellline = dists.columns.str.startswith(f"{m}_{cell_line}")
        cols_with_current_marker = dists.columns.str.startswith(f"{m}_")
        if cell_line == "WT":
            vals = dists.loc[cols_with_current_marker & cols_with_current_cellline,
                             cols_with_current_marker & ~cols_with_current_cellline]
            distances_for_marker_WT[m] = vals.mean(axis=1).values[0]
        else:
            if distances_for_marker_ALS[m] is not None:
                continue
            cols_with_WT = dists.columns.str.startswith(f"{m}_WT")
            vals = dists.loc[cols_with_current_marker & ~cols_with_WT,
                             cols_with_current_marker & ~cols_with_WT]
            distances_for_marker_ALS[m] = np.sum(np.triu(vals.to_numpy())) * 1.0 / (
                    (vals.shape[0] - 1) * vals.shape[0] / 2)

    df_WT = pd.DataFrame.from_dict(distances_for_marker_WT, orient='index')
    df_ALS = pd.DataFrame.from_dict(distances_for_marker_ALS, orient='index')

    merged = pd.merge(df_WT, df_ALS, left_index=True, right_index=True)
    merged.columns = ["WT", "ALS"]

    merged.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_{cell_type}_dists.csv"))
    return merged


def __plot_markers_ranking_combined(fig_id, panel_id, dists_neurons, dists_microglia, model_name="", plot_title=None):
    merged_neurons = __get_markers_ranking_per_type(dists_neurons, fig_id, panel_id, model_name)
    merged_microglia = __get_markers_ranking_per_type(dists_microglia, fig_id, panel_id, model_name)
    # merged_microglia.rename(index={"SCNA": "SNCA", "NCL": "Nucleolin", "phalloidin": "Phalloidin"}, inplace=True)
    # Plot
    logging.info(f"[{fig_id} {panel_id}] Plot")
    plt.scatter(merged_neurons.loc[:, 'WT'], merged_neurons.loc[:, 'ALS'], c='blue', label="Neurons")
    for i in range(merged_neurons.shape[0]):
        plt.annotate(merged_neurons.index[i], (merged_neurons.iloc[i]['WT'], merged_neurons.iloc[i]['ALS']),
                     fontsize=6)
    plt.scatter(merged_microglia.loc[:, 'WT'], merged_microglia.loc[:, 'ALS'], c='red', label="Microglia")
    for i in range(merged_microglia.shape[0]):
        plt.annotate(merged_microglia.index[i], (merged_microglia.iloc[i]['WT'], merged_microglia.iloc[i]['ALS']),
                     fontsize=6)
    plt.xlabel('WT')
    plt.ylabel('ALS')
    if plot_title:
        plt.title(plot_title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"./output/{fig_id}_{panel_id}_{model_name}_ranking.pdf", bbox_inches='tight')
    plt.show()


def __get_figure_5_6(fig_id, input_folders, markers, model_path, groups_terms_line, model_name="MODEL",
                     panels_ids=None):
    # Load data
    logging.info(f"[{fig_id}] Load data")
    images, labels, labels_changepoints, markers_order = load_data(input_folders, markers=markers, \
                                                                   condition_l=False, type_l=True, \
                                                                   split_by_set=False,
                                                                   conds_include=["unstressed"])

    def get_panel_a():
        panel_id = "A"
        logging.info(f"[{fig_id} {panel_id}] init")

        # Load models
        logging.info(f"[{fig_id} {panel_id}] get analytics_{model_name}")
        analytics = get_analytics(images, labels, model_path=model_path)

        # Plot UMAPs
        logging.info(f"[{fig_id} {panel_id}] Plot multiplexed UMAPs")
        _, _ = multiplex(analytics, images, labels, \
                         groups_terms=groups_terms_line, markers=markers, show1=False, \
                         cmap1=["tab20", "tab20b", "tab20c", "tab10", "Set1", "Set2", "Set3", "Accent", "Dark2",
                                "Pastel1", "Pastel2", "Paired", "tab20", "tab20b", "tab20c", "tab10", "Set1",
                                "Set2", "Set3", "Accent", "Dark2", "Pastel1", "Pastel2", "Paired"],
                         markers_order=markers_order, savefig=True, annot_font_size1=9)

    def get_panel_b():
        panel_id = "B"
        dists = __calc_dist(fig_id, panel_id, images, labels, model_path, model_name)
        dists.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_dists_raw.csv"))
        __plot_markers_ranking(fig_id, panel_id, dists, model_name)

        __get_markers_ranking_per_cell_line(fig_id, panel_id, dists, groups_terms_line, model_name)

    def get_panel_c():
        panel_id = "C"
        logging.info(f"[{fig_id} {panel_id}] init")
        logging.info(f"[{fig_id} {panel_id}] Plot umap per marker")
        # Uncomment the following line later:
        __plot_umap_per_marker(images, labels, markers, model_path,
                               fig_id=fig_id, panel_id=panel_id,
                               savefig=True, output_filename_prefix=f"{model_name}_{fig_id}_{panel_id}_")

        dists = __calc_dist(fig_id, panel_id, images, labels, model_path, model_name)
        dists.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_dists_raw.csv"))
        __heatmap_dists(fig_id, panel_id, dists)

    def get_panel_d():
        panel_id = "D"
        logging.info(f"[{fig_id} {panel_id}] init")
        # Load data
        # logging.info(f"[{fig_id} {panel_id}] Load data")
        # images, labels, labels_changepoints, markers_order = load_data(input_folders, condition_l=False, type_l=True,
        #                                                               conds_include=["unstressed"],
        #                                                               split_by_set=True,
        #                                                               set_type="test",
        #                                                               markers=markers,
        #                                                               split_by_set_include=[("WT", "unstressed")])

        # get analytics
        colors_dict = get_colors_dict(groups_terms_line)
        for vq_layer in [1, 2]:
            logging.info(f"[{fig_id} {panel_id}] get analytics (for vq_layer={vq_layer})")
            analytics = get_analytics(images, labels, model_path=model_path)
            # clustermap_filename = f'{model_name}_indhist_heatmap_dgram_index2.npy'
            clustermap_filename_prefix = f'{model_name}_indhist_heatmap'  # TODO: change to this
            clustermap_filename_full = f"{clustermap_filename_prefix}_dgram_index{vq_layer}.npy"  # TODO: change to this
            if not os.path.exists(
                    # os.path.join(analytics.model.savepath_dict["ft"], f"{clustermap_filename}_dgram_index1.npy")):
                    os.path.join(analytics.model.savepath_dict["ft"],
                                 clustermap_filename_full)):  # TODO: change to this
                # calculate clustermaps
                logging.info(f"[{fig_id} {panel_id}] Calculating clustermaps for vq_layer {vq_layer}")
                # analytics.plot_clustermaps(savepath='default', target_vq_layer=1,
                #                            filename=clustermap_filename)
                analytics.plot_clustermaps(savepath='default', target_vq_layer=vq_layer,
                                           filename=clustermap_filename_prefix)  # TODO: change to this

            # Load clustermaps
            logging.info(f"[{fig_id} {panel_id}] Load clustermaps for vq_layer {vq_layer}")
            analytics.load_dendrogram_index(
                # os.path.join(analytics.model.savepath_dict["ft"], f"{clustermap_filename}_dgram_index1.npy"))
                os.path.join(analytics.model.savepath_dict["ft"], clustermap_filename_full))  # TODO: change to this
            logging.info(f"Plotting feature spectrum for vq_layer {vq_layer}:")
            for line in groups_terms_line:
                labels_s = pd.Series(labels.reshape(-1, ))
                line_indexes = labels_s[labels_s.str.contains(line)].index
                images_subset = images[line_indexes]
                line_clean = line.replace("_", "")
                logging.info(f"[{fig_id} {panel_id}] Current Item: {line_clean} for vq_layer {vq_layer}")
                plot_feature_spectrum_from_image(analytics, images_subset, savepath='default',
                                                 filename=f'{model_name}_{line_clean}_{vq_layer}', title=line_clean,
                                                 color=colors_dict[line], target_vq_layer=vq_layer, figsize=(15, 3))

    ##########################
    # Map and run the panels
    panels_mapping = {"a": get_panel_a, "b": get_panel_b, "c": get_panel_c, "d": get_panel_d}

    __run_panels(panels_mapping, panels_ids)
    ############################


def __get_figure_combined(panels_ids=None):
    model_name = 'Combined'
    fig_id = 'combined'
    # Load data
    logging.info(f"[{fig_id}] Load data")
    images, labels, labels_changepoints, markers_order = load_data(input_folders=input_folders_combined,
                                                                   markers=COMBINED_MARKERS,
                                                                   condition_l=False, type_l=True,
                                                                   split_by_set=False, cell_type_l=True,
                                                                   cell_lines_include=["WT", "FUS", "TDP43", 'OPTN'],
                                                                   conds_include=["unstressed"])

    def get_panel_a():
        panel_id = "A"
        logging.info(f"[{fig_id} {panel_id}] init")

        # Load models
        logging.info(f"[{fig_id} {panel_id}] get analytics_{model_name}")
        analytics = get_analytics(images, labels, model_path=combined_model_path)

        # Plot UMAPs
        logging.info(f"[{fig_id} {panel_id}] Plot multiplexed UMAPs")
        _, _ = multiplex(analytics, images, labels, legend_inside=False, alpha=0.3,
                         groups_terms=groups_terms_type, second_umap_figsize=(4, 4),
                         markers=COMBINED_MARKERS, show1=False, \
                         cmap1=["tab20", "tab20b", "tab20c", "tab10", "Set1", "Set2", "Set3", "Accent", "Dark2",
                                "Pastel1", "Pastel2", "Paired", "tab20", "tab20b", "tab20c", "tab10", "Set1",
                                "Set2", "Set3", "Accent", "Dark2", "Pastel1", "Pastel2", "Paired"],
                         markers_order=markers_order, savefig=True, annot_font_size1=9)

    def get_panel_b():
        panel_id = "B"
        dists_microglia, dists_neurons = __calc_dist_combined(fig_id, panel_id, images, labels, combined_model_path,
                                                              model_name)

        dists_microglia.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_dists_raw_microglia.csv"))
        dists_neurons.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_dists_raw_neurons.csv"))

        __plot_markers_ranking_combined(fig_id, panel_id, dists_neurons, dists_microglia, model_name)
        # __get_markers_ranking_per_cell_line(fig_id, panel_id, dists, groups_terms_type, model_name)

    def get_panel_c():
        panel_id = "C"
        logging.info(f"[{fig_id} {panel_id}] init")
        logging.info(f"[{fig_id} {panel_id}] Plot umap per marker")
        __plot_umap_per_marker(images, labels, COMBINED_MARKERS, combined_model_path,
                               fig_id=fig_id, panel_id=panel_id, alpha=0.3, s=1,
                               savefig=True, output_filename_prefix=f"{fig_id}_{panel_id}_{model_name}_")
        # need to think about this metric:
        # dists_microglia, dists_neurons = __calc_dist_combined(fig_id, panel_id, images, labels, combined_model_path,
        #                                                       model_name)
        # dists.to_csv(os.path.join(output_dir, f"{fig_id}_{panel_id}_{model_name}_dists_raw.csv"))
        # __heatmap_dists_combined(fig_id, panel_id, dists) #TODO: fix this!!
        #  if wants to create a 8*8 heatmap, need to change calc_dist function to also calc the
        #  distances between the different cell type...

    def get_panel_d():
        panel_id = "D"
        logging.info(f"[{fig_id} {panel_id}] init")

        # get analytics
        colors_dict = get_colors_dict(groups_terms_type)
        for vq_layer in [1, 2]:
            logging.info(f"[{fig_id} {panel_id}] get analytics (for vq_layer={vq_layer})")
            analytics = get_analytics(images, labels, model_path=combined_model_path)
            clustermap_filename_prefix = f'{model_name}_indhist_heatmap'
            clustermap_filename_full = f"{clustermap_filename_prefix}_dgram_index{vq_layer}.npy"
            if not os.path.exists(
                    os.path.join(analytics.model.savepath_dict["ft"],
                                 clustermap_filename_full)):  # TODO: change to this
                # calculate clustermaps
                logging.info(f"[{fig_id} {panel_id}] Calculating clustermaps for vq_layer {vq_layer}")
                analytics.plot_clustermaps(savepath='default', target_vq_layer=vq_layer,
                                           filename=clustermap_filename_prefix)

            # Load clustermaps
            logging.info(f"[{fig_id} {panel_id}] Load clustermaps for vq_layer {vq_layer}")
            analytics.load_dendrogram_index(
                # os.path.join(analytics.model.savepath_dict["ft"], f"{clustermap_filename}_dgram_index1.npy"))
                os.path.join(analytics.model.savepath_dict["ft"], clustermap_filename_full))  # TODO: change to this
            logging.info(f"Plotting feature spectrum for vq_layer {vq_layer}:")
            for line in groups_terms_type:
                labels_s = pd.Series(labels.reshape(-1, ))
                line_indexes = labels_s[labels_s.str.contains(line)].index
                images_subset = images[line_indexes]
                line_clean = line.replace("_", " ")
                logging.info(f"[{fig_id} {panel_id}] Current Item: {line_clean} for vq_layer {vq_layer}")
                plot_feature_spectrum_from_image(analytics, images_subset, savepath='default',
                                                 filename=f'{model_name}_{line_clean}_{vq_layer}', title=line_clean,
                                                 color=colors_dict[line], target_vq_layer=vq_layer, figsize=(15, 3))

    def get_panel_reconstruct():
        images, labels, labels_changepoints, markers_order = load_data(input_folders_combined, condition_l=False,
                                                                       type_l=False, cell_type_l = True,
                                                                       cell_lines_include=["WT"],
                                                                       split_by_set=False,
                                                                       #    set_type="test",
                                                                       markers=COMBINED_MARKERS,
                                                                       split_by_set_include=[("WT", "unstressed")])
        panel_id = "reconstruct"
        n_images = 20
        # Take only specific (selected) images
        marker_image_mapping = {"G3BP1": 3, "TIA1": 3, "DAPI": 1}
        logging.info(f"[{fig_id} {panel_id}] init")

        labels_s = pd.Series(labels.reshape(-1, ))

        for m in COMBINED_MARKERS:
            for cell_type in ['microglia', 'neurons']:
                logging.info(f"[{fig_id} {panel_id} {m}] Marker: {m} Type: {cell_type}")
                markers_regex = f'^{m}_.*{cell_type}'
                markers_indexes = labels_s[labels_s.str.contains(markers_regex)].index
                logging.info(f"[{fig_id} {panel_id} {m}] Laebls: {labels_s}")
                logging.info(f"[{fig_id} {panel_id} {m}] markers_regex: {markers_regex}")
                logging.info(f"[{fig_id} {panel_id} {m}] markers_indexes: {markers_indexes}")

                markers_indexes_subset = np.random.choice(markers_indexes,
                                                          size=min(n_images, len(markers_indexes)),
                                                          replace=False)
                # Take only specific (selected) images
                # markers_indexes_subset = [markers_indexes_subset[marker_image_mapping[m]]]
                # images_subset = images[markers_indexes_subset]
                # labels_subset = labels[markers_indexes_subset]

                images_subset = images[markers_indexes_subset]
                labels_subset = labels[markers_indexes_subset]

                # Load models
                logging.info(f"[{fig_id} {panel_id} {m} {cell_type}] get analytics_cytoself")
                analytics_cytoself = get_analytics(images_subset, labels_subset, model_path=cytoself_model_path)
                logging.info(f"[{fig_id} {panel_id} {m} {cell_type}] get analytics_combined")
                analytics = get_analytics(images_subset, labels_subset, model_path=combined_model_path)

                # Generate reconsturcted images and calculate reconstruction error (MSE)
                for i in range(len(images_subset)):
                    logging.info(f"[{fig_id} {panel_id} {m} {cell_type}] Generate reconstructed images (cytoself)")
                    rec_error = calc_reconstruction_error(analytics_cytoself, images_indexes=[i], show=True,
                                                          entire_data=images, cmap_original = 'Greys_r', cmap_reconstructed = 'rainbow',
                                                          embvecs=analytics_cytoself.model.embvec, reset_embvec=False)
                    logging.info(f"[{fig_id} {panel_id} {m} {cell_type} (cytoself)] {rec_error}")

                    logging.info(f"[{fig_id} {panel_id} {m} {cell_type}] Generate reconstructed images (combined)")
                    rec_error = calc_reconstruction_error(analytics, images_indexes=[i], show=True,
                                                          entire_data=images,cmap_original = 'Greys_r', cmap_reconstructed = 'rainbow',
                                                          embvecs=analytics.model.embvec, reset_embvec=False)
                    logging.info(f"[{fig_id} {panel_id} {m} {cell_type} (combined)] {rec_error}")

            ##########################

    ##########################
    # Map and run the panels
    panels_mapping = {"a": get_panel_a, "b": get_panel_b, "c": get_panel_c, "d": get_panel_d,
                      "reconstruct": get_panel_reconstruct}

    __run_panels(panels_mapping, panels_ids)
    ############################


def __get_supp(panels_ids=None):
    fig_id = "supp"

    def get_panel_umap0():
        panel_id = 'UMAP0'

        # Load data
        logging.info(f"[{fig_id} {panel_id}] Load data")
        images, labels, labels_changepoints, markers_order = load_data(input_folders, markers=MARKERS, \
                                                                       condition_l=False, type_l=True, \
                                                                       split_by_set=True,
                                                                       set_type='test',
                                                                       conds_include=["unstressed"],
                                                                       split_by_set_include=[("WT", "unstressed")])

        logging.info(f"[{fig_id} [{panel_id}]] Plot umap")
        __plot_umap_per_marker(images, labels, MARKERS, neuroself_model_path,
                               fig_id=fig_id, panel_id=panel_id,
                               savefig=True, output_filename_prefix=f"MODEL18_{fig_id}_{panel_id}_")

    ##########################
    # Map and run the panels
    panels_mapping = {"umap0": get_panel_umap0}

    __run_panels(panels_mapping, panels_ids)
    ############################
    
def __get_pertrubations(panels_ids=None):
    fig_id = "pertrubations"
    markers_pertrubations = ['Calreticulin', 'DAPI', 'NCL', 'NONO', 'PURA', 'SQSTM1']
    pertrubations = ['Chloroquine', 'Riluzole', 'Untreated',
                                   'DMSO1uM', 'Pridopine', 'Edavarone',
                                    'Tubustatin', 'DMSO100uM']
    groups_terms_pertrubations = ['_' + p for p in pertrubations]
    logging.info(f"[{fig_id}] init")

    logging.info(f"[{fig_id}] Loading data")
    logging.info(f"[{fig_id}] Markers: {markers_pertrubations}")
    advanced_selection =  [("WT", "Untreated")] + [("TDP43", "Untreated")]#[("TDP43", p) for p in pertrubations] #  #
    logging.info(f"[{fig_id}] Advanced selection: {advanced_selection}")
    
    
    input_folders_path = input_folders_pertrubations_spd2 #input_folders_pertrubations_spd # input_folders_pertrubations
    
    images, labels, labels_changepoints, markers_order = load_data(input_folders_path, condition_l=True, type_l=True,
                                                                   cell_lines_include=["WT", 'TDP43'],
                                                                   split_by_set=False,
                                                                   advanced_selection=advanced_selection, 
                                                                   # set_type="test",
                                                                   markers=markers_pertrubations)


    def get_panel_b():
        panel_id = 'B'

        logging.info(f"[{fig_id} {panel_id}] Plot UMAPs")

        __plot_umap_per_marker(images, labels, markers_pertrubations, neuroself_model_path, fig_id=fig_id,alpha=0.35,figsize=(10,8),
                               panel_id=panel_id, savefig=True, output_filename_prefix=f'MODEL18_{fig_id}_{panel_id}_',
                               calc_metrics=True, use_colors_dict=False,cmap='Set1')

        # logging.info(f"[{fig_id} {panel_id}] Arrange UMAPs")
        # arrange_plots('./umaps', nrows=5, ncols=6, file_name_contains='MODEL18_')

    def get_panel_d():
        raise NotImplemented()
        
        panel_id = "D"

        # logging.info(f"[{fig_id}] Loading data")
        # images, labels, labels_changepoints, markers_order = load_data(input_folders, condition_l=True, type_l=False,
        #                                                               cell_lines_include=["WT"],
        #                                                               split_by_set=False,
        #                                                               markers=MARKERS,
        #                                                               split_by_set_include=[("WT", "unstressed")])

        # Load models
        logging.info(f"[{fig_id} {panel_id}] get analytics_neuroself")
        analytics_neuroself = get_analytics(images, labels, model_path=neuroself_model_path)

        # Plot UMAPs
        logging.info(f"[{fig_id} {panel_id}] Plot multiplexed UMAPs")
        colors_dict = get_colors_dict(labels)
        _, _ = multiplex(analytics_neuroself, images, labels, \
                         groups_terms=groups_terms_pertrubations, markers=markers_pertrubations, \
                         # s1=5,
                         # s2=20,
                         #colors_dict1=colors_dict,
                         markers_order=markers_order,
                         annot_font_size1=9,
                         show1=False,
                         savefig=True)

    ##########################
    # Map and run the panels
    panels_mapping = {"b": get_panel_b, 'd': get_panel_d}

    __run_panels(panels_mapping, panels_ids)
    ############################


figs_mapping = {"1": __get_figure1, "2": __get_figure2,
                "3": __get_figure3, "4": __get_figure4,
                "5": __get_figure5, "6": __get_figure6,
                "supp": __get_supp, "combined": __get_figure_combined,
                "pertrubations": __get_pertrubations}
