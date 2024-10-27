echo "NOVA_HOME:" $NOVA_HOME

pretrain_model=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/pretrained_model
finetune_model=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model
cytoself_model=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/cytoself_vqvae2_vit_format
cellprofiler=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/cell_profiler
data_configs=./manuscript/manuscript_figures_data_config
plot_configs=./manuscript/manuscript_plot_config

# # #######################
# # # Figure 1
# # # #######################

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
-a $finetune_model $data_configs/NeuronsUMAP1B9FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b9_finetune

# # #######################
# # # Figure 1 - supp
# # #######################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP1B6FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b6_finetune

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP1B9WithoutDapiFigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b9_finetune_wo_dapi

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $cytoself_model $data_configs/NeuronsUMAP1B9WithoutDapiFigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b9_cyto

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $cellprofiler $data_configs/NeuronsUMAP1B9FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b9_cp

# # #######################
# # # Figure 2
# # # #######################
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0StressB9WithoutDAPIFigureConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0StressB9DAPIFigureConfig $plot_configs/UMAP0StressDAPIPlotConfig -q short -j umap0_stress_9_dapi

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 5000 \
# -a $finetune_model $data_configs/NeuronsDistancesStressWith45FigureConfig $plot_configs/DistancesNeuronsStressPlotConfig -q short -j dists_stress_with_45

# # #######################
# # # Figure 2 - supp
# # #######################
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 1000 \
# -a $finetune_model $data_configs/U2OSUMAP0StressDatasetConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress_U2OS

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 5000 \
# -a $cellprofiler $data_configs/NeuronsDistancesStressFigureConfig $plot_configs/DistancesNeuronsStressPlotConfig -q short -j dists_stress_cellprofiler


# # # #######################
# # # # Figure 3
# # # #######################
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 1000 \
# -a $finetune_model $data_configs/dNLSUMAP0B3DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_dnls_b3

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 1000 \
# -a $finetune_model $data_configs/dNLSDistancesFigureConfig $plot_configs/DistancesdNLSPlotConfig -q short -j dnls_dist


# #######################
# # Figure 3 -  supp
# #######################
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 1000 \
# -a $finetune_model $data_configs/dNLSUMAP0B4DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_dnls_b4

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 1000 \
# -a $finetune_model $data_configs/dNLSUMAP0B5DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_dnls_b5

# #######################
# # Figure 5
# #######################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9ALSLinesFigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_b9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_bubble -m 1000 \
# -a $finetune_model $data_configs/NeuronsDistancesALSWith45FigureConfig $plot_configs/DistancesNeuronsALSPlotConfig -q short -j als_dist

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB9FUSFigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b9_FUS

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB9DCP1AFigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b9_dcp1a

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
-a $finetune_model $data_configs/NeuronsUMAP0ALSB6ANXA11FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b6_anxa11

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
-a $finetune_model $data_configs/NeuronsUMAP0ALSB6TDP43FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b6_tdp43

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB9CLTCFigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b9_cltc

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB9SQSTM1FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b9_sqstm1

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j umap2_alyssa_groups

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorControlsPlotConfig -q short -j umap2_alyssa_control

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorsALSPositivePlotConfig -q short -j umap2_alyssa_pos

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorsALSNegativePlotConfig -q short -j umap2_alyssa_neg

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorC9PlotConfig -q short -j umap2_alyssa_c9

# # #######################
# # # Figure 5 - supp
# # #######################

# # #### stress umap2 ####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2StressB6FigureConfig $plot_configs/UMAP2StressPlotConfig -q short -j umap2_stress_b6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2StressB9FigureConfig $plot_configs/UMAP2StressPlotConfig -q short -j umap2_stress_b9

# # #### dnls umap2 ####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/dNLSUMAP2B3FigureConfig $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_dNLS_b3

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/dNLSUMAP2B4FigureConfig $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_dNLS_b4

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/dNLSUMAP2B5FigureConfig $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_dNLS_b5

# #### als umap2 ####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB6ALSLinesFigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_b6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_shuffled_synthetic_superposition -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9ALSLinesFigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j shuffled_umap2

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9ALSLinesWOSNCAFigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_snca_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB6ALSLinesWOSNCAFigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_snca_6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB6ALSLines_wo_fusFigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fus_6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9ALSLines_wo_fusFigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fus_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9FUSFigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_FUS

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSD18B1FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_d18_1

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSD18B2FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_d18_2

# # #### distances ####

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_bubble -m 1000 \
# -a $finetune_model $data_configs/NeuronsDistancesALSD18FigureConfig $plot_configs/DistancesNeuronsFUSD18PlotConfig -q short -j als_dist_d18

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_bubble -m 1000 \
# -a $finetune_model $data_configs/NeuronsDistancesALSFUSFigureConfig $plot_configs/DistancesNeuronsFUSPlotConfig -q short -j als_dist_FUS


###############################

########## ALYSSA ################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_bubble -m 1000 \
# -a $finetune_model $data_configs/AlyssaCoyneDistancesFigureConfig $plot_configs/DistancesAlyssaCoynePlotConfig -q short -j alyssa_dist

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP0FigureConfig $plot_configs/UMAP0AlyssaCoyneColorByGroupPlotConfig -q short -j umap0_alyssa

