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

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP1B9FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b9_finetune

# # #######################
# # # Figure 1 - supp
# # #######################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP1B6FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b6_finetune

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $pretrain_model $data_configs/NeuronsUMAP1B6FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b6_pretrain

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $pretrain_model $data_configs/NeuronsUMAP1B9FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b9_pretrain

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $cytoself_model $data_configs/NeuronsUMAP1B6FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b6_cyto

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $cytoself_model $data_configs/NeuronsUMAP1B9FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b9_cyto

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $cellprofiler $data_configs/NeuronsUMAP1B6FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b6_cp

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $cellprofiler $data_configs/NeuronsUMAP1B9FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_b9_cp

# # #######################
# # # Figure 2
# # # #######################
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0StressB9FigureConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 5000 \
# -a $finetune_model $data_configs/NeuronsDistancesStressFigureConfig $plot_configs/DistancesNeuronsStressPlotConfig -q short -j dists_stress

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 5000 \
# -a $finetune_model $data_configs/NeuronsDistancesStressFigureConfig $plot_configs/DistancesNeuronsStressNoBaselinePlotConfig -q short -j dists_stress

# # #######################
# # # Figure 2 - supp
# # #######################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0StressB6FigureConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress_6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0StressB3FigureConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress_3

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0StressB4FigureConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress_4

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0StressB5FigureConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress_5

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 1000 \
# -a $finetune_model $data_configs/U2OSUMAP0StressDatasetConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress_U2OS

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 5000 \
# -a $finetune_model $data_configs/NeuronsDistancesStressWith45FigureConfig $plot_configs/DistancesNeuronsStressPlotConfig -q short -j dists_stress_with_45

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 5000 \
# -a $cellprofiler $data_configs/NeuronsDistancesStressFigureConfig $plot_configs/DistancesNeuronsStressPlotConfig -q short -j dists_stress_cellprofiler

# # # #######################
# # # # Figure 3
# # # #######################
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 1000 \
# -a $finetune_model $data_configs/dNLSUMAP0B3DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_dnls_b3

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 1000 \
# -a $finetune_model $data_configs/dNLSUMAP0B4DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_dnls_b4

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 1000 \
# -a $finetune_model $data_configs/dNLSUMAP0B5DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_dnls_b5

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 1000 \
# -a $finetune_model $data_configs/dNLSDistancesFigureConfig $plot_configs/DistancesdNLSPlotConfig -q short -j dnls_dist

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 1000 \
# -a $finetune_model $data_configs/dNLSDistancesFigureConfig $plot_configs/DistancesdNLSNoBaselinePlotConfig -q short -j dnls_dist

# #######################
# # Figure 3 -  supp - some umap0 of other batch
# #######################

# #######################
# # Figure 5
# #######################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_b9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_bubble -m 1000 \
# -a $finetune_model $data_configs/NeuronsDistancesALSFigureConfig $plot_configs/DistancesNeuronsALSPlotConfig -q short -j als_dist

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB9FUSFigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b9_FUS

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB9DCP1AFigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b9_dcp1a

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB9ANXA11FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b9_anxa11

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB9CLTCFigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b9_cltc

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j umap2_alyssa

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorControlsPlotConfig -q short -j umap2_alyssa

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorsALSPositivePlotConfig -q short -j umap2_alyssa

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorsALSNegativePlotConfig -q short -j umap2_alyssa

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorC9PlotConfig -q short -j umap2_alyssa

# # #######################
# # # Figure 5 - supp
# # #######################

# #### als umap2 ####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB6FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_b6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB3FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_b3

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB4FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_b4

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB5FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_b5

# # #### stress umap2 ####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2StressB6FigureConfig $plot_configs/UMAP2StressPlotConfig -q short -j umap2_stress_b6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2StressB9FigureConfig $plot_configs/UMAP2StressPlotConfig -q short -j umap2_stress_b9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2StressB3FigureConfig $plot_configs/UMAP2StressPlotConfig -q short -j umap2_stress_b3

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2StressB4FigureConfig $plot_configs/UMAP2StressPlotConfig -q short -j umap2_stress_b4

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2StressB5FigureConfig $plot_configs/UMAP2StressPlotConfig -q short -j umap2_stress_b5

# # #### dnls umap2 ####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/dNLSUMAP2B3FigureConfig $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_dNLS_b3

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/dNLSUMAP2B4FigureConfig $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_dNLS_b4

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/dNLSUMAP2B5FigureConfig $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_dNLS_b5

# # #### distances ####

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_bubble -m 1000 \
-a $finetune_model $data_configs/NeuronsDistancesALSWith45FigureConfig $plot_configs/DistancesNeuronsALSPlotConfig -q short -j als_dist_with_45



# #### multiple pairs (batch6) ####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB6FUSFigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b6_FUS

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB6DCP1AFigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b6_dcp1a

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB6ANXA11FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b6_anxa11

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALSB6CLTCFigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_b6_cltc


# ### pairs ####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_FUSHomozygous_B9FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_fushomo_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_FUSHeterozygous_B9FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_fushetero_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_FUSRevertant_B9FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_fusrev_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_TBK1_B9FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_tbk1_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_OPTN_B9FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_optn_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_TDP43_B9FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_tdp43_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_FUSHomozygous_B6FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_fushomo_6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_FUSHeterozygous_B6FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_fushetero_6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_FUSRevertant_B6FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_fusrev_6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_TBK1_B6FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_tbk1_6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_OPTN_B6FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_optn_6

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_TDP43_B6FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_tdp43_6
###############################

########## ALYSSA ################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_bubble -m 1000 \
# -a $finetune_model $data_configs/AlyssaCoyneDistancesFigureConfig $plot_configs/DistancesAlyssaCoynePlotConfig -q short -j alyssa_dist

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP0FigureConfig $plot_configs/UMAP0AlyssaCoyneColorByGroupPlotConfig -q short -j umap0_alyssa

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/AlyssaCoyneUMAP0FigureConfig $plot_configs/UMAP0AlyssaCoyneColorByPatientPlotConfig -q short -j umap0_alyssa


############ special umap2 als ###################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB6_without_fus_marker_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fus

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB6_without_SCNA_line_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_snca

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB6_without_fushomo_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fushomo

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB6_without_fushetero_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fushetero

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB6_without_fusrev_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fusrev

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9_without_fus_marker_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fus_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9_without_SCNA_line_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_snca_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9_without_fushomo_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fushomo_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9_without_fushetero_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fushetero_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSB9_without_fusrev_FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_no_fusrev_9

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSD18B1FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_d18_1

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP2ALSD18B2FigureConfig $plot_configs/UMAP2ALSPlotConfig -q short -j umap2_als_d18_2


# # #######################
# # # experimental
# # #######################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $finetune_model $data_configs/NeuronsUMAP1D18B1FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_d18

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 1000 \
# -a $pretrain_model $data_configs/U2OSUMAP0StressDatasetConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress_U2OS


# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_FUSHomozygous_B69FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_fushomo_69

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_FUSHeterozygous_B69FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_fushetero_69

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_FUSRevertant_B69FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_fusrev_69

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_TBK1_B69FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_tbk1_69

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_OPTN_B69FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_optn_69

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0ALS_TDP43_B69FigureConfig $plot_configs/UMAP0ALSPlotConfig -q short -j umap0_als_tdp43_69
