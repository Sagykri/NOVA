echo "NOVA_HOME:" $NOVA_HOME

vit_models=/home/projects/hornsteinlab/Collaboration/NOVA/outputs/vit_models
nova_model=$vit_models/finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen
pretrained_model=$vit_models/pretrained_model

cytoself_model=/home/projects/hornsteinlab/Collaboration/NOVA/outputs/cytoself_model


data_configs=./manuscript/manuscript_figures_data_config
plot_configs=./manuscript/manuscript_plot_config
effects_configs=./manuscript/effects_config


###################################################################################################
################################# Figure 1 and Supplementary Figures ##############################
###################################################################################################

# Panel C: NIH - UMAP1 (all batches combined)  #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 60000 \
# -a $nova_model $data_configs/NIH_UMAP1_DatasetConfig $plot_configs/UMAP1PlotConfig -q short -j u1_NIH1

# ###################### Supplementary Figures #######################

# ### Sup Fig 4 ###
# # Panel B: NIH - UMAP1 (pretrained)  #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 60000 \
# -a $pretrained_model $data_configs/NIH_UMAP1_DatasetConfig $plot_configs/UMAP1PlotConfig -q short -j u1_NIH1

# ### Sup Fig 8 ###
# # NIH - UMAP1 (pretrained)  #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 60000 \
# -a $cytoself_model $data_configs/NIH_UMAP1_DatasetConfig $plot_configs/UMAP1PlotConfig -q short -j u1_NIH1


# ###################################################################################################
# ################################# Figure 2 and Supplementary Figures ##############################
# ###################################################################################################

# # Panel A: NIH - effect size Untreated vs Stress #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes -m 1000 \
# -a $nova_model $data_configs/NIHEffectsFigureConfig  $plot_configs/DistancesNeuronsStressPlotConfig -q short -j effect_sizes

# # Panel B: NIH - UMAP0 (all batches combined) Untreated vs Stress #
# # G3BP1, FMRP, Mitotracker, PML, TOMM20, PURA, DCP1A
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
# -a $nova_model $data_configs/NIH_UMAP0_Stress_DatasetConfig_Hits $plot_configs/UMAP0StressPlotConfig -q short -j u0_NIH1_stress

# ###################################################################################################
# ################################# Figure 3 and Supplementary Figures ##############################
# ###################################################################################################

# # Panel C: dNLS - effect size #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes -m 1000 -a \
# $nova_model $data_configs/dNLSEffectsFigureConfig  $plot_configs/DistancesdNLSPlotConfig -q short -j dnls_new

# # Panel D - F: dNLS - UMAP0 (TDP43, LSM14A, DCP1A) #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
# -a $nova_model $data_configs/newDNLSUMAP0DatasetConfig_Hits $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_new_dnls_Hits


# Panel G: AlyssaCoyne (pilot) UMAP0 #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $nova_model $data_configs/AlyssaCoyneUMAP0FigureConfig_Ctrl_C9 $plot_configs/UMAP0AlyssaCoyneColorByPatientPlotConfig -q short -j umap0_alyssa_patient

# Panel H: AlyssaCoyne (new) UMAP0 #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $nova_model $data_configs/newAlyssaFigureConfig_UMAP0_B1_per_patient_Ctrl_C9 $plot_configs/UMAP0newAlyssaCoyne -q short -j umap0_newAlyssa_patient


# ###################################################################################################
# ################################# Figure 5 and Supplementary Figures ##############################
# ###################################################################################################

# Panel H: Effect size multiplexed neuronsDay8_new #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes_multiplex -m 3000 -a \
# $nova_model $data_configs/NeuronsMultiplexedEffectsFigureConfig  $plot_configs/DistancesNeuronsALSPlotConfig -q short -j effect_multi_d8_plot

# ###################### Supplementary Figures #######################

#### Sup Fig 20 ###

# Panel A: UMAP2 - NIH Untreated vs Stress #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
# -a $nova_model $data_configs/NIH_UMAP2_Stress_DatasetConfig $plot_configs/UMAP2StressPlotConfig -q short -j u2_NIH1_stress

# Panel B: UMAP2 - dNLS Untreated vs DOX #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
# -a $nova_model $data_configs/newDNLSFigureConfig_UMAP2 $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_new_dnls

# ###################################################################################################
# ################################# Figure 6 and Supplementary Figures ##############################
# ###################################################################################################

# Panel A: AlyssaCoyne (pilot) UMAP2 #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $nova_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j umap2_alyssa_groups

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $nova_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorControlsPlotConfig -q short -j umap2_alyssa_control

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $nova_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorsALSPositivePlotConfig -q short -j umap2_alyssa_pos

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $nova_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorsALSNegativePlotConfig -q short -j umap2_alyssa_neg

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
# -a $nova_model $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorC9PlotConfig -q short -j umap2_alyssa_c9

# Panel B: AlyssaCoyne (new) marker ranking (Ctrl vs C9) #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_alyssa_new \
# -a $nova_model $effects_configs/AlyssaCoyneNEWEffectConfig_Ctrl_C9 -q short -j effect_alyssa_new

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes_alyssa_new -m 1000 \
# -a $nova_model $data_configs/AlyssaCoyneNEWEffectsFigureConfig_Ctrl_C9 $plot_configs/DistancesAlyssaCoynePlotConfig -q short -j new_alyssa_dist

# Panel D: AlyssaCoyne (new) UMAP2 (+distances, effect size) (Ctrl vs C9) #
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
# -a $nova_model $data_configs/newAlyssaFigureConfig_Ctrl_C9_UMAP2_B1_P1 $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j u2_nAlys_p1_$model_name

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
# -a $nova_model $data_configs/newAlyssaFigureConfig_Ctrl_C9_UMAP2_B1_P2 $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j u2_nAlys_p2_$model_name

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
# -a $nova_model $data_configs/newAlyssaFigureConfig_Ctrl_C9_UMAP2_B1_P3 $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j u2_nAlys_p3_$model_name

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
# -a $nova_model $data_configs/newAlyssaFigureConfig_Ctrl_C9_UMAP2_B1_with_patientID $plot_configs/UMAP2AlyssaCoyneColorByPatientPlotConfig -q short -j u2_nAlys_p_$model_name

# Effect size
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_alyssa_new_multiplex \
# -a $nova_model $effects_configs/AlyssaCoyneNEWMultiplexEffectConfig_Ctrl_C9 -q short -j effect_multi_alyssaNew -m 200000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes_alyssa_new_multiplex -m 1000 \
# -a $nova_model $data_configs/AlyssaCoyneNEWEffectsFigureConfig_Ctrl_C9_Multiplexed $plot_configs/DistancesAlyssaCoynePlotConfig -q short -j new_alyssa_dist_mul
