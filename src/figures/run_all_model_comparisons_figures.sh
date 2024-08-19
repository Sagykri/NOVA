echo "MOMAPS_HOME:" $MOMAPS_HOME

# models_folder=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/
# # models="batch78_infoNCE transfer_b78_freeze_all_but_attn transfer_b78_freeze_least_changed transfer_b78_freeze_mostly_changed transfer_b78_no_freeze"
# models="opencell" #"batch78_infoNCE_FMRP"
# figures_configs=./src/figures/model_comparisons_figures_config
#parameters order:
# model_folder config_data umap_type

# for model in $models
# do
#     model_folder=$models_folder$model
#     ####
#     # UMAP1
#     ###

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#     -a $model_folder $figures_configs/NeuronsUMAP1B78FigureConfig umap1 -q short -j umap1

#     ####
#     # UMAP0 - stress
#     # ###
#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1StressFigureConfig umap0 stress -q short -j umap0

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2StressFigureConfig umap0 stress -q short -j umap0

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B6BothRepsStressFigureConfig umap0 stress -q short -j umap0

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1StressFigureConfig umap0 stress -q short -j umap0

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000\
#      -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2StressFigureConfig umap0 stress -q short -j umap0

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B9BothRepsStressFigureConfig umap0 stress -q short -j umap0

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B69BothRepsStressFigureConfig umap0 stress -q short -j umap0

    
#     ####
#     # UMAP0 - ALS lines
#     ###
#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALSFigureConfig umap0 ALS -q short -j umap0

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALSFigureConfig umap0 ALS -q short -j umap0

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B6BothRepsALSFigureConfig umap0 ALS -q short -j umap0

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALSFigureConfig umap0 ALS -q short -j umap0

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALSFigureConfig umap0 ALS -q short -j umap0

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000\
#      -a $model_folder $figures_configs/NeuronsUMAP0B9BothRepsALSFigureConfig umap0 ALS -q short -j umap0

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP0B69BothRepsALSFigureConfig umap0 ALS -q short -j umap0

#     ####
#     # UMAP0 - deltaNLS
#     ###
#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/EmbeddingsdNLSB2Rep1DatasetConfig umap0 -q short -j umap0

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/EmbeddingsdNLSB2Rep2DatasetConfig umap0 -q short -j umap0

#       $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/EmbeddingsdNLSB3Rep1DatasetConfig umap0 -q short -j umap0

#       $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/EmbeddingsdNLSB3Rep2DatasetConfig umap0 -q short -j umap0

#       $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/EmbeddingsdNLSB4Rep1DatasetConfig umap0 -q short -j umap0

#       $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/EmbeddingsdNLSB4Rep2DatasetConfig umap0 -q short -j umap0

#       $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/EmbeddingsdNLSB5Rep1DatasetConfig umap0 -q short -j umap0

#       $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/EmbeddingsdNLSB5Rep2DatasetConfig umap0 -q short -j umap0


#     ####
#     # UMAP2
#     ###
#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2FUSFigureConfig sm fus_lines -q short -j sm

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2FigureConfig sm -q short -j sm

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2FUSLinesNOFUSFigureConfig sm fus_lines_no_fus -q short -j sm

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2NOFUSFigureConfig sm no_fus -q short -j sm

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2NOFUSLinesFigureConfig sm no_fus_lines -q short -j sm

#      $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#      -a $model_folder $figures_configs/dNLSUMAP2B3BothRepsFigureConfig sm -q short -j sm

# done

#######################################################
models_folder=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/
figures_configs=./src/figures/model_comparisons_figures_config



#### Opencell #####
# model_folder=$models_folder/opencell_new

#### Fine Tuned ###
model_folder=$models_folder/transfer_b78_freeze_least_changed


#######################################


# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsU2OSRep1FigureConfig umap0 -q short -j umap0

####
# UMAP1
###

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP1B78OpencellFigureConfig umap1 -q short -j umap1

# ####
# # UMAP0 - stress
# # ###
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1StressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2StressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6BothRepsStressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1StressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2StressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9BothRepsStressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B69BothRepsStressFigureConfig umap0 stress -q short -j umap0

    
# ####
# # UMAP0 - ALS lines
# ###
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6BothRepsALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9BothRepsALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B69BothRepsALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6BothRepsOnlyFUSLinesOnlyFUSMarkerFigureConfig umap0 only_FUS_lines -q short -j umap0

# ####
# # UMAP0 - deltaNLS
# ###
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB2Rep1DatasetConfig umap0 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB2Rep2DatasetConfig umap0 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB3Rep1DatasetConfig umap0 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB3Rep2DatasetConfig umap0 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB4Rep1DatasetConfig umap0 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB4Rep2DatasetConfig umap0 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB5Rep1DatasetConfig umap0 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB5Rep2DatasetConfig umap0 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB3BothRepsJoinedDatasetConfig umap0 joined_reps -q short -j umap0

### deltaNLS - only WT Untreated
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB3BothRepsOnlyWTOnlyDCP1ADatasetConfig umap0 WT_Untreated -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB4BothRepsOnlyWTOnlyDCP1ADatasetConfig umap0 WT_Untreated -q short -j umap0


### deltaNLS - only TDP43 Untreated
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB3BothRepsOnlyTDP43UntreatedDatasetConfig umap0 TDP43_Untreated -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB4BothRepsOnlyTDP43UntreatedDatasetConfig umap0 TDP43_Untreated -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB5BothRepsOnlyTDP43UntreatedDatasetConfig umap0 TDP43_Untreated -q long -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB2BothRepsOnlyTDP43UntreatedDatasetConfig umap0 TDP43_Untreated -q long -j umap0

### deltaNLS - only TDP43 Dox
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
-a $model_folder $figures_configs/EmbeddingsdNLSB3BothRepsOnlyTDP43DoxDatasetConfig umap0 TDP43_Dox -q short -j umap0

$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
-a $model_folder $figures_configs/EmbeddingsdNLSB4BothRepsOnlyTDP43DoxDatasetConfig umap0 TDP43_Dox -q short -j umap0

### deltaNLS - only TDP43 lines
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB3Rep1OnlyTDP43LineOnlyDCP1ADatasetConfig umap0 TDP43Lines -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB3Rep2OnlyTDP43LineOnlyDCP1ADatasetConfig umap0 TDP43Lines -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB4Rep1OnlyTDP43LineOnlyDCP1ADatasetConfig umap0 TDP43Lines -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/EmbeddingsdNLSB4Rep2OnlyTDP43LineOnlyDCP1ADatasetConfig umap0 TDP43Lines -q short -j umap0


# ####
# # UMAP2
# ###
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2FUSFigureConfig sm fus_lines -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2FigureConfig sm -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2FUSLinesNOFUSFigureConfig sm fus_lines_no_fus -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2NOFUSFigureConfig sm no_fus -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2B6Rep2NOFUSLinesFigureConfig sm no_fus_lines -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/dNLSUMAP2B3BothRepsFigureConfig sm -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/dNLSUMAP2B3BothRepsWithoutWTFigureConfig sm without_wt -q short -j sm


### ALS Pairs ##

# ####
# # UMAP0 - ALS lines
# ###
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0


# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0

# b78 full

#
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2ALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2ALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2ALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2ALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2ALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2ALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1ALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1ALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1ALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1ALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1ALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1ALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0

# #
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2ALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2ALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2ALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2ALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2ALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2ALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1ALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1ALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1ALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1ALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1ALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1ALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0

# b78 testset

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2TestsetALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2TestsetALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2TestsetALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2TestsetALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2TestsetALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2TestsetALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1TestsetALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1TestsetALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1TestsetALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1TestsetALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1TestsetALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1TestsetALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43_testset -q short -j umap0

# #
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2TestsetALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2TestsetALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2TestsetALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2TestsetALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2TestsetALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2TestsetALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1TestsetALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1TestsetALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1TestsetALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1TestsetALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1TestsetALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1_testset -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1TestsetALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43_testset -q short -j umap0

# Only WT Untreated
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6BothRepsOnlyWTUntreatedFigureConfig umap0 WT_Untreated -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9BothRepsOnlyWTUntreatedFigureConfig umap0 WT_Untreated -q short -j umap0


# TBK1 as baseline
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALS_TBK1_OPTN_FigureConfig umap0 ALS_TBK1_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALS_TBK1_OPTN_FigureConfig umap0 ALS_TBK1_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALS_TBK1_FUSHet_FigureConfig umap0 ALS_TBK1_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALS_TBK1_FUSHet_FigureConfig umap0 ALS_TBK1_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep1ALS_TBK1_TDP43_FigureConfig umap0 ALS_TBK1_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B6Rep2ALS_TBK1_TDP43_FigureConfig umap0 ALS_TBK1_TDP43 -q short -j umap0



# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALS_TBK1_OPTN_FigureConfig umap0 ALS_TBK1_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALS_TBK1_OPTN_FigureConfig umap0 ALS_TBK1_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALS_TBK1_FUSHet_FigureConfig umap0 ALS_TBK1_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALS_TBK1_FUSHet_FigureConfig umap0 ALS_TBK1_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep1ALS_TBK1_TDP43_FigureConfig umap0 ALS_TBK1_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B9Rep2ALS_TBK1_TDP43_FigureConfig umap0 ALS_TBK1_TDP43 -q short -j umap0


#

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1ALS_TBK1_OPTN_FigureConfig umap0 ALS_TBK1_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2ALS_TBK1_OPTN_FigureConfig umap0 ALS_TBK1_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1ALS_TBK1_FUSHet_FigureConfig umap0 ALS_TBK1_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2ALS_TBK1_FUSHet_FigureConfig umap0 ALS_TBK1_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep1ALS_TBK1_TDP43_FigureConfig umap0 ALS_TBK1_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B7Rep2ALS_TBK1_TDP43_FigureConfig umap0 ALS_TBK1_TDP43 -q short -j umap0



# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1ALS_TBK1_OPTN_FigureConfig umap0 ALS_TBK1_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2ALS_TBK1_OPTN_FigureConfig umap0 ALS_TBK1_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1ALS_TBK1_FUSHet_FigureConfig umap0 ALS_TBK1_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2ALS_TBK1_FUSHet_FigureConfig umap0 ALS_TBK1_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep1ALS_TBK1_TDP43_FigureConfig umap0 ALS_TBK1_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B8Rep2ALS_TBK1_TDP43_FigureConfig umap0 ALS_TBK1_TDP43 -q short -j umap0


######################## OPERA 18 days (REIMAGED) ############

####
# UMAP1
###

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP1B12Opera18daysREIMAGEDFigureConfig umap1 -q short -j umap1

# ####
# # UMAP0 - stress
# # ###
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Opera18daysREIMAGEDB1BothRepsStressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Opera18daysREIMAGEDB2BothRepsStressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Opera18daysREIMAGEDB1Rep1StressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Opera18daysREIMAGEDB2Rep1StressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Opera18daysREIMAGEDB1Rep2StressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Opera18daysREIMAGEDB2Rep2StressFigureConfig umap0 stress -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Opera18daysREIMAGEDBothBatchesBothRepsStressFigureConfig umap0 stress -q short -j umap0

# ####
# # UMAP0 - ALS lines
# ###
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB1BothRepsALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB2BothRepsALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB1Rep1ALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB1Rep2ALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB2Rep1ALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB2Rep2ALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDBothBatchesBothRepsALSFigureConfig umap0 ALS -q short -j umap0

# with SCNA
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB1BothRepsALSWithSNCAFigureConfig umap0 ALS_with_SNCA -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB2BothRepsALSWithSNCAFigureConfig umap0 ALS_with_SNCA -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB2BothRepsJoinedALSWithSNCAFigureConfig umap0 ALS_with_SNCA_joined_reps -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB1BothRepsJoinedALSWithSNCAFigureConfig umap0 ALS_with_SNCA_joined_reps -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB1Rep1ALSWithSNCAFigureConfig umap0 ALS_with_SNCA -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB1Rep2ALSWithSNCAFigureConfig umap0 ALS_with_SNCA -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB2Rep1ALSWithSNCAFigureConfig umap0 ALS_with_SNCA -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0Bpera18daysREIMAGEDB2Rep2ALSWithSNCAFigureConfig umap0 ALS_with_SNCA -q short -j umap0

# ####
# # UMAP2
# ###
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB1BothRepsALSFigureConfig sm ALS -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB2BothRepsALSFigureConfig sm ALS -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB1Rep1ALSFigureConfig sm ALS -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB1Rep2ALSFigureConfig sm ALS -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB2Rep1ALSFigureConfig sm ALS -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB2Rep2ALSFigureConfig sm ALS -q short -j sm

# with SNCA
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB1BothRepsALSWithSNCAFigureConfig sm ALS_with_SNCA -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB2BothRepsALSWithSNCAFigureConfig sm ALS_with_SNCA -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB1Rep1ALSWithSNCAFigureConfig sm ALS_with_SNCA -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB1Rep2ALSWithSNCAFigureConfig sm ALS_with_SNCA -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB2Rep1ALSWithSNCAFigureConfig sm ALS_with_SNCA -q short -j sm

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP2Bpera18daysREIMAGEDB2Rep2ALSWithSNCAFigureConfig sm ALS_with_SNCA -q short -j sm


### ALS Pairs ##

# ####
# # UMAP0 - ALS lines
# ###
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep2Opera18daysREIMAGEDALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B1Rep1Opera18daysREIMAGEDALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0


# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep2Opera18daysREIMAGEDALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_FUSHet_FigureConfig umap0 ALS_WT_FUSHet -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_FUSHom_FigureConfig umap0 ALS_WT_FUSHom -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_FUSRev_FigureConfig umap0 ALS_WT_FUSRev -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_OPTN_FigureConfig umap0 ALS_WT_OPTN -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_TBK1_FigureConfig umap0 ALS_WT_TBK1 -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B2Rep1Opera18daysREIMAGEDALS_WT_TDP43_FigureConfig umap0 ALS_WT_TDP43 -q short -j umap0