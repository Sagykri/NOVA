echo "MOMAPS_HOME:" $MOMAPS_HOME

models_folder=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/
# models="batch78_infoNCE transfer_b78_freeze_all_but_attn transfer_b78_freeze_least_changed transfer_b78_freeze_mostly_changed transfer_b78_no_freeze opencell_new"
models="opencell_new" #"batch78_infoNCE_FMRP"
figures_configs=./src/figures/model_comparisons_figures_config
#parameters order:
# model_folder config_data umap_type

for model in $models
do
    model_folder=$models_folder$model
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

     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 25000 \
     -a $model_folder $figures_configs/NeuronsUMAP2B6BothRepsFigureConfig sm -q short -j sm

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 25000 \
     -a $model_folder $figures_configs/NeuronsUMAP2B6BothRepsNOFUSFigureConfig sm no_fus -q short -j sm

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 25000 \
     -a $model_folder $figures_configs/NeuronsUMAP2B6BothRepsNOFUSNOSCNAFigureConfig sm no_fus_no_scna -q short -j sm

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 25000 \
     -a $model_folder $figures_configs/NeuronsUMAP2B6BothRepsNOSCNAFigureConfig sm no_scna -q short -j sm

     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 25000 \
     -a $model_folder $figures_configs/NeuronsUMAP2B9BothRepsFigureConfig sm -q short -j sm

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 25000 \
     -a $model_folder $figures_configs/NeuronsUMAP2B9BothRepsNOFUSFigureConfig sm no_fus -q short -j sm

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 25000 \
     -a $model_folder $figures_configs/NeuronsUMAP2B9BothRepsNOFUSNOSCNAFigureConfig sm no_fus_no_scna -q short -j sm

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 25000 \
     -a $model_folder $figures_configs/NeuronsUMAP2B9BothRepsNOSCNAFigureConfig sm no_scna -q short -j sm


done


# #### Opencell #####
# models_folder=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/
# model_folder=$models_folder/opencell_new
# figures_configs=./src/figures/model_comparisons_figures_config

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

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000\
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

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000\
# -a $model_folder $figures_configs/NeuronsUMAP0B9BothRepsALSFigureConfig umap0 ALS -q short -j umap0

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 50000 \
# -a $model_folder $figures_configs/NeuronsUMAP0B69BothRepsALSFigureConfig umap0 ALS -q short -j umap0

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