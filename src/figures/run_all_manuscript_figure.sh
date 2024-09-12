echo "MOMAPS_HOME:" $MOMAPS_HOME

pretrain_model=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/pretrained_model
figures_data_configs=./src/figures/manuscript_figures_data_config
plot_configs=./src/figures/manuscript_plot_config
#parameters order:
# model_folder config_data umap_type

for model in $models
do
    model_folder=$models_folder$model
#     ####
#     # UMAP1
#     ###

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_vit -m 5000 \
#     -a $model_folder $figures_configs/NeuronsUMAP1B78FigureConfig $plot_configs/UMAP1Config umap1 -q short -j umap1
