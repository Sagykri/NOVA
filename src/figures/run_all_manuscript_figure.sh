echo "MOMAPS_HOME:" $MOMAPS_HOME

pretrain_model=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/pretrained_model
finetune_model=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model

data_configs=./src/figures/manuscript_figures_data_config
plot_configs=./src/figures/manuscript_plot_config

#######################
# Figure 1
#######################
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_and_plot -m 10000 \
# -a $pretrain_model $data_configs/NeuronsUMAP1B6FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1


#######################
# Figure 1 - supp
#######################
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_and_plot -m 10000 \
-a $pretrain_model $data_configs/NeuronsUMAP1B9FigureConfig $plot_configs/UMAP1PlotConfig -q short -j umap1

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps_and_plot -m 5000 \
# -a $finetune_model $data_configs/NeuronsUMAP0StressB6FigureConfig $plot_configs/UMAP0StressPlotConfig -q short -j umap0_stress