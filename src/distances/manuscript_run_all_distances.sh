echo "MOMAPS_HOME:" $MOMAPS_HOME

output_folder=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model

distances_configs=./src/distances/distances_config

##################
# finetuned_model
##################

### Neurons d8
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_distances \
-a $output_folder $distances_configs/NeuronsDistanceConfig -j dist_neurons

$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_distances \
-a $output_folder $distances_configs/NeuronsTBK1DistanceConfig -j dist_neurons_tbk1

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_distances \
# -a $output_folder $distances_configs/dNLS345DistanceConfig -j dist_dnls345

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_distances \
# -a $output_folder $distances_configs/Day18DistanceConfig -j dist_d18