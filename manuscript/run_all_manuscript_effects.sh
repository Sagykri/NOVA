echo "NOVA_HOME:" $NOVA_HOME

output_folder=/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model
output_folder_cp=/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/cell_profiler

distances_configs=./manuscript/distances_config

##################
# finetuned_model
##################

### Neurons d8

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder $distances_configs/NeuronsDistanceConfig -q short -j dist_neurons

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder $distances_configs/NeuronsDistanceWithBioReplicateConfig -q short -j dist_neurons_with_345

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder $distances_configs/NeuronsTBK1DistanceConfig -q short -j dist_neurons_tbk1

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
-a $output_folder $distances_configs/dNLS345DistanceConfig -q short -j dist_dnls345

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder $distances_configs/Day18DistanceConfig -q short -j dist_d18

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects\
# -a $output_folder $distances_configs/AlyssaCoyneDistanceConfig -q short -j dist_alyssa

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder_cp $distances_configs/NeuronsDistanceConfig -q short -j dist_cellprofiler
