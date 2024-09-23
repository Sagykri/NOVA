echo "NOVA_HOME:" $NOVA_HOME

output_folder=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model

distances_configs=./manuscript/distances_config

##################
# finetuned_model
##################

### Neurons d8
$NOVA_HOME/runnables/run.py $NOVA_HOME/runnables/generate_distances \
-a $output_folder $distances_configs/NeuronsDistanceConfig -j dist_neurons

$NOVA_HOME/runnables/run.py $NOVA_HOME/runnables/generate_distances \
-a $output_folder $distances_configs/NeuronsTBK1DistanceConfig -j dist_neurons_tbk1

$NOVA_HOME/runnables/run.py $NOVA_HOME/runnables/generate_distances \
-a $output_folder $distances_configs/dNLS345DistanceConfig -j dist_dnls345

$NOVA_HOME/runnables/run.py $NOVA_HOME/runnables/generate_distances \
-a $output_folder $distances_configs/Day18DistanceConfig -j dist_d18