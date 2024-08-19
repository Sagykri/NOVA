echo "MOMAPS_HOME:" $MOMAPS_HOME

models_folder=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/
models="batch78_infoNCE batch78_infoNCE_FMRP transfer_b78_freeze_all_but_attn transfer_b78_freeze_least_changed transfer_b78_freeze_mostly_changed transfer_b78_no_freeze opencell_new"
distances_configs=./src/distances/model_comparisons_distances_config
#parameters order:
# model_folder config_data

for model in $models
do
    model_folder=$models_folder$model

    # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/calculate_embeddings_distances_vit -m 50000 \
    # -a $model_folder $distances_configs/NeuronsDistanceConfig -q short -j dist

    # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/calculate_embeddings_distances_vit -m 50000 \
    # -a $model_folder $distances_configs/NeuronsTest78DistanceConfig _test78 -q short -j dist

    # # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/calculate_embeddings_distances_vit -m 30000 \
    # # -a $model_folder $distances_configs/dNLSDistanceConfig _full -q short -j dist

    # # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/calculate_embeddings_distances_vit -m 30000 \
    # # -a $model_folder $distances_configs/dNLSTest25DistanceConfig _test25 -q short -j dist

    # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/calculate_embeddings_distances_vit -m 30000 \
    # -a $model_folder $distances_configs/dNLS345DistanceConfig _no_b2 -q short -j dist

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/calculate_embeddings_distances_vit -m 30000 \
    -a $model_folder $distances_configs/EmbeddingsDay18DistanceConfig -q short -j dist
    

done
