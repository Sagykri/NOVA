echo "NOVA_HOME:" $NOVA_HOME

vit_models=/home/projects/hornsteinlab/Collaboration/NOVA/outputs/vit_models

models_path=(
    $vit_models/pretrained_model
    $vit_models/finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen
)

data_configs=./manuscript/manuscript_figures_data_config

for model_path in "${models_path[@]}"; do
    model_name=$(basename "$model_path")

    echo "Running distances for model: $model_name"

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_B1 -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_B1

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_B2 -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_B2

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_B3 -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_B3

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_B7 -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_B7

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_B8 -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_B8

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_B9 -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_B9

    # Rep effect

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_RepEffect_B1 ref_effect -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_repEffect_B1

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_RepEffect_B2 ref_effect -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_repEffect_B2

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_RepEffect_B3 ref_effect -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_repEffect_B3

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_RepEffect_B7 ref_effect -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_repEffect_B7

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_RepEffect_B8 ref_effect -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_repEffect_B8

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/calculate_distances -g -m 20000 -b 10 \
    -a $model_path $data_configs/newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker_RepEffect_B9 ref_effect -q short-gpu -j dis_nD8_WT_vs_FUSLines_FUSMarker_repEffect_B9

done