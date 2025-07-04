echo "NOVA_HOME:" $NOVA_HOME

output_folder=/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model
output_folder_cp=/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/cell_profiler

effects_configs=./manuscript/effects_config

##################
# finetuned_model
##################

### Neurons d8

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder $effects_configs/NeuronsEffectWithBioReplicateConfig -q short -j dist_neurons

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder $effects_configs/NeuronsEffectWithBioReplicateConfig -q short -j dist_neurons_with_345

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder $effects_configs/NeuronsTBK1EffectConfig -q short -j dist_neurons_tbk1

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder $effects_configs/dNLSNewEffectConfig -q short -j dist_dnlsNew -m 30000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder $effects_configs/Day18EffectConfig -q short -j dist_d18

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects\
# -a $output_folder $effects_configs/AlyssaCoyneEffectConfig -q short -j dist_alyssa

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $output_folder_cp $effects_configs/NeuronsEffectConfig -q short -j dist_cellprofiler

finetunedModel_MLPHead_frozen=/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetunedModel_MLPHead_frozen

# ##### new dNLS #####
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
-a $finetunedModel_MLPHead_frozen $effects_configs/dNLSNewEffectConfig -q short -j dist_dnlsNew -m 60000

# ###### old dNLS ######
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_frozen $effects_configs/dNLS345EffectConfig -q short -j dist_dnls -m 30000

###### neurons ######
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_frozen $effects_configs/NeuronsEffectWithBioReplicateConfig -q short -j dists -m 30000