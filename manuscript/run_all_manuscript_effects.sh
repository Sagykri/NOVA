echo "NOVA_HOME:" $NOVA_HOME

output_folder=/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model
output_folder_cp=/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/cell_profiler
finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen=/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen

effects_configs=./manuscript/effects_config

##################
# finetuned_model
##################

### Neurons d8 ###
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineFUSHomoConfig -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineFUSHeteroConfig -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineFUSRevConfig -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineTDP43Config -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineOPTNConfig -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineTBK1Config -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineStressConfig -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineFUSHomoConfig -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineFUSHeteroConfig -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineWTConfig -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineTDP43Config -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineOPTNConfig -q short -j dists -m 60000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineTBK1Config -q short -j dists -m 60000

### Alyssa OLD###
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_alyssa \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/AlyssaCoyneEffectConfig -q short -j dist_alyssa


# ##### new dNLS #####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/dNLSNewEffectConfig -q short -j dist_dnlsNew -m 60000


$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_alyssa \
-a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/iAstrocytesEffectConfig -q short -j dist_iAstrocytes
