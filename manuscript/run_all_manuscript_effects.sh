echo "NOVA_HOME:" $NOVA_HOME

finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen=/home/projects/hornsteinlab/Collaboration/NOVA/outputs/vit_models/finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen

effects_configs=./manuscript/effects_config

##################
# finetuned_model
##################

### Neurons d8 ###
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineFUSHomoConfig -q short -j effects_WT_FUSHom -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineFUSHeteroConfig -q short -j effects_WT_FUSHet -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineFUSRevConfig -q short -j effects_WT_FUSRev -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineTDP43Config -q short -j effects_WT_TDP43 -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineOPTNConfig -q short -j effects_WT_OPTN -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineTBK1Config -q short -j effects_WT_TBK1 -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineSNCAConfig -q short -j effects_WT_SNCA -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectWTBaselineStressConfig -q short -j effects_WT_Stress -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineFUSHomoConfig -q short -j effects_FUSRev_FUSHom -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineFUSHeteroConfig -q short -j effects_FUSRev_FUSHet -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineWTConfig -q short -j effects_FUSRev_WT -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineTDP43Config -q short -j effects_FUSRev_TDP43 -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineOPTNConfig -q short -j effects_FUSRev_OPTN -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineTBK1Config -q short -j effects_FUSRev_TBK1 -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsEffectFUSRevBaselineSNCAConfig -q short -j effects_FUSRev_SNCA -m 250000

# ### Alyssa OLD###
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_alyssa \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/AlyssaCoyneEffectConfig -q short -j effect_alyssa_old

# ### Alyssa NEW###
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_alyssa_new \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/AlyssaCoyneNEWEffectConfig -q short -j effect_alyssa_new


# ##### new dNLS #####
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/dNLSNewEffectConfig -q short -j effect_dnlsNew -m 250000

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_alyssa \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/iAstrocytesEffectConfig -q short -j effect_iAstrocytes


#### Multiplex ####

# neuronsDay8_new

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_multiplex \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsMultiplexEffectConfig -q short -j effect_multi_d8 -m 200000


$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_multiplex \
-a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NeuronsMultiplexEffectConfig_WithSNCA -q short -j effect_multi_d8_withSNCA -m 200000

# dNLS 
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_multiplex \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/dNLSMultiplexEffectConfig -q short -j effect_multi_dNLS -m 200000


# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_multiplex \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/dNLSMultiplexEffectConfig_NBoot1000 -q short -j effect_multi_dNLS -m 200000

# # NIH
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_effects_multiplex \
# -a $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $effects_configs/NIHMultiplexEffectConfig -q short -j effect_multi_NIH -m 200000


# NANCY: runnables/run.sh runnables/generate_effects_multiplex -a outputs/vit_models/finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen/ ./manuscript/effects_config/NeuronsMultiplexEffectConfig -q short -j effect_multi -m 500000