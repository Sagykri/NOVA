echo "NOVA_HOME:" $NOVA_HOME

# No freeze

# # CL
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/train -g -m 40000 -b 44 -j train_across -a ./manuscript/model_config/ContrastiveModelConfig ./manuscript/trainer_config/ContrastiveTrainerNoFreezeConfig_MLPHead_acrossBatches_B56789_80pct  ./manuscript/dataset_config/B56789DatasetConfig_80pct
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/train -g -m 40000 -b 44 -j train -a ./manuscript/model_config/ContrastiveModelConfig ./manuscript/trainer_config/ContrastiveTrainerNoFreezeConfig_MLPHead_B56789_80pct  ./manuscript/dataset_config/B56789DatasetConfig_80pct


# Frozen

# CL
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/train -g -m 40000 -b 44 -j train_CL_b_f -a ./manuscript/model_config/ContrastiveModelConfig ./manuscript/trainer_config/ContrastiveTrainerFrozenConfig_MLPHead_acrossBatches_B56789_80pct  ./manuscript/dataset_config/B56789DatasetConfig_80pct
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/train -g -m 40000 -b 44 -j train_CL_f -a ./manuscript/model_config/ContrastiveModelConfig ./manuscript/trainer_config/ContrastiveTrainerFrozenConfig_MLPHead_B56789_80pct  ./manuscript/dataset_config/B56789DatasetConfig_80pct