echo "NOVA_HOME:" $NOVA_HOME

mem=100000
output_dir="$NOVA_HOME/outputs/model_evaluations"
models_path_filepath="$NOVA_HOME/manuscript/models_path.json"
embeddings_path="./manuscript/embeddings_config_80pct"
k=10
neg_k=20
sample_fraction=0.1

# Alyssa Coyne
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/evaluate_models -g -b 1 -m 10000 -j eval_coyne -a $output_dir $embeddings_path/"EmbeddingsAlyssaCoyneDatasetConfig_NoRepNoBatchNoDAPI" $models_path_filepath $k $neg_k 1

# # dNLS
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/evaluate_models -g -b 1 -m $mem -j eval_dNLS3 -a $output_dir $embeddings_path/"EmbeddingsdNLSB3DatasetConfig_NoRepNoBatchNoDAPI" $models_path_filepath $k $neg_k 0.3
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/evaluate_models -g -b 1 -m $mem -j eval_dNLS4 -a $output_dir $embeddings_path/"EmbeddingsdNLSB4DatasetConfig_NoRepNoBatchNoDAPI" $models_path_filepath $k $neg_k 0.3
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/evaluate_models -g -b 1 -m $mem -j eval_dNLS5 -a $output_dir $embeddings_path/"EmbeddingsdNLSB5DatasetConfig_NoRepNoBatchNoDAPI" $models_path_filepath $k $neg_k 0.5

# neurons
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/evaluate_models -g -b 1 -m $mem -j eval_neurons4 -a $output_dir $embeddings_path/"EmbeddingsB4DatasetConfig_NoRepNoBatchNoDAPI" $models_path_filepath $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/evaluate_models -g -b 1 -m $mem -j eval_neurons5 -a $output_dir $embeddings_path/"EmbeddingsB5DatasetConfig_NoRepNoBatchNoDAPI" $models_path_filepath $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/evaluate_models -g -b 1 -m $mem -j eval_neurons6 -a $output_dir $embeddings_path/"EmbeddingsB6DatasetConfig_NoRepNoBatchNoDAPI" $models_path_filepath $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/evaluate_models -g -b 1 -m $mem -j eval_neurons9 -a $output_dir $embeddings_path/"EmbeddingsB9DatasetConfig_NoRepNoBatchNoDAPI" $models_path_filepath $k $neg_k $sample_fraction