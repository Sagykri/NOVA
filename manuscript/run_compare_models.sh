echo "NOVA_HOME:" $NOVA_HOME

output_dir="/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/model_evaluation_01"
k=20
neg_k=20
sample_fraction=0.1
mem=70000

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_coyne -a AlyssaCoyne_7tiles batch1 $output_dir $k $neg_k $sample_fraction

# dNLS
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_dNLS3 -a deltaNLS batch3 $output_dir $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_dNLS4 -a deltaNLS batch4 $output_dir $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_dNLS5 -a deltaNLS batch5 $output_dir $k $neg_k $sample_fraction

# neurons
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_neurons4 -a neurons batch4 $output_dir $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_neurons5 -a neurons batch5 $output_dir $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_neurons6 -a neurons batch6 $output_dir $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_neurons7 -a neurons batch7 $output_dir $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_neurons8 -a neurons batch8 $output_dir $k $neg_k $sample_fraction
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/compare_models -g -b 1 -m $mem -j eval_neurons9 -a neurons batch9 $output_dir $k $neg_k $sample_fraction