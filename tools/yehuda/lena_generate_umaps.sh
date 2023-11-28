echo "Starting script..."

echo "Exporting MOMAPS_HOME and MOMAPS_DATA_HOME"
export MOMAPS_HOME="/home/labs/hornsteinlab/Collaboration/MOmaps"
export MOMAPS_DATA_HOME="/home/labs/hornsteinlab/Collaboration/MOmaps/input"

momaps_main="/home/labs/hornsteinlab/Collaboration/MOmaps"

echo "MOMAPS_HOME=$MOMAPS_HOME, MOMAPS_DATA_HOME=$MOMAPS_DATA_HOME"

echo "Loading miniconda module"
module load miniconda

echo "Activating conda env"
conda activate "$momaps_main/anaconda3/momaps_torch"

echo "Running script"
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 -a ./tools/yehuda/lena_model_config/NeuroselfLenaConfig ./tools/yehuda/lena_dataset_config/LenaDatasetConfig /home/labs/hornsteinlab/Collaboration/MOmaps/tools/yehuda/ -q gpu-short

echo "The script is now running on the LSF (the cloud). Keep track via the RTM website: https://rtm1.wexac.weizmann.ac.il/"