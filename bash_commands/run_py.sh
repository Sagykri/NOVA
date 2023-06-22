#!/bin/bash

# Usage example:
# ./bash_commands/run_py.sh ./src/runables/training -a /home/labs/hornsteinlab/Collaboration/MOmaps/src/models/neuroself/configs/train_config/NeuroselfTrainConfig /home/labs/hornsteinlab/Collaboration/MOmaps/src/models/neuroself/configs/val_config/NeuroselfValConfig /home/labs/hornsteinlab/Collaboration/MOmaps/src/models/neuroself/configs/test_config/NeuroselfTestConfig -m 40000 -g 
# ./bash_commands/run_py.sh ./src/runables/preprocessing -a ./src/preprocessing/configs/spd_batch7/SPD_Batch7  -m 70000 -g 
# ./bash_commands/run_py.sh ./src/runables/training -g -m 70000 -b 40 -a ./src/datasets/configs/train_batch2_dm_config/TrainBatch2DMDatasetConfig ./src/datasets/configs/val_batch2_dm_config/ValBatch2DMDatasetConfig ./src/datasets/configs/test_batch2_dm_config/TestBatch2DMDatasetConfig

# Required param
py_name=$1

# Default values
mem=15000
use_gpu=false
gmem=5
ngpu=1
queue=""

shift

while getopts "m:ga:b:n:q:" opt; do
  case ${opt} in
    m ) mem="$OPTARG";;
    g ) use_gpu=true;;
    b ) gmem="${OPTARG,,}";;
    n ) ngpu="$OPTARG";;
    q ) queue="$OPTARG";;
    a ) 
        args=($OPTARG)
        while [[ ${@:$OPTIND:1} != -* && ${@:$OPTIND:1} != "" ]]; do
          args+=("${@:$OPTIND:1}")
          ((OPTIND++))
        done
        IFS=" " args="${args[*]}"
        ;;
    \? ) echo "Invalid option: -$OPTARG" 1>&2; exit 1;;
    : ) echo "Option -$OPTARG requires an argument." 1>&2; exit 1;;
  esac
done
  
if [ -z "$queue" ]; then  
  if [ "$use_gpu" = false ]; then
    queue="new-long"
  else
    queue="gpu-long"
  fi
fi

echo "py_name: $py_name, mem: $mem, use_gpu: $use_gpu, args: $args, gmem: $gmem, ngpu: $ngpu, queue: $queue"

if [ "$use_gpu" = false ]
then
  bsub -n 1 -q $queue -m "public_himem_2020_hosts public_2017_hosts" -J Run_$py_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
else
  bsub -n 1 -q $queue -gpu "num=${ngpu}:gmem=${gmem}G:j_exclusive=yes" -J Run_$py_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
fi

