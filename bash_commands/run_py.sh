#!/bin/bash

# Usage example:
# ./bash_commands/run_py.sh ./src/runables/training -g -m 30000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfBatch8BS4TrainingConfig ./src/datasets/configs/train_config/TrainDatasetConfig
# ./bash_commands/run_py.sh ./src/runables/training -g -m 20000 -b 40 -a ./src/models/cytoself_vanilla/configs/config/CytoselfTrainingConfig ./src/datasets/configs/train_config/TrainOpenCellDatasetConfig
# ./bash_commands/run_py.sh ./src/runables/preprocessing -g -m 30000 -b 40 -a ./src/preprocessing/configs/spd_batch3/SPD_Batch3  
# ./bash_commands/run_py.sh ./tests/test_gpu  -m 1000 -g

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
