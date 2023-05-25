#!/bin/bash

# Usage example:
# ./bash_commands/run_py.sh ./src/runables/training -a /home/labs/hornsteinlab/Collaboration/MOmaps/src/models/neuroself/configs/train_config/NeuroselfTrainConfig /home/labs/hornsteinlab/Collaboration/MOmaps/src/models/neuroself/configs/val_config/NeuroselfValConfig /home/labs/hornsteinlab/Collaboration/MOmaps/src/models/neuroself/configs/test_config/NeuroselfTestConfig -m 40000 -g True
# ./bash_commands/run_py.sh ./src/runables/preprocessing -a ./src/preprocessing/configs/spd_batch7/SPD_Batch7  -m 70000 -g
# ./bash_commands/run_py.sh ./tests/test_gpu  -m 1000 -g

# Required param
py_name=$1

# Default values
mem=15000
use_gpu=false
gmem=5

shift

while getopts "m:ga:b:" opt; do
  case ${opt} in
    m ) mem="$OPTARG";;
    g ) use_gpu=true;;
    b ) gmem="${OPTARG,,}";;
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
  
echo "py_name: $py_name, mem: $mem, use_gpu: $use_gpu, args: $args, gmem: $gmem"

if [ "$use_gpu" = false ]
then
  bsub -n 1 -q new-long -m "public_himem_2020_hosts public_2017_hosts" -J Run_$py_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
else
  bsub -n 1 -q gpu-long -gpu "num=1:gmem=${gmem}G" -J Run_$py_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
fi
