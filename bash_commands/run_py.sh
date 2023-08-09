#!/bin/bash

# Required param
py_name=$1

# Default values
mem=15000
use_gpu=false
gmem=5
ngpu=1
queue=""
job_name="Run_$py_name"

shift

while getopts "m:ga:b:n:q:j:" opt; do
  case ${opt} in
    m ) mem="$OPTARG";;
    g ) use_gpu=true;;
    b ) gmem="${OPTARG,,}";;
    n ) ngpu="$OPTARG";;
    q ) queue="$OPTARG";;
    j ) job_name="$OPTARG";;
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

echo "py_name: $py_name, mem: $mem, use_gpu: $use_gpu, args: $args, gmem: $gmem, ngpu: $ngpu, queue: $queue, job_name: $job_name"

if [ "$use_gpu" = false ]
then
  bsub -n 1 -q $queue -m "public_himem_2020_hosts public_2017_hosts" -J $job_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
else
  bsub -n 1 -q $queue -gpu "num=${ngpu}:gmem=${gmem}G:j_exclusive=yes" -J $job_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
fi
