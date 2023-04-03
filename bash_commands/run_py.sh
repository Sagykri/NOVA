py_name=$1
mem=$2
use_gpu=${3,,}

if [ -z "$2" ]; then
    mem=15000
fi

if [ -z "$3" ]; then
    use_gpu=false
fi

echo "py_name: $py_name, mem: $mem, use_gpu: $use_gpu"

if [ "$use_gpu" = false ]
then
  bsub -n 1 -q new-long -m "public_himem_2020_hosts public_2017_hosts" -J Run_$py_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py
else
  bsub -q gpu-long -gpu "num=1:gmem=10G" -J Run_$py_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py
fi

