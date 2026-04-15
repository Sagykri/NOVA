#!/bin/bash

# Script to submit all FuNOVA organizer jobs for panels A-D and plates 1-4

# Define panels and plates
PANELS=("A" "B" "C" "D")
PLATES=(1 2 3 4)

# Base paths
MAIN_PY="/home/projects/hornsteinlab/giliwo/NOVA/tools/images_organizer/FuNOVA_screen/main.py"
CONFIG_DIR="./NOVA/tools/images_organizer/FuNOVA_screen/config_panels_funova"

# Loop over plates and panels
for plate in "${PLATES[@]}"; do
    for panel in "${PANELS[@]}"; do
        JOB_NAME="org_p${plate}_${panel}"
        OUT_FILE="screen_batch1_plate${plate}_panel${panel}_organizer.out"
        CONFIG_FILE="${CONFIG_DIR}/Config_${panel}"

        bsub -q long -R "rusage[mem=4800]" -J "$JOB_NAME" -o "$OUT_FILE" "python $MAIN_PY $CONFIG_FILE 1 $plate"
    done
done

echo "All jobs submitted!"