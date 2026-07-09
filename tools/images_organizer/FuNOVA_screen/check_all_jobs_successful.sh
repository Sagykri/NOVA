#!/bin/bash

# Script to submit all FuNOVA organizer jobs for panels A-D and plates 1-4

# Define panels and plates
PANELS=("A" "B" "C" "D")
PLATES=(1 2 3 4)

CONFIG_DIR="./NOVA/tools/images_organizer/FuNOVA_screen/config_panels_funova"

# Loop over plates and panels
for plate in "${PLATES[@]}"; do
    for panel in "${PANELS[@]}"; do

        OUT_FILE="screen_batch2_plate${plate}_panel${panel}_organizer.out"

        echo "Checking job for plate ${plate}, panel ${panel}..."
        tail -50 "$OUT_FILE" | grep "Successfully completed."
         
    done
done

