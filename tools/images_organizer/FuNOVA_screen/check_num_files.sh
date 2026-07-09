
panels=("1" "2" "3" "4")
BATCH="batch2"
BASE_DIR="/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen/sorted/$BATCH/C9"
LOG_FILE="/home/projects/hornsteinlab/giliwo/NOVA/tools/images_organizer/FuNOVA_screen/check_num_files_per_marker_$BATCH.log"

current_date=$(date +%F)
echo "$current_date" > $LOG_FILE
for panel_num in "${panels[@]}"; do
        panel_dir="${BASE_DIR}/Panel${panel_num}"
        echo "Panel: ${panel_num}" | tee -a "$LOG_FILE"
        if [ ! -d "$panel_dir" ]; then
            echo "SKIP (missing): $panel_dir" | tee -a "$LOG_FILE"
            continue
        fi
    
        for condition_dir in "$panel_dir"/*/; do
            condition_name=$(basename "$condition_dir")

            for rep_dir in "$condition_dir"*/; do
                rep_name=$(basename "$rep_dir")

                for marker_dir in "$rep_dir"*/; do
                    marker_name=$(basename "$marker_dir")
                    num_files=$(ls "$marker_dir" | wc -l)

                    if [[ $num_files -ne 169 ]]; then
                        echo "ERROR: panel:$panel_num, cond:$condition_name, rep:$rep_name, marker:$marker_name - found $num_files files (expected 169)" | tee -a "$LOG_FILE"
                    else
                        echo "panel:$panel_num, cond:$condition_name, rep:$rep_name, marker:$marker_name - $num_files files" | tee -a "$LOG_FILE"
                    fi
                done
            done
        done
        #     condition_name=$(basename "$condition_dir")
        #     num_files=$(find "$condition_dir" -type f | wc -l)
        #     echo "  Condition: ${condition_name} - Num files: ${num_files}" | tee -a "$LOG_FILE"
        # done
        echo "*********************************************" | tee -a "$LOG_FILE"
done