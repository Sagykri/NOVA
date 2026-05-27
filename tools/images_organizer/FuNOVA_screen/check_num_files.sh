
panels=("1" "2" "3" "4")

BASE_DIR="/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen/sorted/batch1/C9"
LOG_FILE="/home/projects/hornsteinlab/giliwo/NOVA/tools/images_organizer/FuNOVA_screen/check_num_files.log"


for panel_num in "${panels[@]}"; do
        panel_dir="${BASE_DIR}/Panel${panel_num}"
        echo "Panel: ${panel_num}" | tee -a "$LOG_FILE"
        if [ ! -d "$panel_dir" ]; then
            echo "SKIP (missing): $panel_dir" | tee -a "$LOG_FILE"
            continue
        fi
    
        for condition_dir in "$panel_dir"/*/; do
            condition_name=$(basename "$condition_dir")
            num_files=$(find "$condition_dir" -type f | wc -l)
            echo "  Condition: ${condition_name} - Num files: ${num_files}" | tee -a "$LOG_FILE"
        done
        echo "*********************************************" | tee -a "$LOG_FILE"
done