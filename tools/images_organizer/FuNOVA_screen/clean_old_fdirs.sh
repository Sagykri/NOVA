
panels=("1" "2" "3" "4")
plates=("" "_p1" "_p2" "_p3" "_p4")
conditions=("non-targeting_00004_00017" "non-targeting_00010_00031" "non-targeting_00035_00050" "non-targeting_00053_00059" "non-targeting_00111_00121" "Empty" "Ranbp17" "TDP-43")
BASE_DIR="/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen/sorted/batch1/C9"
LOG_FILE="/home/projects/hornsteinlab/giliwo/NOVA/tools/images_organizer/FuNOVA_screen/clean_old_dirs.log"

DRY_RUN=false  # set to false to actually run

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "$@"
    else
        "$@"
    fi
}

for panel_num in "${panels[@]}"; do
    for condition in "${conditions[@]}"; do
        for plate_num in "${plates[@]}"; do

            old_name="${BASE_DIR}/Panel${panel_num}/${condition}${plate_num}_old"

            # check if old dir does not exist
            if [ ! -d "$old_name" ]; then
                echo "Skipping: ${old_name} does not exist" >> $LOG_FILE
                echo "*********************************************" >> $LOG_FILE
                continue
            fi

            # delete old dir
            echo "Deleting ${old_name}" >> $LOG_FILE
            echo "removing num files:" >> $LOG_FILE
            find "$old_name" -type f | wc -l >> $LOG_FILE
            run_cmd rm -rf "$old_name" >> $LOG_FILE

            echo "*********************************************" >> $LOG_FILE
        done
    done
done