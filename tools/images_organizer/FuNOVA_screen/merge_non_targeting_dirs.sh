
panels=("1" "2" "3" "4")
plates=("_p1" "_p2" "_p3" "_p4")
reps=("1" "2")
# continue with "00010_00031" -- need to mix "non_targeting_00010_00031" and non-targeting_00010_00031
ref_condition="non_targeting_00111_00121"
alt_condition="non-targeting-00111-00121"
# conditions=("non-targeting_00004_00017" V"non-targeting_00010_00031" V "non-targeting_00035_00050" V "non-targeting_00053_00059" V "non-targeting_00111_00121")
BASE_DIR="/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen/sorted/batch1/C9"
DATE=$(date)
LOG_FILE="/home/projects/hornsteinlab/giliwo/NOVA/tools/images_organizer/FuNOVA_screen/merge_non_targeting_dirs_${ref_condition}_to_${alt_condition}.log"


DRY_RUN=false  # set to false to actually run

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "$@"
    else
        "$@"
    fi
}

echo "Merging non-targeting dirs: ${ref_condition} → ${alt_condition}" | tee -a "$LOG_FILE"
echo "date: $DATE" | tee -a "$LOG_FILE"
echo "dry_run: $DRY_RUN"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" | tee -a "$LOG_FILE"

for panel_num in "${panels[@]}"; do
        echo "Panel: ${panel_num}" | tee -a "$LOG_FILE"
        if [ ! -d "${BASE_DIR}/Panel${panel_num}" ]; then
            echo "SKIP (missing): ${BASE_DIR}/Panel${panel_num}" | tee -a "$LOG_FILE"
            continue
        fi

        for plate_num in "${plates[@]}"; do
            echo "Plate: ${plate_num}" | tee -a "$LOG_FILE"

            ref_name="${BASE_DIR}/Panel${panel_num}/${ref_condition}${plate_num}"
            alt_name="${BASE_DIR}/Panel${panel_num}/${alt_condition}${plate_num}"

            # check if ref dir does not exist
            if [ ! -d "$ref_name" ]; then
                echo "Skipping: ${ref_name} does not exist" | tee -a "$LOG_FILE"
                continue
            fi
            # check if alt dir does not exist
            if [ ! -d "$alt_name" ]; then
                echo "${alt_name} does not exist, creating one" | tee -a "$LOG_FILE"
                run_cmd mkdir -p "${alt_name}"
            fi 
            # move all subdirs from ref to alt
            echo "Merging ${ref_name} into ${alt_name}" | tee -a "$LOG_FILE"

            for rep in "${reps[@]}"; do
                echo "==== rep: $rep ====" | tee -a "$LOG_FILE"

                # skip if rep dir under ref doesn't exist
                if [ ! -d "${ref_name}/rep${rep}" ]; then
                    echo "no rep${rep} under ${ref_name}, skipping" | tee -a "$LOG_FILE"
                    continue
                fi

                # skip if no dirs to move -
                if [ -z "$(ls -A "${ref_name}/rep${rep}")" ]; then
                    echo "no dirs to move..." | tee -a "$LOG_FILE"
                    continue
                fi

                # make sure alt name / rep exist
                run_cmd mkdir -p "${alt_name}/rep${rep}"

                for subdir in "$ref_name"/rep"$rep"/*; do
                        subdir_name=$(basename "$subdir")
                        new_subdir="${alt_name}/rep${rep}/${subdir_name}"

                        # check if new subdir already exists
                        if [ -d "$new_subdir" ]; then
                            echo "EXISTS: ${new_subdir}" | tee -a "$LOG_FILE"
                            # remove if empty
                            if [ -z "$(ls -A "$new_subdir")" ]; then
                                echo "---> empty: deleting" | tee -a "$LOG_FILE"
                                run_cmd rmdir "$new_subdir"
                            else 
                                echo "---> non-empty: skipping" | tee -a "$LOG_FILE"
                                continue
                            fi
                        fi

                    

                        echo "Moving ${subdir} → ${new_subdir}" | tee -a "$LOG_FILE"
                        run_cmd mv "$subdir" "$new_subdir"
                done
            done
            
            # delete ref dir if empty
            if ! find "$ref_name" -type f | read; then
                echo "---> no files left in $ref_name: deleting" | tee -a "$LOG_FILE"
                run_cmd rm -r "$ref_name"
            fi
            echo "-----------------------------------------------------" | tee -a "$LOG_FILE"
            
        done
        echo "*********************************************" | tee -a "$LOG_FILE"
done