
conditions=("non-targeting_00004_00017" "non-targeting_00010_00031" "non-targeting_00035_00050" "non-targeting_00053_00059" "non-targeting_00111_00121" "Empty" "Ranbp17" "TDP-43")
BASE_DIR="/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/FuNOVA_Screen/batch1/C9"


DRY_RUN=false  # set to false to actually run

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "$@"
    else
        "$@"
    fi
}

for condition in "${conditions[@]}"; do

            old_name="${BASE_DIR}/${condition}"
            new_name="${BASE_DIR}/${condition}_old"

            # check if old dir does not exist
            if [ ! -d "$old_name" ]; then
                echo "Skipping: ${old_name} does not exist"
                echo "*********************************************"
                continue
            fi

            # rename
            echo "Renaming ${old_name} → ${new_name}"
            run_cmd mv "$old_name" "$new_name"

            echo "*********************************************"

done