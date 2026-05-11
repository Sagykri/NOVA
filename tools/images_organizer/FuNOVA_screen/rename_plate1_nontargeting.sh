
panels=("1" "2" "3" "4")
conditions=("non-targeting_00004_00017" "non-targeting_00010_00031" "non-targeting_00035_00050" "non-targeting_00053_00059" "non-targeting_00111_00121" "Empty")
BASE_DIR="/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen/sorted/batch1/C9"

DRY_RUN=true  # set to false to actually run

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "$@"
    else
        "$@"
    fi
}

for panel_num in "${panels[@]}"; do
    for condition in "${conditions[@]}"; do

        old_name="${BASE_DIR}/Panel${panel_num}/${condition}"
        new_name="${BASE_DIR}/Panel${panel_num}/${condition}_p1"

        # check if old dir exists
        if [ ! -d "$old_name" ]; then
            echo "Skipping: ${old_name} does not exist"
            continue
        fi

        # handle existing new dir (check only for files)
        if [ -d "$new_name" ]; then
            if find "$new_name" -type f -print -quit | grep -q .; then
                echo "ERROR: ${new_name} exists and contains files"
                exit 1
            else
                echo "Deleting ${new_name} (no files inside)"
                run_cmd rm -rf "$new_name"
            fi
        fi

        # rename
        echo "Renaming ${old_name} → ${new_name}"
        run_cmd mv "$old_name" "$new_name"

        echo "*********************************************"

    done
done