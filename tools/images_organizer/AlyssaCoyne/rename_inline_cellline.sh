#!/bin/bash

# Usage: ./bulk_rename.sh /path/to/root_folder subfolder_prefix new_file_prefix

root="$1"
subfolder_prefix="$2"
new_file_prefix="$3"

if [[ -z "$root" || -z "$subfolder_prefix" || -z "$new_file_prefix" ]]; then
    echo "Usage: $0 /path/to/root_folder subfolder_prefix new_file_prefix"
    exit 1
fi

# Go through each subfolder matching prefix
for dir in "$root"/"$subfolder_prefix"*; do
    if [[ -d "$dir" ]]; then
        echo "Processing folder: $dir"
        shopt -s nullglob
        for file in "$dir"/*; do
            filename=$(basename "$file")
            if [[ "$filename" == *_* ]]; then
                suffix="${filename#*_}"
                new_name="${new_file_prefix}_${suffix}"
                new_path="${dir}/${new_name}"
                if [[ "$filename" != "$new_name" ]]; then
                    mv -- "$file" "$new_path"
                    echo "Renamed: $file -> $new_path"
                fi
            fi
        done
    fi
done
