#!/bin/bash

# Usage: ./move_files_up.sh /path/to/root_folder

root="$1"

if [[ -z "$root" ]]; then
    echo "Usage: $0 /path/to/root_folder"
    exit 1
fi

for subfolder in "$root"/*/; do
    [[ -d "$subfolder" ]] || continue  # Skip if not a directory
    echo "Moving files from $subfolder to $root"

    for file in "$subfolder"*; do
        [[ -f "$file" ]] || continue
        mv -- "$file" "$root"
        echo "Moved: $file -> $root"
    done
done
