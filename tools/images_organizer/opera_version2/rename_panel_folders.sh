#!/bin/bash

# Check for input
if [ -z "$1" ]; then
    echo "Usage: $0 <root_folder>"
    exit 1
fi

root="$1"

# Ensure the root folder exists
if [ ! -d "$root" ]; then
    echo "Error: Directory '$root' does not exist."
    exit 1
fi

# Find all matching directories recursively
find "$root" -type d -name 'flexp_B*_panel*' | while read -r dir; do
    base=$(basename "$dir")
    parent=$(dirname "$dir")

    # Extract the panel name (e.g., panelA)
    panel_name=$(echo "$base" | grep -o 'panel[A-Z]')

    # Safety check
    if [ -n "$panel_name" ]; then
        target="$parent/$panel_name"

        # Avoid overwriting existing directories
        if [ -e "$target" ]; then
            echo "Skipping $dir → $target (already exists)"
        else
            echo "Renaming $dir → $target"
            mv "$dir" "$target"
        fi
    fi
done
