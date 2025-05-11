# Check if a folder path is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <root_folder_path>"
  exit 1
fi

# Get the root folder path from the first command-line argument
root_folder_path="$1"

# Check if the provided path is a directory
if [ ! -d "$root_folder_path" ]; then
  echo "Error: Provided path is not a directory."
  exit 1
fi

# Find all files in the root folder and its subfolders
find "$root_folder_path" -type f | while read file; do
  # Extract the filename from the path
  filename=$(basename "$file")
  # Skip files that contain an underscore
  if [[ "$filename" == *_* ]]; then
    continue
  fi
  # Extract the directory path
  dir_path=$(dirname "$file")
  # Extract the prefix (first part before the first '-')
  prefix=$(echo "$filename" | cut -d'-' -f1)
  # Construct the new filename with the postfix
  new_filename="${filename%.*}_${prefix}.${filename##*.}"
  # Rename (move) the file with the new name in its respective folder
  mv "$file" "$dir_path/$new_filename"
done


# example:
# '/home/projects/hornsteinlab/Collaboration/MOmaps/tools/images_organizer/opera_version/add_site.sh' /home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/Opera_sorted/batch1/