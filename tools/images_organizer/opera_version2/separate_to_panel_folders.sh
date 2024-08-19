# Check if at least three arguments are given (source folder, destination folder, and at least one pattern)
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 source_directory destination_directory pattern1 [pattern2 ...]"
    exit 1
fi

# Assign the first two arguments to variables for readability
SOURCE_DIR=$1
DESTINATION_DIR=$2

# Shift the first two arguments to leave only the patterns
shift 2

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Check if destination directory exists, create it if it does not
if [ ! -d "$DESTINATION_DIR" ]; then
    mkdir -p "$DESTINATION_DIR"
fi


echo "Source directory: $SOURCE_DIR Destination directory: $DESTINATION_DIR"

# Prepare an array to hold the find command arguments
FIND_ARGS=(-maxdepth 2 -type f)

# Append each pattern as an OR condition
for pattern in "$@"; do
    FIND_ARGS+=(-name "$pattern")
done

FIND_ARGS=("${FIND_ARGS[@]}")

# Execute find command and move files
find "$SOURCE_DIR" "${FIND_ARGS[@]}" -exec mv {} "$DESTINATION_DIR" \;

echo "Files moved successfully."
