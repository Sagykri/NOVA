script="/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/tools/images_organizer/AlyssaCoyne/rename_inline_cellline.sh"
# script2="/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/tools/images_organizer/AlyssaCoyne/rename_inline_cellline2.sh"
root_folder="/home/projects/hornsteinlab/Collaboration/NOVA/input/images/raw/AlyssaCoyne_new/panelC_new"

# Usage: ./bulk_rename.sh /path/to/root_folder subfolder_prefix new_file_prefix

# # Control
$script $root_folder "EDi022_" "control EDi022"
$script $root_folder "EDi029_" "control EDi029"
$script $root_folder "EDi037_" "control EDi037"

# C9Orf72
$script $root_folder "CS2YNL_" "c9 CS2YNL"
$script $root_folder "CS7VCZ_" "c9 CS7VCZ"
$script $root_folder "CS8RFT_" "c9 CS8RFT"

sALS+
$script $root_folder "CS2FN3_" "sALS+ CS2FN3"
$script $root_folder "CS4ZCD_" "sALS+ CS4ZCD"
$script $root_folder "CS7TN6_" "sALS+ CS7TN6"

# sALS-
$script $root_folder "CS0ANK_" "sALS- CS0ANK"
$script $root_folder "CS0JPP_" "sALS- CS0JPP"
$script $root_folder "CS6ZU8_" "sALS- CS6ZU8"