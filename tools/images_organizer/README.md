
██╗███╗░░░███╗░█████╗░░██████╗░███████╗░██████╗
██║████╗░████║██╔══██╗██╔════╝░██╔════╝██╔════╝
██║██╔████╔██║███████║██║░░██╗░█████╗░░╚█████╗░
██║██║╚██╔╝██║██╔══██║██║░░╚██╗██╔══╝░░░╚═══██╗
██║██║░╚═╝░██║██║░░██║╚██████╔╝███████╗██████╔╝
╚═╝╚═╝░░░░░╚═╝╚═╝░░╚═╝░╚═════╝░╚══════╝╚═════╝░

░█████╗░██████╗░░██████╗░░█████╗░███╗░░██╗██╗███████╗███████╗██████╗░
██╔══██╗██╔══██╗██╔════╝░██╔══██╗████╗░██║██║╚════██║██╔════╝██╔══██╗
██║░░██║██████╔╝██║░░██╗░███████║██╔██╗██║██║░░███╔═╝█████╗░░██████╔╝
██║░░██║██╔══██╗██║░░╚██╗██╔══██║██║╚████║██║██╔══╝░░██╔══╝░░██╔══██╗
╚█████╔╝██║░░██║╚██████╔╝██║░░██║██║░╚███║██║███████╗███████╗██║░░██║
░╚════╝░╚═╝░░╚═╝░╚═════╝░╚═╝░░╚═╝╚═╝░░╚══╝╚═╝╚══════╝╚══════╝╚═╝░░╚═╝


This tool organizes the Bioimg folder structure to MOmaps structure.
(Version: 1.0, Published: 02.05.23)

## How to use it:
- Grab the files from bioimg (see details below)
- Adjust the conig file to your needs. Don't forget set the input (SRC_ROOT_PATH) and output (DST_ROOT_PATH) paths correctly.
- If the output folder doesn't exist, you must create it
- Run: python main.py
- Delete the unorganized folder you pulled from bioimg (the one in Wexac, not the one in bioimg!)

## batch_3_6_version:
Please use that version when handling batches 3 to 6.
Since batches 3 to 6 have some differences from later batches, that folder holds a customized version.
The changes are:
- Reps are no longer separated to folders (in the raw data)
- Today there is no _ (underscore) in the folder name between panel and its letter

## How to grab files from bioimg via Wexac:
- Login to access4 (no other access is an option)
- cd to your destination folder
- Login to bioimg using samba. 
    Run:
    -- smbclient -U USERNAME //bioimg.weizmann.ac.il/USERNAME
    -- *Enter you password*
    -- *cd your source folder*
    -- recurse on
    -- propmpt off
    -- mget *


