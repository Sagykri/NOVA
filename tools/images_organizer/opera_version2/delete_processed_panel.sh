#!/bin/bash

FOLDER=/home/projects/hornsteinlab/Collaboration/NOVA/input/images/processed/ManuscriptFinalData_80pct/neuronsDay8_new/batch1
PANEL_TO_DELETE=panelK

find "$FOLDER" -type f -name "*_${PANEL_TO_DELETE}_*.npy" -delete
