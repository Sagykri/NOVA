import os
from pathlib import Path
from typing import Union


class PathParts:
    def __init__(self, cell_line_part_indx:int, panel_part_indx:int, condition_part_indx:int,\
                        rep_part_indx:int, marker_part_indx:int):
        self.cell_line_part_indx = cell_line_part_indx
        self.panel_part_indx = panel_part_indx
        self.condition_part_indx = condition_part_indx
        self.rep_part_indx = rep_part_indx
        self.marker_part_indx = marker_part_indx

"""Holds the parts' indexes in the raw path configuration"""
raw_parts = PathParts(cell_line_part_indx=-6, panel_part_indx=-5, condition_part_indx=-4, rep_part_indx=-3, marker_part_indx=-2)

"""Holds the parts' indexes in the processed path configuration"""
processed_parts = PathParts(cell_line_part_indx=-4, panel_part_indx=-3, condition_part_indx=-3, rep_part_indx=0, marker_part_indx=-2)

def get_raw_cell_line(path:Union[str, Path])->str:
    """Get the cell line from a given path to the raw data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The cell line
    """
    return __get_part(path, raw_parts.cell_line_part_indx)
    
def get_raw_panel(path:Union[str, Path])->str:
    """Get the panel from a given path to the raw data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The panel 
    """
    return __get_part(path, raw_parts.panel_part_indx)

def get_raw_condition(path:Union[str, Path])->str:
    """Get the condition from a given path to the raw data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The condition
    """
    return __get_part(path, raw_parts.condition_part_indx)

def get_raw_rep(path:Union[str, Path])->str:
    """Get the rep from a given path to the raw data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The rep 
    """
    return __get_part(path, raw_parts.rep_part_indx)

def get_raw_marker(path:Union[str, Path])->str:
    """Get the marker from a given path to the raw data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The marker name
    """
    return __get_part(path, raw_parts.marker_part_indx)

def get_processed_cell_line(path:Union[str, Path])->str:
    """Get the cell line from a given path to the processed data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The cell line
    """
    return __get_part(path, processed_parts.cell_line_part_indx)
    
def get_processed_panel(path:Union[str, Path])->str:
    """Get the panel from a given path to the processed data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The panel
    """
    filename = get_filename(path)
    filename_split = filename.split('_')
    
    return filename_split[processed_parts.panel_part_indx]

def get_processed_condition(path:Union[str, Path])->str:
    """Get the condition from a given path to the processed data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The condition
    """
    return __get_part(path, processed_parts.condition_part_indx)

def get_processed_rep(path:Union[str, Path])->str:
    """Get the rep from a given path to the processed data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The rep
    """
    filename = get_filename(path)
    filename_split = filename.split('_')
    
    return filename_split[processed_parts.rep_part_indx]

def get_processed_marker(path:Union[str, Path])->str:
    """Get the marker from a given path to the processed data

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The marker name
    """
    return __get_part(path, processed_parts.marker_part_indx)

def get_filename(path:Union[str, Path])->str:
    """Get the filename from a given path

    Args:
        path (Union[str, Path]): The path

    Returns:
        str: The file name
    """
    return os.path.splitext(__get_part(path, -1))[0]

def __get_part(path:Union[str, Path], indx:int)->str:
    """Return the part at given index within the given path (after splitting by os.sep)

    Args:
        path (Union[str, Path]): The path
        indx (int): The part index 

    Returns:
        str: The part of the path
    """
    if type(path) is not Path:
        path = Path(path)
    
    return path.parts[indx]
