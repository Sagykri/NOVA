import os
import sys


sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
import pandas as pd
import logging
from functools import partial
from typing import List, Tuple, Callable, Optional, Union
from enum import Enum

from src.datasets.dataset_config import DatasetConfig
from src.common.utils import get_if_exists
from src.figures.plot_config import PlotConfig

MARKER_IDX = 0
CELL_LINE_IDX = 1
CONDITION_IDX = 2
BATCH_IDX = 3
REP_IDX = 4

CELL_LINE_IDX_MULTIPLEX = 0
CONDITION_IDX_MULTIPLEX = 1
BATCH_IDX_MULTIPLEX = 2
REP_IDX_MULTIPLEX = 3

class LabelInfo:
    """Holds all the needed info on the label
    """
    def __init__(self, label:str, dataset_config:DatasetConfig, index:int=-1):
        __labels_np = np.asarray([label])
        self.batch:str = get_batches_from_labels(__labels_np, dataset_config)[0]
        self.cell_line_cond:str = get_cell_lines_conditions_from_labels(__labels_np, dataset_config)[0]
        self.marker:str = get_markers_from_labels(__labels_np, dataset_config)[0]
        self.rep:str = get_reps_from_labels(__labels_np, dataset_config)[0]
        self.index:int = index

def get_rep_idx(dataset_config, multiplex):
    rep_idx = REP_IDX_MULTIPLEX if multiplex else REP_IDX
    return rep_idx - int(not dataset_config.ADD_CONDITION_TO_LABEL) \
                    - int(not dataset_config.ADD_LINE_TO_LABEL) \
                    - int(not dataset_config.ADD_BATCH_TO_LABEL)

def get_condition_idx(dataset_config, multiplex):
    condition_idx = CONDITION_IDX_MULTIPLEX if multiplex else CONDITION_IDX
    return condition_idx - int(not dataset_config.ADD_LINE_TO_LABEL)

def get_batch_idx(dataset_config, multiplex):
    batch_idx = BATCH_IDX_MULTIPLEX if multiplex else BATCH_IDX
    return batch_idx - int(not dataset_config.ADD_CONDITION_TO_LABEL) \
                     - int(not dataset_config.ADD_LINE_TO_LABEL)

def get_cell_line_idx(dataset_config, multiplex):
    return CELL_LINE_IDX_MULTIPLEX if multiplex else CELL_LINE_IDX


def extract_parts_from_label(label: str, indices:Tuple[int, int]) -> str:
    return '_'.join(label.split("_")[indices[0]:indices[1]])

def get_parts_from_labels(labels: np.ndarray[str], indices: Tuple[int, int]) -> np.ndarray[str]:
    vectorized_extraction = np.vectorize(partial(extract_parts_from_label, indices=indices))
    parts = vectorized_extraction(labels)
    return parts


def get_cell_lines_conditions_batch_reps_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig) -> np.ndarray[str]:
    if not dataset_config.ADD_LINE_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_LINE_TO_LABEL is FALSE, cannot extract cell lines from labels!')
        return None
    if not dataset_config.ADD_CONDITION_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_CONDITION_TO_LABEL is FALSE, cannot extract conditions from labels!')
        return None
    result = get_parts_from_labels(labels=labels, indices=(CELL_LINE_IDX,None))
    return result


def get_markers_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig=None) -> np.ndarray[str]:
    markers = get_parts_from_labels(labels=labels, indices=(MARKER_IDX, MARKER_IDX+1))
    return markers

def opencell_map(labels: np.ndarray[str], dataset_config:DatasetConfig):
    markers = get_markers_from_labels(labels, dataset_config)
    mapping = pd.read_csv(os.path.join(os.getenv("NOVA_DATA_HOME"), 'gt_tables', 'opencell_mappings_41592_2022_1541_MOESM4_ESM.csv'))
    mapping_dict = dict(zip(mapping['gene_name'], mapping['localization']))

    converted_labels = pd.Series(markers).map(mapping_dict).fillna('Unknown').to_numpy()

    return converted_labels



def get_reps_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig, multiplex:bool = False) -> np.ndarray[str]:
    if not dataset_config.ADD_REP_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_REP_TO_LABEL is FALSE, cannot extract reps from labels!')
        return None

    rep_idx = get_rep_idx(dataset_config, multiplex)
    reps = get_parts_from_labels(labels=labels, indices=(rep_idx, rep_idx+1))
    return reps

def get_conditions_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig, multiplex:bool = False) -> np.ndarray[str]:
    if not dataset_config.ADD_CONDITION_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_CONDITION_TO_LABEL is FALSE, cannot extract condition from labels!')
        return None
    
    condition_idx = get_condition_idx(dataset_config, multiplex)
    conditions = get_parts_from_labels(labels=labels, indices=(condition_idx, condition_idx+1))
    return conditions

def get_batches_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig, multiplex:bool = False) -> np.ndarray[str]:
    if not dataset_config.ADD_BATCH_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_BATCH_TO_LABEL is FALSE, cannot extract batches from labels!')
        return None

    batch_idx = get_batch_idx(dataset_config, multiplex)
    batches = get_parts_from_labels(labels=labels, indices=(batch_idx, batch_idx+1))
    return batches

def get_cell_lines_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig, multiplex:bool = False) -> np.ndarray[str]:
    if not dataset_config.ADD_LINE_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_LINE_TO_LABEL is FALSE, cannot extract cell lines from labels!')
        return None

    cell_line_idx = get_cell_line_idx(dataset_config, multiplex)
    cell_lines = get_parts_from_labels(labels=labels, indices=(cell_line_idx, cell_line_idx+1))

    return cell_lines

def get_cell_lines_conditions_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig, multiplex:bool = False) -> np.ndarray[str]:
    if not dataset_config.ADD_LINE_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_LINE_TO_LABEL is FALSE, cannot extract cell lines from labels!')
        return None
    if not dataset_config.ADD_CONDITION_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_CONDITION_TO_LABEL is FALSE, cannot extract conditions from labels!')
        return None

    condition_idx = get_condition_idx(dataset_config, multiplex)

    cell_line_idx = get_cell_line_idx(dataset_config, multiplex)

    cell_line_conditions = get_parts_from_labels(labels=labels, indices=(cell_line_idx,condition_idx+1))
    return cell_line_conditions

def get_cell_lines_condition_batches_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig, multiplex:bool = False) -> np.ndarray[str]:
    if not dataset_config.ADD_LINE_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_LINE_TO_LABEL is FALSE, cannot extract cell lines from labels!')
        return None
    if not dataset_config.ADD_CONDITION_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_CONDITION_TO_LABEL is FALSE, cannot extract conditions from labels!')
        return None
    if not dataset_config.ADD_BATCH_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_BATCH_TO_LABEL is FALSE, cannot extract batches from labels!')
        return None

    batch_idx = get_batch_idx(dataset_config, multiplex)
    cell_line_idx = get_cell_line_idx(dataset_config, multiplex)

    cell_line_conditions_batches = get_parts_from_labels(labels=labels, indices=(cell_line_idx,batch_idx+1))
    return cell_line_conditions_batches


def get_unique_parts_from_labels(labels:np.ndarray[str], get_part_function:Callable[[np.ndarray[str]], np.ndarray[str]], dataset_config:Optional[DatasetConfig]=None)->np.ndarray[str]:
    if dataset_config is None:
        return np.unique(get_part_function(labels))
    else:
        return np.unique(get_part_function(labels, dataset_config))


def get_batches_from_input_folders(input_folders:List[str])->List[str]:
    batches = [folder.split(os.sep)[-1].split('_')[0] for folder in input_folders]
    return batches


def edit_labels_by_config(labels:np.ndarray[str], dataset_config:DatasetConfig, multiplex:bool = False)->np.ndarray[str]:
    vectorized_edit = np.vectorize(partial(edit_label_by_config, dataset_config=dataset_config, multiplex=multiplex))
    labels = vectorized_edit(labels)
    return labels


def edit_label_by_config(label:str, dataset_config:DatasetConfig, multiplex:bool = False)->str:
    label_parts = label.split('_')
    label_parts_new = [] if multiplex else [label_parts[0]]

    cell_line_idx = get_cell_line_idx(dataset_config, multiplex)
    condition_idx = get_condition_idx(dataset_config, multiplex)
    batch_idx = get_batch_idx(dataset_config, multiplex)
    rep_idx = get_rep_idx(dataset_config, multiplex)

    if dataset_config.ADD_LINE_TO_LABEL:
        cell_line = label_parts[cell_line_idx]
        remove_patient_id = get_if_exists(dataset_config, 'REMOVE_PATIENT_ID_FROM_CELL_LINE', False)
        if remove_patient_id:
            cell_line = cell_line.split('-')[0]
        label_parts_new.append(cell_line)
    if dataset_config.ADD_CONDITION_TO_LABEL:
        label_parts_new.append(label_parts[condition_idx])
    if dataset_config.ADD_BATCH_TO_LABEL:
        label_parts_new.append(label_parts[batch_idx])
    if dataset_config.ADD_REP_TO_LABEL:
        label_parts_new.append(label_parts[rep_idx])
    return '_'.join(label_parts_new)


def split_markers_from_labels(labels:np.ndarray[str], dataset_config:DatasetConfig)->Tuple[np.ndarray[str],np.ndarray[str]]:
    markers = get_markers_from_labels(labels)
    rest_of_labels = get_parts_from_labels(labels=labels, indices=(MARKER_IDX+1,None))

    return markers, rest_of_labels

def remove_markers(labels:np.ndarray[str], dataset_config:DatasetConfig)->np.ndarray[str]:
    _, rest_of_labels = split_markers_from_labels(labels, dataset_config)
    return rest_of_labels

def remove_patient_id_from_cell_line_multiplex(labels:np.ndarray[str], dataset_config:DatasetConfig)->np.ndarray[str]:
    
    cell_lines_with_patient_id = get_cell_lines_from_labels(labels, dataset_config, multiplex= True)

    print(f'Removing patient ID from cell lines: {cell_lines_with_patient_id}')

    cell_lines = np.array([cell_line.split('-')[0] for cell_line in cell_lines_with_patient_id])

    print(f'Cell lines after removing patient ID: {cell_lines}')

    return cell_lines


# --- Picklable wrappers for MULTIPLEX functions ---
def multiplex_reps(labels, dataset_config):
    return get_reps_from_labels(labels, dataset_config, multiplex=True)

def multiplex_conditions(labels, dataset_config):
    return get_conditions_from_labels(labels, dataset_config, multiplex=True)

def multiplex_cell_lines(labels, dataset_config):
    return get_cell_lines_from_labels(labels, dataset_config, multiplex=True)

def multiplex_batches(labels, dataset_config):
    return get_batches_from_labels(labels, dataset_config, multiplex=True)    

def multiplex_cell_lines_conditions(labels, dataset_config):
    return get_cell_lines_conditions_from_labels(labels, dataset_config, multiplex=True)

def multiplex_cell_lines_condition_batches(labels, dataset_config):
    return get_cell_lines_condition_batches_from_labels(labels, dataset_config, multiplex=True)


# ----------------------------------------------------
class MapLabelsFunction(Enum):
    MARKERS = (get_markers_from_labels,)
    CONDITIONS = (get_conditions_from_labels,)
    CELL_LINES = (get_cell_lines_from_labels,)
    CELL_LINES_CONDITIONS = (get_cell_lines_conditions_from_labels,)
    REPS = (get_reps_from_labels,)
    CELL_LINES_CONDITIONS_BATCH_REPS = (get_cell_lines_conditions_batch_reps_from_labels,)

    # Picklable MULTIPLEX functions
    MULTIPLEX_CONDITIONS = (multiplex_conditions,)
    MULTIPLEX_CELL_LINES = (multiplex_cell_lines,)
    MULTIPLEX_CELL_LINES_CONDITIONS = (multiplex_cell_lines_conditions,)
    MULTIPLEX_CELL_LINES_CONDITIONS_BATCHES = (multiplex_cell_lines_condition_batches,)
    MULTIPLEX_REMOVE_PATIENT_ID_FROM_CELL_LINE = (remove_patient_id_from_cell_line_multiplex, )

    REMOVE_MARKER = (remove_markers,)
    OPENCELL = (opencell_map,)

def map_labels(labels: np.ndarray[str], config_plot: Union[PlotConfig, DatasetConfig],
                config_data: DatasetConfig, config_function_name:str = 'MAP_LABELS_FUNCTION') -> np.ndarray[str]:
    """Maps labels based on the provided function in the configuration."""
    map_function_name:str = get_if_exists(config_plot, config_function_name, None)
    
    if map_function_name:
        map_function = MapLabelsFunction[map_function_name].value[0]
        return map_function(labels, config_data)
    return labels
