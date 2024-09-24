import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
import logging
from functools import partial
from typing import List, Tuple, Callable, Optional
from src.datasets.dataset_config import DatasetConfig

MARKER_IDX = 0
CELL_LINE_IDX = 1
CONDITION_IDX = 2
BATCH_IDX = 3
REP_IDX = 4

def extract_parts_from_label(label: str, indices:Tuple[int, int]) -> str:
    return '_'.join(label.split("_")[indices[0]:indices[1]])

def get_parts_from_labels(labels: np.ndarray[str], indices: Tuple[int, int]) -> np.ndarray[str]:
    vectorized_extraction = np.vectorize(partial(extract_parts_from_label, indices=indices))
    parts = vectorized_extraction(labels)
    return parts


def get_cell_lines_conditions_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig) -> np.ndarray[str]:
    if not dataset_config.ADD_LINE_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_LINE_TO_LABEL is FALSE, cannot extract cell lines from labels!')
        return None
    if not dataset_config.ADD_CONDITION_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_CONDITION_TO_LABEL is FALSE, cannot extract conditions from labels!')
        return None
    condition_idx = CONDITION_IDX - int(not dataset_config.ADD_LINE_TO_LABEL)
    cell_line_conditions = get_parts_from_labels(labels=labels, indices=(CELL_LINE_IDX,condition_idx+1))
    return cell_line_conditions


def get_cell_lines_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig) -> np.ndarray[str]:
    if not dataset_config.ADD_LINE_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_LINE_TO_LABEL is FALSE, cannot extract cell lines from labels!')
        return None
    cell_lines = get_parts_from_labels(labels=labels, indices=(CELL_LINE_IDX, CELL_LINE_IDX+1))
    return cell_lines


def get_conditions_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig) -> np.ndarray[str]:
    if not dataset_config.ADD_CONDITION_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_CONDITION_TO_LABEL is FALSE, cannot extract condition from labels!')
        return None
    condition_idx = CONDITION_IDX - int(not dataset_config.ADD_LINE_TO_LABEL)
    conditions = get_parts_from_labels(labels=labels, indices=(condition_idx, condition_idx+1))
    return conditions


def get_markers_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig=None) -> np.ndarray[str]:
    markers = get_parts_from_labels(labels=labels, indices=(MARKER_IDX, MARKER_IDX+1))
    return markers


def get_batches_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig) -> np.ndarray[str]:
    if not dataset_config.ADD_BATCH_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_BATCH_TO_LABEL is FALSE, cannot extract batches from labels!')
        return None
    batch_idx = BATCH_IDX - int(not dataset_config.ADD_CONDITION_TO_LABEL) -int(not dataset_config.ADD_LINE_TO_LABEL)
    batches = get_parts_from_labels(labels=labels, indices=(batch_idx, batch_idx+1))
    return batches


def get_reps_from_labels(labels: np.ndarray[str], dataset_config:DatasetConfig) -> np.ndarray[str]:
    if not dataset_config.ADD_REP_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_REP_TO_LABEL is FALSE, cannot extract reps from labels!')
        return None
    rep_idx = REP_IDX - int(not dataset_config.ADD_CONDITION_TO_LABEL) -int(not dataset_config.ADD_LINE_TO_LABEL) -int(not dataset_config.ADD_BATCH_TO_LABEL)
    reps = get_parts_from_labels(labels=labels, indices=(rep_idx, rep_idx+1))
    return reps

def get_conditions_from_multiplex_labels(labels: np.ndarray[str], dataset_config:DatasetConfig) -> np.ndarray[str]:
    if not dataset_config.ADD_CONDITION_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_CONDITION_TO_LABEL is FALSE, cannot extract condition from labels!')
        return None
    condition_idx = CONDITION_IDX - int(not dataset_config.ADD_LINE_TO_LABEL) -1
    conditions = get_parts_from_labels(labels=labels, indices=(condition_idx, condition_idx+1))
    return conditions

def get_cell_lines_from_multiplex_labels(labels: np.ndarray[str], dataset_config:DatasetConfig) -> np.ndarray[str]:
    if not dataset_config.ADD_LINE_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_LINE_TO_LABEL is FALSE, cannot extract condition from labels!')
        return None
    cell_line_idx = CELL_LINE_IDX -1
    cell_lines = get_parts_from_labels(labels=labels, indices=(cell_line_idx, cell_line_idx+1))
    return cell_lines

def get_cell_lines_conditions_from_multiplex_labels(labels: np.ndarray[str], dataset_config:DatasetConfig) -> np.ndarray[str]:
    if not dataset_config.ADD_LINE_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_LINE_TO_LABEL is FALSE, cannot extract cell lines from labels!')
        return None
    if not dataset_config.ADD_CONDITION_TO_LABEL:
        logging.warning(f'DatasetConfig.ADD_CONDITION_TO_LABEL is FALSE, cannot extract conditions from labels!')
        return None
    condition_idx = CONDITION_IDX - int(not dataset_config.ADD_LINE_TO_LABEL) -1
    cell_line_idx = CELL_LINE_IDX -1
    cell_line_conditions = get_parts_from_labels(labels=labels, indices=(cell_line_idx,condition_idx+1))
    return cell_line_conditions


def get_unique_parts_from_labels(labels:np.ndarray[str], get_part_function:Callable[[np.ndarray[str]], np.ndarray[str]], dataset_config:Optional[DatasetConfig]=None)->np.ndarray[str]:
    if dataset_config is None:
        return np.unique(get_part_function(labels))
    else:
        return np.unique(get_part_function(labels, dataset_config))


def get_batches_from_input_folders(input_folders:List[str])->List[str]:
    batches = [folder.split(os.sep)[-1].split('_')[0] for folder in input_folders]
    return batches


def edit_labels_by_config(labels:np.ndarray[str], dataset_config:DatasetConfig)->np.ndarray[str]:
    vectorized_edit = np.vectorize(partial(edit_label_by_config, dataset_config=dataset_config))
    labels = vectorized_edit(labels)
    return labels


def edit_label_by_config(label:str, dataset_config:DatasetConfig)->str:
    label_parts = label.split('_')
    label_parts_new = [label_parts[0]]
    if dataset_config.ADD_LINE_TO_LABEL:
        label_parts_new.append(label_parts[CELL_LINE_IDX])
    if dataset_config.ADD_CONDITION_TO_LABEL:
        label_parts_new.append(label_parts[CONDITION_IDX])
    if dataset_config.ADD_BATCH_TO_LABEL:
        label_parts_new.append(label_parts[BATCH_IDX])
    if dataset_config.ADD_REP_TO_LABEL:
        label_parts_new.append(label_parts[REP_IDX])
    return '_'.join(label_parts_new)

def split_markers_from_labels(labels:np.ndarray[str], dataset_config:DatasetConfig)->Tuple[np.ndarray[str],np.ndarray[str]]:
    markers = get_markers_from_labels(labels)
    rest_of_labels = get_parts_from_labels(labels=labels, indices=(MARKER_IDX+1,None))

    return markers, rest_of_labels