"""
dNLS Pbody CellProfiler pipeline execution script.
Runs the CellProfiler pipeline for analyzing dNLS Pbody data.
"""

# ===== Packages =====
import io
import os
import sys
import atexit

from datetime import datetime
from functools import partial
import multiprocessing as mp
from functools import partial

	
import logging
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

import cell_profiler_utils as cp_utils  

# ====================

# Allow imports from project tree

BASE_DIR = os.path.join('/home', 'projects', 'hornsteinlab', 'Collaboration', 'NOVA')
sys.path.insert(1, BASE_DIR)

# ----- Config -----


input_folder_name   = 'NIH'
marker_target       = ['DAPI']  # list of markers to include; Empty list or None for all
cell_line           = ["WT"]
output_folder_name  = "unbiased_analysis_NIH"
pipeline_name       = '250930_CellProfiler_unbiased-analysis_minimal_onlyDAPI.cppipe'
Batch_name = "batch3"
panel_target = ['panelH', 'panelI', 'panelJ', 'panelK'] # Empty list for all
condition_target = ['Untreated']

DATA_INPUT_DIR = os.path.join(
    BASE_DIR, 'cell_profiler', 'outputs', 'filtered_by_brenner_post_rescale_outputs', input_folder_name
)
pipeline_path = os.path.join(
    BASE_DIR, 'cell_profiler', 'pipelines', pipeline_name
)

# Paths to local directories
OUTPUT_DIR   = os.path.join(BASE_DIR, 'cell_profiler', 'outputs', 'cell_profiler_RUNS', output_folder_name)
LOG_DIR_PATH = os.path.join(BASE_DIR, 'cell_profiler', 'logs')


# ------------------


def _find_items(input_data_batch_dir):
    """Return list of (input_dir, output_dir) pairs per marker."""
    return cp_utils.find_marker_folders(
        batch_path=input_data_batch_dir,
        output_dir=OUTPUT_DIR,
        depth=5,
        markers_to_include=marker_target,
        include_nucleus=True
    )

def _filter_items(items, line=None, panel=None, condition=None):
    """
    Keep only items whose marker folder path matches the requested filters.
    Layout: batch / line / panel / condition / rep / marker
    For input_dir == .../rep/marker:
        p.parents[3].name == line
        p.parents[2].name == panel
        p.parents[1].name == condition
    Each filter arg can be:
        - list/tuple/set of allowed values
        - single string (treated as one allowed value)
        - None or empty list => no filtering on that dimension
    """
    # Normalize filters into sets (or None)
    def _norm(x):
        if x is None: return None
        if isinstance(x, (list, tuple, set)):
            vals = [str(v).strip() for v in x if v is not None and str(v).strip() != ""]
            return set(vals) if vals else None
        s = str(x).strip()
        return {s} if s else None

    line_set      = _norm(line)
    panel_set     = _norm(panel)
    condition_set = _norm(condition)

    if not any([line_set, panel_set, condition_set]):
        return items  # nothing to filter

    filtered = []
    for inp, outp in items:
        p = Path(inp)
        try:
            line_name      = p.parents[3].name      # rep(0) -> condition(1) -> panel(2) -> line(3)
            panel_name     = p.parents[2].name
            condition_name = p.parents[1].name
        except IndexError:
            # path is shallower than expected; skip safely
            continue

        if line_set and line_name not in line_set:
            continue
        if panel_set and panel_name not in panel_set:
            continue
        if condition_set and condition_name not in condition_set:
            continue

        filtered.append((inp, outp))

    logging.info("Filtering — line: %s | panel: %s | condition: %s",
                 sorted(line_set) if line_set else "ALL",
                 sorted(panel_set) if panel_set else "ALL",
                 sorted(condition_set) if condition_set else "ALL")
    return filtered

def _worker_init(log_queue):
    # Route Python logging to the queue
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(QueueHandler(log_queue))
    # Route print/stdout/stderr (e.g., CP prints) into logging
    class _StreamToLogger(io.TextIOBase):
        def __init__(self, level): self.level = level
        def write(self, buf):
            buf = str(buf)
            if buf.strip():
                for line in buf.rstrip().splitlines():
                    logging.log(self.level, line)
        def flush(self): pass
    sys.stdout = _StreamToLogger(logging.INFO)
    sys.stderr = _StreamToLogger(logging.ERROR)
    
    # Ensure CP is ready in each worker
    cp_utils.init_cell_profiler(pipeline_path=pipeline_path, max_heap='512m')

def main(input_data_batch_dir, use_multiprocessing, batch_name, dataset_name):

    logging.info(f"\n\nStarting to run Cell Profiler pipeline on batch: {dataset_name} {batch_name}")
    logging.info(
        f"Run name: {output_folder_name}, input_data_batch_dir: {input_data_batch_dir}, "
        f"Pipeline: {pipeline_name}, marker_target: {marker_target}, "
        f"cell_line_target: {cell_line}, panel_target: {panel_target}, "
        f"condition_target: {condition_target}, FinalFileLocation: {output_folder_name}"
    )

    items = list(_find_items(input_data_batch_dir))

    logging.info('Starting unified filter (line/panel/condition)')
    items = _filter_items(items, line=cell_line, panel=panel_target, condition=condition_target)

    if not items:
        logging.warning("No marker folders found after filtering. Check batch/cell_line/marker settings.")
        return None
    
    if use_multiprocessing:
        
        # set spawn once (safe if already set)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

        ctx = mp.get_context("spawn")
        
        # Use the parent's existing handlers (file + console) as QueueListener targets
        parent_handlers = logging.getLogger().handlers
        log_queue = ctx.Queue()
        listener = QueueListener(log_queue, *parent_handlers, respect_handler_level=True)
        listener.start()
        
        logging.info("Running in multiprocess mode")

        # Make a picklable callable for workers
        func = partial(
            cp_utils.run_cell_profiler_pipeline,
            pipeline_path=pipeline_path,
            dataset_name=dataset_name,
            debug= False, # set False for full run
            unbiased = True,
            max_sets = 10
        )

        try:
            with ctx.Pool(processes=4,
                        initializer=_worker_init, # starts JVM+pipeline once per worker
                        initargs=(log_queue,),
                        maxtasksperchild=20) as pool:
                for result in pool.imap_unordered(func, items, chunksize=1):
                    logging.info(result)
        finally:
            listener.stop()

        logging.info("All workers finished (JVMs stop via atexit).")

    else:
        logging.info("Running in uniprocess mode")
        # Starts JVM once + loads pipeline once; reused by run_cell_profiler_pipeline()
        cp_utils.init_cell_profiler(pipeline_path=pipeline_path, max_heap='2g')
        for paths in items:
            logging.info(f"Running Cell Profiler on paths: {paths}")
            res = cp_utils.run_cell_profiler_pipeline(
                paths, pipeline_path=pipeline_path, dataset_name=dataset_name
            )
            logging.info(res)
        
        # No explicit JAVA stop needed (atexit handles it)

    return None

if __name__ == '__main__':
    try:
        
        # set True when you want the Pool multiprocessing version
        use_multiprocessing = True  
        
        # One log file for the whole run
        cp_utils.set_logging(
            log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M'))
        )

        # Iterate over batches
        for BATCH_TO_RUN in [Batch_name]:  
            data_input_dir_batch = os.path.join(DATA_INPUT_DIR, BATCH_TO_RUN)
            logging.info(
                f"Pipeline: {pipeline_name}, marker_target: {marker_target}, "
                f"cell_line_target: {cell_line}, FinalFileLocation: {output_folder_name}"
            )
            
            main(data_input_dir_batch, use_multiprocessing, batch_name=BATCH_TO_RUN, dataset_name=input_folder_name)
            
            logging.info(f"Finished batch: {BATCH_TO_RUN}")
            

    except Exception as e:
        
        logging.info(f"Error:{e}")
        logging.exception(e)

    logging.info("\n\nDone!")