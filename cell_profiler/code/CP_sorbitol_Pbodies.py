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

import cell_profiler_utils as cp_utils  

# ====================

# Allow imports from project tree
BASE_DIR = os.path.join('/home', 'projects', 'hornsteinlab', 'Collaboration', 'NOVA')
sys.path.insert(1, BASE_DIR)

# ----- Config -----
input_folder_name   = 'Sorbitol_experiment_PBs_TDP43_sorted'
marker_targte       = ['LSM14A']  # list of markers to include; None for all
output_folder_name  = 'Sorbitol_PB_FE_LSM14A_120925'
pipeline_name       = 'PB_feature_extraction_sorbitol_LSM14A.cppipe'

DATA_INPUT_DIR = os.path.join(
    BASE_DIR, 'cell_profiler', 'outputs', 'filtered_by_brenner_post_rescale_outputs', input_folder_name
)
pipeline_path = os.path.join(
    BASE_DIR, 'cell_profiler', 'pipelines', 'For_sorbitol', pipeline_name
)

# Paths to local directories
OUTPUT_DIR   = os.path.join(BASE_DIR, 'cell_profiler', 'outputs', 'cell_profiler_RUNS', output_folder_name)
LOG_DIR_PATH = os.path.join(BASE_DIR, 'cell_profiler', 'logs')

Batch_name = "batch3"
# ------------------

def _find_items(input_data_batch_dir):
    """Return list of (input_dir, output_dir) pairs per marker."""
    return cp_utils.find_marker_folders(
        batch_path=input_data_batch_dir,
        output_dir=OUTPUT_DIR,
        depth=5,
        markers_to_include=marker_targte
    )

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
        f"Pipeline: {pipeline_name} marker_targte: {marker_targte}"
    )

    items = list(_find_items(input_data_batch_dir))

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
        )

        try:
            with ctx.Pool(processes=5,
                        initializer=_worker_init, # starts JVM+pipeline once per worker
                        initargs=(log_queue,),
                        maxtasksperchild=25) as pool:
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
        logging.info(f"Pipeline: {pipeline_name}, marker_targte: {marker_targte}, FinalFileLocation: {output_folder_name}")

        # Iterate over batches
        for BATCH_TO_RUN in [Batch_name]:  
            data_input_dir_batch = os.path.join(DATA_INPUT_DIR, BATCH_TO_RUN)
            logging.info(f"Pipeline: {pipeline_name}, marker_targte: {marker_targte}, FinalFileLocation: {output_folder_name}")
            
            main(data_input_dir_batch, use_multiprocessing, batch_name=BATCH_TO_RUN, dataset_name=input_folder_name)
            
            logging.info(f"Finished batch: {BATCH_TO_RUN}")
            

    except Exception as e:
        
        logging.info(f"Error:{e}")
        logging.exception(e)

    logging.info("\n\nDone!")