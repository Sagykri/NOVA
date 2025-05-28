import os
import subprocess
import datetime
import logging
import sys
import argparse


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

from src.models.utils.eval_model_utils import run_evaluation, save_plots

def init_logging(path):
    """Init logging.
    Writes to log file and console.
    Args:
        path (string): Path to log file
    """
  
    jobid = os.getenv('LSB_JOBID')
    jobname = os.getenv('LSB_JOBNAME')
    # if jobname is not specified, the jobname will include the path of the script that was run.
    # In this case we'll have some '/' and '.' in the jobname that should be removed.
    if jobname:
        jobname = jobname.replace('/','').replace('.','') 

    username = 'UnknownUser'
    if jobid:
        # Run the bjobs command to get job details
        result = subprocess.run(['bjobs', '-o', 'user', jobid], capture_output=True, text=True, check=True)
        # Extract the username from the output
        username = result.stdout.replace('USER', '').strip()
    
    __now_str = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
    log_file_path = os.path.join(path, __now_str + f'_{jobid}_{username}_{jobname}.log')
    if not os.path.exists(path):
        os.makedirs(path)
        
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])

    logging.info(f"Init (log path: {log_file_path}; JOBID: {jobid} Username: {username}) JOBNAME: {jobname}")
    logging.info(f"NOVA_HOME={os.getenv('NOVA_HOME')}, NOVA_DATA_HOME={os.getenv('NOVA_DATA_HOME')}")

def parse_positional_args(argv):
    """
    Parse positional arguments in the following order:
    1. experiment (str)
    2. batch (str)
    3. k (int)
    4. neg_k (int)
    5. save_dir (str, optional)

    Args:
        argv (List[str]): List of command-line arguments.

    Returns:
        dict: Parsed values.
    """
    if len(argv) < 4 or len(argv) > 7:
        raise ValueError("Usage: script.py <experiment> <batch> <save_dir> [k] [neg_k] [sample_fraction]")

    experiment = argv[1]
    batch = argv[2]
    save_dir = argv[3]
    k = int(argv[4]) if len(argv) > 3 else 20
    neg_k = int(argv[5]) if len(argv) > 4 else 20
    sample_fraction = float(argv[6]) if len(argv) > 5 else 1.0

    return {
        'experiment': experiment,
        'batches': [batch],  # Wrapped in list for compatibility with rest of pipeline
        'save_dir': save_dir,
        'k': k,
        'neg_k': neg_k,
        'sample_fraction': sample_fraction
    }

if __name__ == "__main__":
    args = parse_positional_args(sys.argv)

    experiment = args['experiment']
    batches = args['batches']
    save_dir = args['save_dir']
    k = args['k']
    neg_k = args['neg_k']
    sample_fraction = args['sample_fraction']

    model_folders_dict = {
            'pretrained': '/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/pretrained_model',  
            'finetuned_CL': '/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model',  
            'finetuned_CL_nofreeze': '/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model_no_freeze',  
            'finetuned_CE_nofreeze': '/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model_classification_with_batch_no_freeze',
            'finetuned_CE': "/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model_classification_with_batch_freeze"
        }

    precomputed_dists_paths = {
            'pretrained': None, 
            'finetuned_CL': None,
            'finetuned_CL_nofreeze': None, 
            'finetuned_CE_nofreeze': None,
            'finetuned_CE': None
        }

    init_logging(os.path.join(save_dir, experiment, '_'.join(batches), 'logs'))
    logging.info(f"Model folders: {model_folders_dict}; Experiment: {experiment}; Batches: {batches}; k: {k}; neg_k: {neg_k} ; Save dir: {save_dir}; Sample fraction: {sample_fraction}")
    
    try:
        results = run_evaluation(model_folders_dict, experiment_type=experiment, batches=batches, precomputed_dists_paths=precomputed_dists_paths, save_dir=save_dir, k=k, neg_k=neg_k, sample_fraction=sample_fraction)
        save_plots(results, save_dir=save_dir, experiment_type=experiment, batches=batches, k=k, neg_k=neg_k)
    except Exception as e:
        logging.exception(f"Error during evaluation {str(e)}")
        raise