import os
import sys
from typing import Dict

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

from src.common.lib.models.NOVA_model import NOVAModel
from src.common.lib.models.fine_tuning_utils import angle_metrics, compare_models, get_changed_layers_based_on_cutoff


def __extracting_params_from_sys()->Dict:
    """Exctract paras from sys

    Returns:
        Dict: The extracted values
    """
    assert len(sys.argv) == 3, f"Invalid config paths. You must specify: pretrained model checkpoint path, finetuned model checkpoint path. ({len(sys.argv)}: {sys.argv})"
    
    pretrained_path:str = sys.argv[1]
    finetuned_path:str = sys.argv[2]
    
    return {
        pretrained_path:pretrained_path,
        finetuned_path:finetuned_path
    }

def __get_least_changed_layers(pretrained_path: str, finetuned_path:str)->None:
    """Print a list of the least (below the median) changed layers between the models

    Args:
        pretrained_path (str): The path to the checkpoint of the pretrained model
        finetuned_path (str): The path to the checkpoint of the finetuned (without freeze at all) model
    """
    # Load the pretrained and finetuned (without freezing) models from the given checkpoints paths
    pretrained_model:NOVAModel = NOVAModel.load_from_checkpoint(pretrained_path)
    finetuned_model:NOVAModel = NOVAModel.load_from_checkpoint(finetuned_path)
    
    # The function used for the comparison of the two models
    comparison_func = angle_metrics
    
    # Compare the models based on the comparison_func
    diff_between_weights:Dict[str, float] = compare_models(pretrained_model, finetuned_model, comparison_func=comparison_func)
    
    # Extract the layers that have been changed below the overall median
    layers_names = get_changed_layers_based_on_cutoff(diff_between_weights, percentile_cutoff=50, is_below=True)
    
    print("Layer names:")
    print(layers_names)

if __name__ == "__main__":    
    print("Calling the get_least_changed_layers func...")
    try:
        args = __extracting_params_from_sys()
        __get_least_changed_layers(**args)
    except Exception as e:
        raise e
    print("Done")
    
    