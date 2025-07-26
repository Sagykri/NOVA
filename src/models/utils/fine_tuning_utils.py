from typing import Callable, Dict, List
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Module


def compare_models(pretrained:Module, finetuned:Module, comparison_func: Callable[[torch.Tensor,torch.Tensor], float], skip_head:bool=True)->Dict[str, float]:
    """Comparing two models weights using the given func function

    Args:
        pretrained (Module): The pretrained model
        finetuned (Module): The finetuned model
        comparison_func (Callable[[torch.Tensor,torch.Tensor], float]): The comparison function to apply on the weights. Should be func(pretrained_layer_weights, finetuned_layer_weights)
        skip_head (bool, optional): Should skip the head. Defaults to True.

    Returns:
        Dict[str, float]: Keys are layers, values are the result of comparison_func on the weights of this layer
    """
    results = {}
    
    for name_p, param_p in pretrained.model.named_parameters():
        if 'head.' in name_p and skip_head:
            # Skip the head since we replaced it
            continue
        if not param_p.requires_grad:
            # If no grads are requires, no change has been done
            continue
        
        # I can't get layer by name, therefore this additional for loop (and saving all in big dict is too much space for nothing)
        for name_f, param_f in finetuned.model.named_parameters():
            if name_f == name_p:
                # We find the corresponding layer in the finetune model
                
                w_p = param_p.data
                w_f = param_f.data
                
                res = comparison_func(w_p, w_f)
                
                # Saving the result per layer
                results[name_p] = res
                
                # Move to the next layer in the pretrained
                break
    
    return results
    
    
def angle_metrics(wp:torch.Tensor, wf:torch.Tensor)->float:
    """Angle metrics for comparing two weights
    (as proposed here: https://arxiv.org/pdf/2312.15681)
    
    theta(L) = arccos( ( wp * wf ) / ( L2(wp) * L2(wf) ) )

    Args:
        wp (torch.Tensor): The weights for a single layer in the pretrained model
        wf (torch.Tensor): The weights for a single layer in the finetuned model
        
    Returns:
        float: The angle between the two weights
    """
    
    # Flat the weights
    wp = wp.reshape(-1)
    wf = wf.reshape(-1)
    
    # Norm the weights (set the L2 norm to 1)
    wp = F.normalize(wp, p=2, dim=0)
    wf = F.normalize(wf, p=2, dim=0)
        
    # wp * wf
    tmp = torch.matmul(wp, wf.T)
    
    angle = torch.arccos(tmp)
    
    return angle.item()
    

def get_changed_layers_based_on_cutoff(metric_dict:Dict[str, float], percentile_cutoff:int=50, is_below:bool=True)->List[str]:
    """Get the names of the layers that have been changed above/below the given percentile cutoff

    Args:
        metric_dict (Dict[str, float]): Keys are the layers' names, values are the changes
        percentile_cutoff (int, optional): The cutoff, in percentiles, between 0 to 100. Defaults to 50.
        is_below (bool, optional): Should we get the layers below the cutoff (above if set to False). Defaults to True.
        
    Returns:
        List[str]: The list of layers that have been changed above/below the given percentile cutoff
    """
    
    def __get_sorted_dict(metric_dict):
        # Sort the dictionary by values
        sorted_metric_values = dict(sorted(metric_dict.items(), key=lambda item: item[1]))

        layers_names = list(sorted_metric_values.keys())
        metric_values = list(sorted_metric_values.values())
        
        return layers_names, metric_values
    
    layers_names, metric_values = __get_sorted_dict(metric_dict)

    metric_values_cutoff = np.percentile(metric_values, percentile_cutoff)
    
    # Find the valid indexes and sort them based on the metric_value in descending order
    layers_names = np.asarray(layers_names)
    if is_below:
        valid_layers_indexes = np.where(metric_values <= metric_values_cutoff)[0]
    else:
        valid_layers_indexes = np.where(metric_values >= metric_values_cutoff)[0]
    # Retrieve values at these indices
    metric_values = np.asarray(metric_values)
    values = metric_values[valid_layers_indexes]
    # Sort the indices based on the values in descending order
    sorted_indices = valid_layers_indexes[np.argsort(-values)]
    
    changed_layers = layers_names[sorted_indices]
    
    return changed_layers
    

