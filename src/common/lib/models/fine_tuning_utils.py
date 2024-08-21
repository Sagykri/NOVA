import torch
import torch.nn.functional as F
import logging
import numpy as np
import matplotlib.pyplot as plt


def compare_models(pretrained, finetuned, func, skip_head=True):
    """Comparing two models weights using the given func function

    Args:
        pretrained (Module): The pretrained model
        finetuned (Module): The finetuned model
        func (function(pre_layer_weights, fine_layer_weights)): The comparison function to apply on the weights. Should be func(pretrained_layer_weights, finetuned_layer_weights)
        skip_head (bool, optional): Should skip the head. Defaults to True.

    Returns:
        dict: Keys are layers, values are the result of func on the weights of this layer
    """
    results = {}
    
    for name_p, param_p in pretrained.named_parameters():
        if name_p in ['head.bias', 'head.weight'] and skip_head:
            # Skip the head since we replaced it
            continue
        if not param_p.requires_grad:
            # If no grads are requires, no change has been done
            continue
        
        # I can't get layer by name, therefore this additional for loop (and saving all in big dict is too much space for nothing)
        for name_f, param_f in finetuned.named_parameters():
            if name_f == name_p:
                # We find the corresponding layer in the finetune model
                
                w_p = param_p.data
                w_f = param_f.data
                
                res = func(w_p, w_f)
                
                # Saving the result per layer
                results[name_p] = res
                
                # Move to the next layer in the pretrained
                break
    
    return results
    
    
def angle_metrics(wp, wf):
    """Angle metrics for comparing two weights
    
    theta(L) = arccos( ( wp * wf ) / ( L2(wp) * L2(wf) ) )

    Args:
        wp (_type_): The weights for a single layer in the pretrained model
        wf (_type_): The weights for a single layer in the finetuned model
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
    
def __get_sorted_dict(metric_dict):
    # Sort the dictionary by values
    sorted_metric_values = dict(sorted(metric_dict.items(), key=lambda item: item[1]))

    layers_names = list(sorted_metric_values.keys())
    metric_values = list(sorted_metric_values.values())
    
    return layers_names, metric_values
    
def get_changed_layers(metric_dict, percentile_cutoff=50, is_above=True):
    layers_names, metric_values = __get_sorted_dict(metric_dict)

    metric_values_cutoff = np.percentile(metric_values, percentile_cutoff)
    
    # Find the valid indexes and sort them based on the metric_value in descending order
    layers_names = np.asarray(layers_names)
    if is_above:
        valid_layers_indexes = np.where(metric_values >= metric_values_cutoff)[0]
    else:
        valid_layers_indexes = np.where(metric_values <= metric_values_cutoff)[0]
    # Retrieve values at these indices
    metric_values = np.asarray(metric_values)
    values = metric_values[valid_layers_indexes]
    # Sort the indices based on the values in descending order
    sorted_indices = valid_layers_indexes[np.argsort(-values)]
    
    changed_layers = layers_names[sorted_indices]
    
    return changed_layers
    
def plot_comparison(metric_dict, percentile_cutoff=50):
    layers_names, metric_values = __get_sorted_dict(metric_dict)
    
    metric_values_cutoff = np.percentile(metric_values, percentile_cutoff)
    
    # Create the histogram
    plt.figure(figsize=(10, 40))
    plt.barh(layers_names, metric_values)
    plt.axvline(metric_values_cutoff, color='red')

    # Add titles and labels
    plt.title('Metric per layer')
    plt.xlabel('Metric')
    plt.ylabel('Layers')
    plt.show()
    
def freeze_layers(model, layers_names):
    freezed_layers = []
    
    if len(layers_names) == 0:
        logging.warn("len(layers_names) == 0 -> No layer got frozen")
        return
    
    # Freeze the specified layers
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layers_names):
            param.requires_grad = False
            freezed_layers.append(name)
    
    return freezed_layers