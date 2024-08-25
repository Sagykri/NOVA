
import numpy as np

class EarlyStoppingInfo():
    """Holds information for handeling the early stopping
    """
    def __init__(self, counter:int):
        self.__init_value = counter
        self.reset()
        
    def reset(self):
        self.counter: int = self.__init_value

def get_params_groups(*models):
    regularized = []
    not_regularized = []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]

def cosine_scheduler(
    base_value, 
    final_value, 
    epochs, 
    niter_per_ep, 
    warmup_epochs=0, 
    start_warmup_value=0
):
    # Calculate the number of warmup iterations
    warmup_iters = warmup_epochs * niter_per_ep
    
    # Create the warmup schedule if warmup is required
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    else:
        warmup_schedule = np.array([])

    # Calculate the number of iterations for the cosine schedule
    total_iters = epochs * niter_per_ep - warmup_iters
    iters = np.arange(total_iters)
    
    # Create the cosine schedule
    cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / total_iters)
    )

    # Concatenate warmup and cosine schedules
    schedule = np.concatenate((warmup_schedule, cosine_schedule))
    
    # Ensure the schedule has the correct number of elements
    assert len(schedule) == epochs * niter_per_ep, "Schedule length does not match the expected number of iterations."
    
    return schedule
