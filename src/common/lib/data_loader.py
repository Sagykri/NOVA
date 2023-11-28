import sys
import os
from torch.utils.data import DataLoader
import logging

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")


from src.common.lib.dataset import Dataset    

def get_dataloader(dataset, batch_size, indexes=None, num_workers=2, shuffle=True):
    __shuffle = shuffle
    __pin_memory = True
    logging.warning(f"Using subset + shuffle={__shuffle} + pin_memory={__pin_memory}")
    
    ds = Dataset.get_subset(dataset, indexes) if indexes is not None else dataset  
    
    return DataLoader(ds,
                    num_workers=num_workers,
                    collate_fn=Dataset.get_collate_fn(shuffle=__shuffle),
                    batch_size=batch_size,
                    shuffle=__shuffle,
                    pin_memory=__pin_memory,
                    drop_last=False)