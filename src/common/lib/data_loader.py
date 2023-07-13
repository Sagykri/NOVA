import sys
import os
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")


from src.common.lib.dataset import Dataset    

def get_dataloader(dataset, batch_size, indexes, num_workers=2):
    return DataLoader(dataset,
                            num_workers=num_workers,
                            collate_fn=Dataset.get_collate_fn(shuffle=dataset.shuffle),
                            batch_sampler=BatchSampler(SubsetRandomSampler(indexes),
                                                       batch_size=batch_size,
                                                       drop_last=False))
