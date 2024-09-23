import sys
import os
from typing import List, Tuple, Union
from torch.utils.data import DataLoader
import logging

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")


from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.dataset_config import DatasetConfig
from src.models.trainers.trainer_config import TrainerConfig
from src.datasets.dataset_base import DatasetBase    

def get_dataloader(dataset:DatasetBase, batch_size:int, indexes:List[int]=None, num_workers:int=6, shuffle:bool=True, drop_last:bool=True)->DataLoader:
    """Get a dataloader object initialized with the given params

    Args:
        dataset (Dataset): The dataset the dataloader would work with
        batch_size (int): Number of samples the dataloader would return each call
        indexes (List[int], optional): Indexes to take a subset from the data. Defaults to None.
        num_workers (int, optional): Num of workers on the machine to use for the loading of the data. Defaults to 6.
        shuffle (bool, optional): Should the data be shuffled. Defaults to True.
        drop_last (bool, optional): Should we get the residuals samples that don't fill an entire batch_size. Defaults to True.

    Returns:
        DataLoader: The initialized dataloader
    """
    ds = DatasetBase.get_subset(dataset, indexes) if indexes is not None else dataset  
    
    return DataLoader(ds,
                    num_workers=num_workers,
                    collate_fn=DatasetBase.get_collate_fn(shuffle=shuffle),
                    batch_size=batch_size,
                    shuffle=shuffle,
                    pin_memory=True,
                    drop_last=drop_last)
    
def init_dataloaders_with_config(trainer_config:TrainerConfig, dataset_config:DatasetConfig,
                                 dataset_type:type=DatasetNOVA)->Union[DataLoader, Tuple[DataLoader, DataLoader, DataLoader]]:
    """Return dataloaders initialized with the dataset config and trainer config.\n
       If SPLIT_DATA is True, it returns dataloaders for train, val and test,\n
       otherwise a single dataloader for the entire dataset is returned
       
    Args:
        trainer_config (TrainerConfig): The trainer configuration
        dataset_config (DatasetConfig): The dataset configuration
        dataset_type (type, optional): The type of dataset to init. Defaults to DatasetSPD.

    Returns:
        Union[DataLoader, Tuple[DataLoader, DataLoader, DataLoader]]: Dataloader for the entiredata,\n
        or three dataloaders splitted to train, val and test if SPLIT_DATA in the dataset_config is set to True
    """

    logging.info("Init datasets")
    dataset = dataset_type(dataset_config)
    logging.info(f"Data shape: {dataset.X_paths.shape}, {dataset.y.shape}")
    
    batch_size = trainer_config.BATCH_SIZE
    num_workers = trainer_config.NUM_WORKERS
    drop_last = trainer_config.DROP_LAST_BATCH
    logging.info(f"Init dataloaders (batch_size: {batch_size}, num_workers: {num_workers})")
    
    if not dataset_config.SPLIT_DATA:
        dataloader = get_dataloader(dataset, batch_size, num_workers=num_workers)
        return dataloader
    
    logging.info("Split data...")
    train_indexes, val_indexes, test_indexes = dataset.split()
    dataloader_train, dataloader_val, dataloader_test = get_dataloader(dataset, batch_size, indexes=train_indexes, num_workers=num_workers, shuffle=True, drop_last=drop_last),\
                                                        get_dataloader(dataset, batch_size, indexes=val_indexes, num_workers=num_workers, shuffle=True, drop_last=drop_last), \
                                                        get_dataloader(dataset, batch_size, indexes=test_indexes, num_workers=num_workers, shuffle=False, drop_last=False)
                                                        
    return dataloader_train, dataloader_val, dataloader_test
                                                            