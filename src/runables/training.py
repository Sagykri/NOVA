import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
print(f"MOMAPS_HOME: {os.getenv('MOMAPS_HOME')}")

import numpy as np
import logging
import tensorflow as tf
from copy import deepcopy
from src.datasets.dataset_spd import DatasetSPD
from src.datasets.dataset_conf import DatasetConf
from src.common.lib.utils import load_config_file
from src.common.lib.model import Model
from src.common.lib.data_loader import DataLoader

from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1.keras import backend as K

def train_with_datamanager():
    
    # Importing customize config for this run
    if len(sys.argv) == 2:
        config_path_train, config_path_val, config_path_test = sys.argv[1], sys.argv[1], sys.argv[1]
    elif len(sys.argv) == 3:
        config_path_train, config_path_val, config_path_test = sys.argv[1], sys.argv[2], sys.argv[2]
    elif len(sys.argv) == 4:
        config_path_train, config_path_val, config_path_test = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        raise ValueError("Invalid config paths. Must specify at least one config path.")
    
    config_train, config_val, config_test = load_config_file(config_path_train, 'train'),\
                                            load_config_file(config_path_val, 'val'),\
                                            load_config_file(config_path_test, 'test')
    logging.info("init")
    
    
    logging.info("Set split_by_set to True and data_set_type accordingly")
    config_train.DATA_SET_TYPE, config_train.SPLIT_BY_SET = "train", True
    config_val.DATA_SET_TYPE, config_val.SPLIT_BY_SET = "val", True
    config_test.DATA_SET_TYPE, config_test.SPLIT_BY_SET = "test", True
    
    logging.info(f"Is GPU available: {tf.test.is_gpu_available()}")
    
    logging.info("Creating model")
    
    model = Model(config_train)
    logging.info("Loading training data")
    model.load_with_datamanager()
    
    logging.info(f"Calculating training data variance...")
    logging.info(f"Training data variance: {np.var(model.train_data)}")
    
    logging.info("Loading validation data")
    model.set_params(config_val)           
    model.load_with_datamanager()
    
    logging.info("Loading test data")
    model.set_params(config_test)           
    model.load_with_datamanager()

    logging.info("[Start] Training..")

    model.train_with_datamanager()
                
    logging.info("[End] Training..")
    
    
def train_with_dataloader():
    
    # Importing customize config for this run
    config_path_model = sys.argv[1]
    is_one_config_supplied = len(sys.argv) == 3
    
    if is_one_config_supplied:
        config_path_train, config_path_val, config_path_test = sys.argv[2], sys.argv[2], sys.argv[2]
    elif len(sys.argv) == 5:
        config_path_train, config_path_val, config_path_test = sys.argv[2], sys.argv[3], sys.argv[4]
    else:
        raise ValueError(f"Invalid config paths. Must specify one or three config paths and training config. ({len(sys.argv)}: {sys.argv})")
    
    config_model = load_config_file(config_path_model, 'model')
    config_train, config_val, config_test = load_config_file(config_path_train, 'train', config_model.CONFIGS_USED_FOLDER),\
                                            load_config_file(config_path_val, 'val', config_model.CONFIGS_USED_FOLDER),\
                                            load_config_file(config_path_test, 'test', config_model.CONFIGS_USED_FOLDER)
    
    logging.info("init")
    
    
    # gpus = tf.compat.v1.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.compat.v1.config.experimental.set_logical_device_configuration(
    #             gpus[0],
    #             [tf.compat.v1.config.experimental.LogicalDeviceConfiguration(memory_limit=1024)])
    #         logical_gpus = tf.compat.v1.config.experimental.list_logical_devices('GPU')
    #         logging.info(f"{len(gpus)}, Physical GPUs, {len(logical_gpus)}, Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         logging.info(f"ERROR: {e}")
    logging.info(f"ENV: {os.environ['TF_FORCE_GPU_ALLOW_GROWTH']}")
    gpus = tf.compat.v1.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.compat.v1.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.compat.v1.config.experimental.list_logical_devices('GPU')
                logging.info(f"{len(gpus)}, Physical GPUs, {len(logical_gpus)}, Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(f"ERROR: {e}")
    
    
    logging.info(f"Is GPU available: {tf.test.is_gpu_available()}")
    logging.info(f"Num GPUs Available: physical: {tf.compat.v1.config.experimental.list_physical_devices('GPU')} logical: {tf.compat.v1.config.experimental.list_logical_devices('GPU')}")
    logging.info(f"Visible devices: {tf.compat.v1.config.experimental.get_visible_devices()}")
    
    logging.info("Creating model")
    
    model = Model(config_model)
    
    logging.info("Init datasets")
    dataset_train = DatasetSPD(config_train)
    train_indexes, val_indexes, test_indexes = None, None, None
    dataset_val, dataset_test = None, None
    
    logging.info(f"Data shape: {dataset_train.X_paths.shape}, {dataset_train.y.shape}")
    
    if is_one_config_supplied:
        dataset_val, dataset_test = deepcopy(dataset_train), deepcopy(dataset_train) # the deepcopy is important. do not change. 
        dataset_test.flip, dataset_test.rot = False, False
        if config_train.SPLIT_DATA:
            logging.info("Split data...")
            train_indexes, val_indexes, test_indexes = dataset_train.split()
    else:
        dataset_val, dataset_test = DatasetSPD(config_val), DatasetSPD(config_test)
    
    #######################
    # DEBUG NANCY
    # from sklearn.model_selection import train_test_split
    # logging.info(f"\n\n\nXXXXX before train_test_split, {train_indexes.shape} {dataset_train.y}")
    # train_indexes, _ = train_test_split(train_indexes, 
    #                                         train_size=0.4,
    #                                         random_state=config_train.SEED,
    #                                         shuffle=True,
    #                                         stratify=dataset_train.y[train_indexes])
    # logging.info(f"\n\n\nXXXXX after train_test_split, {train_indexes.shape} {dataset_train.y} {val_indexes.shape} {test_indexes.shape}")
    #######################
    
    batch_size = config_model.BATCH_SIZE
    logging.info(f"Init dataloaders (batch_size: {batch_size})")
    dataloader_train, dataloader_val, dataloader_test = DataLoader(dataset_train, batch_size=batch_size, indexes=train_indexes, tpe='train'),\
                                                        DataLoader(dataset_val, batch_size=batch_size, indexes=val_indexes, tpe='val'),\
                                                        DataLoader(dataset_test, batch_size=batch_size, indexes=test_indexes, tpe='test')
    
    logging.info(f"\n\n\n\n\nBefore model.load_with_dataloader.. {dataloader_train.X_paths.shape}, {dataloader_val.X_paths.shape}, {dataloader_test.X_paths.shape}")
    
    model.load_with_dataloader(dataloader_train, dataloader_val, dataloader_test)

    logging.info("[Start] Training..")

   
    
    model.train_with_dataloader()
                
    logging.info("[End] Training..")


if __name__ == "__main__":
    
    print("Set allow_growth = True")
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = Session(config=config)
    K.set_session(sess)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # print(os.environ['TF_FORCE_GPU_ALLOW_GROWTH'])
    print("set allow_grouwth = True in another way")
    # Limit GPU memory growth
    gpus = tf.compat.v1.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.compat.v1.config.experimental.set_memory_growth(gpu, True)



    # print("set mixed_precision")
    # from tensorflow.keras.mixed_precision import experimental as mixed_precision

    # # Enable mixed precision training
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    
    print("Calling the training func...")
    try:
        # tf.enable_eager_execution()
        # train_with_datamanager()
        train_with_dataloader()
    except Exception as e:
        logging.exception(e)
        raise e
    logging.info("Done")
    
    