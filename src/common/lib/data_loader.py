import gc
import os
import random
import sys
from typing import Union



sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import math
import logging
import numpy as np
from random import shuffle
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras import backend as K

# from tensorflow.compat.v1.data import Dataset

import tensorflow as tf

from src.common.lib.utils import apply_for_all_gpus, getfreegpumem, get_nvidia_smi_output
from src.common.lib.dataset import Dataset


class DataLoader(keras.utils.Sequence):
    # def __init__(self, X, y, unique_markers, is_test:bool, conf: ModelConfig):
    def __init__(self, dataset: Dataset, batch_size, indexes=None, tpe=None):
        # self.X_paths, self.y = np.asarray(X), np.asarray(y) 
        # self.unique_markers = unique_markers
        self.batch_size = batch_size
        self.shuffle = dataset.shuffle
        self.flip = dataset.flip
        self.rot = dataset.rot
        self.conf = dataset.conf
        
        self.dataset = dataset
        
        self.X_paths, self.y, self.unique_markers = self.dataset.X_paths,\
                                                    self.dataset.y,\
                                                    self.dataset.unique_markers
                                                    
        ##### FOR TESTING ####
        # self.unique_markers = np.unique(self.y)
        ######################
                                                    
        
        if indexes is not None:
            self.X_paths = self.X_paths[indexes]
            self.y = self.y[indexes]
        
        
        ####
        # TODO:DELETE
        self.tpe = tpe
        ########
        
        self.len = len(self.y)
        
        # self.__set_params(conf)
        
        logging.info(f"{[self.tpe]} flip: {self.flip} rot: {self.rot}")
        logging.info(f"{[self.tpe]} unique_markers ({len(self.unique_markers)}): {self.unique_markers}")
        self.on_epoch_end(test="NANANANNAANNANA")
        # self.__split_train_val_test(X_paths, y)

    @staticmethod
    def get_generator(loader, use_multiprocessing=True, shuffle=True, workers=5, max_queue_size=5):
        def generator():
            multi_enqueuer = keras.utils.OrderedEnqueuer(loader,
                                                        use_multiprocessing=use_multiprocessing,
                                                        shuffle=shuffle)
            multi_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            while True:
                yield next(multi_enqueuer.get()) 
                
        return generator
    
    def get_generator2(self):
        # TODO:
        # TODO: Test this (+remove the inheritence) instead of using sequnce
        # TODO:
        j = 0
        flow = True
        num_cycle = self.__len__()
        
        while flow:
            i = np.mod(j, num_cycle).astype(int)
            
            j += 1
            flow = j < num_cycle
            
            # x, y = self.__getitem__(i)
            rows = 250
            x = np.ones((250, 100, 100, 2))
            cols = len(self.unique_markers)
            y_temp = np.concatenate([np.eye(1, cols, dtype=int) for i in range(rows)], axis=0)
            np.random.shuffle(y_temp)
            y = (x, y_temp, y_temp)
            logging.info(f"New shape: {x.shape}, {y_temp.shape}")
            res = apply_for_all_gpus(getfreegpumem)
            logging.info(f"Before Resources (Free, Used, Total): {res}")
            # nvidia_smi_info = apply_for_all_gpus(get_nvidia_smi_output)
            # logging.info(f"nvidia_smi: {nvidia_smi_info}")
            # train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
            # test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
            yield x,y
            logging.info("Deleting x and y")
            del x
            del y
            res = apply_for_all_gpus(getfreegpumem)
            logging.info(f"After Resources (Free, Used, Total): {res}")

    def __len__(self):
        logging.info(f"\n\n\n\n X [{self.tpe}] __len__ {math.ceil(self.len / self.batch_size)}")
        'Denotes the number of batches per epoch'
        return math.ceil(self.len / self.batch_size)
        

    def __getitem__(self, index):
        logging.info(f"\n\n\n\n XX [{self.tpe}] __getitem__ {index}")
        'Generate one batch of data'
        
        X_batch, y_batch = self.get_batch(index)
    
        logging.info(f"\n\n [load_batch] [{self.tpe}] X_batch: {X_batch.shape}, y [{np.unique(y_batch)}]: {y_batch.shape}, {y_batch}")
        
        labels_numeric = np.asarray([self.__get_numeric_label(lbl) for lbl in y_batch]) 
        logging.info(f"\n\n [load_batch] [{self.tpe}] labels_numeric: {labels_numeric}")
        y_batch = np.asarray(keras.utils.to_categorical(labels_numeric, num_classes=len(self.unique_markers)), dtype=np.float32)
        
        logging.info(f"\n\n [load_batch] [{self.tpe}] y_batch categorical: {y_batch}")
        logging.info(f"\n\n [load_batch] [{self.tpe}] labels_numeric unique: {len(np.unique(labels_numeric))}") #NANCY
        
        
        output = (X_batch,) * 2 + (y_batch,) * 2
    
        logging.info(f"Output shape: x:{output[0].shape}, y:{len(output[1:])}")
        # X passed as a target (y) for comparing it with the reconstructed image!
        return output[0], output[1:]

    def get_batch(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] 
        
        X_paths_batch = self.X_paths[indexes]
        y_batch = self.y[indexes]

        return self.__load_batch(X_paths_batch, y_batch)

    def on_epoch_end(self, test=''):
        logging.info(f"\n\n\n\n XXX [{self.tpe}] on_epoch_end {self.X_paths.shape} {test} {self.len}")
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
        logging.info("Clearing cache!")
        gc.collect()
        K.clear_session()


    def __load_batch(self, paths, labels):
        # logging.warning("ATTENTION! Taking only the first 10x10 pixels from each tile! (Change it back!)")
        'Generates data containing batch_size samples' 
        X_batch = []
        y_batch = []
        
        # Generate data
        for i, path in enumerate(paths):
            # path, tile_index = path.split('#')
            # tile_index = int(tile_index)
            imgs = np.load(path)#, mmap_mode='r')
            logging.warning("!ATTENTION! (DELETE THIS!) Taking only the first 2 tiles!")
            imgs = imgs[:2,...] # Nancy (&Sagy) changed this to 2 tiles of each site NANCY 
            n_tiles = imgs.shape[0]
            #n_tiles = imgs.shape[0]
            
            augmented_images = []
            
            if self.flip or self.rot:
                for j in range(n_tiles): 
                    # Augmentations
                    img = np.copy(imgs[j])
                    if self.flip:
                        img = random_horizontal_flip(img)
                        img = random_vertical_flip(img)
                    if self.rot:
                        img = random_choice_rotate(img)
                    
                    if not np.array_equal(img, imgs[j]):
                        augmented_images.append(img)
        
            # Store sample - all the tiles in site
            X_batch.append(imgs)
            y_batch.append([labels[i]]*n_tiles)
            
            # Append augmented images
            if len(augmented_images) > 0:
                augmented_images = np.stack(augmented_images)
                X_batch.append(augmented_images)
                y_batch.append([labels[i]]*len(augmented_images))
        
        X_batch = np.concatenate(X_batch)
        y_batch = np.asarray(flat_list_of_lists(y_batch))
        logging.info(f"y_batch shape: {y_batch.shape}")
        # y_batch = np.asarray(y_batch).reshape(-1,)#.reshape(-1,1)
        
        ###############################################################
        
        res = apply_for_all_gpus(getfreegpumem)
        logging.info(f"Resources (Free, Used, Total): {res}")
        nvidia_smi_info = apply_for_all_gpus(get_nvidia_smi_output)
        logging.info(f"nvidia_smi: {nvidia_smi_info}")
        
        return X_batch, y_batch
        
        # logging.info(f"\n\n [load_batch] [{self.tpe}] X_batch: {X_batch.shape}, y [{labels}]: {y_batch.shape}, {y_batch}")
        
        # labels_numeric = np.asarray([self.__get_numeric_label(lbl) for lbl in y_batch]) 
        # logging.info(f"\n\n [load_batch] [{self.tpe}] labels_numeric: {labels_numeric}")
        # y_batch = np.asarray(keras.utils.to_categorical(labels_numeric, num_classes=len(self.unique_markers)), dtype=np.float32)
        
        # logging.info(f"\n\n [load_batch] [{self.tpe}] y_batch categorical: {y_batch}")
        
        # output = (X_batch,) * 2 + (y_batch,) * 2
    
        # # X passed as a target (y) for comparing it with the reconstructed image!
        # return output[0], output[1:]
        
    def __get_numeric_label(self, lbl):
        res = np.argwhere(self.unique_markers == lbl).flatten()
        
        if res.size == 0:
            raise Exception(f"{lbl} not found in {self.unique_markers}")
        
        return res[0] 


# def test():
    
#     batch_path = ['/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/batch7',
#                   '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/batch8']
#     load_data_paths(input_folders=batch_path, 
#                     condition_l=True, line_l=True ,batch_l=True ,cell_type_l=True, 
#                     # markers, 
#                     #markers_to_exclude=['LAMP1', 'DCP1A', 'GM130', 'CLTC', 'PML', 'KIF5A', 'PURA', 'FMRP', 'ANXA11', 'GM130'], 
#                     #cell_lines_include=['WT', 'TBK1'], 
#                     #conds_include=['stress'], 
#                     train_part=0.7, shuffle=True, depth=3, sample_half_batch=False)
        
# test()

# TODO: Move to a different file
# Augmentation transforms
def random_horizontal_flip(image):
    if np.random.rand() < 0.5:
        return np.fliplr(image)
    return image

def random_vertical_flip(image):
    if np.random.rand() < 0.5:
        return np.flipud(image)
    return image

def random_choice_rotate(image):
    k = np.random.choice([0, 1, 2, 3])
    return np.rot90(image, k=k, axes=(0,1))

def flat_list_of_lists(l):
    return [item for sublist in l for item in sublist]




# class DataLoaderSPDFactory():
#     def __init__(self, conf: ModelConfig):
#         # self.__set_params(conf)
        
#         self.mapping = {
#             "train" : (self.X_train, self.y_train, self.unique_markers),
#             "valid" : (self.X_valid, self.y_valid, self.unique_markers),
#             "test"  : (self.X_test, self.y_test, self.unique_markers)
#         }
        
#         self.data_loaders = {}
        

#     def get(self, dataset_type: Union["train", "valid", "test"]) -> DataLoaderSPD:
#         """
#         Get dataloader by dataset type: train, valid or test
#         """
        
#         # Init only on demand + cache
#         if dataset_type not in self.data_loaders.keys():
#             X, y, unique_markers = self.mapping[dataset_type]
#             self.data_loaders[dataset_type] = DataLoaderSPD(X, y, unique_markers,
#                                                             dataset_type == 'test', self.conf)
            
#         return self.data_loaders[dataset_type]
        

                
            



