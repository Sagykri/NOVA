
from abc import ABC, ABCMeta, abstractmethod

import logging
import sys
import os
import numpy as np
import torch
from copy import deepcopy

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from sklearn.model_selection import train_test_split
from src.common.configs.dataset_config import DatasetConfig
import src.common.lib.utils as utils
from copy import deepcopy

## FOR SYNTHETIC (020724)
import itertools
from collections import defaultdict
##

# TODO: Use torch.transforms instead!
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


class Dataset(torch.utils.data.Dataset ,metaclass=ABCMeta):
    def __init__(self, conf: DatasetConfig, transform=None):
        self.__set_params(conf)
        self.transform = transform
        
    def __set_params(self, conf: DatasetConfig):
        self.input_folders = conf.INPUT_FOLDERS
        self.add_condition_to_label = conf.ADD_CONDITION_TO_LABEL
        self.add_line_to_label = conf.ADD_LINE_TO_LABEL
        self.add_batch_to_label = conf.ADD_BATCH_TO_LABEL
        self.add_type_to_label = conf.ADD_TYPE_TO_LABEL
        self.markers = conf.MARKERS
        self.markers_to_exclude = conf.MARKERS_TO_EXCLUDE
        self.cell_lines = conf.CELL_LINES
        self.conditions = conf.CONDITIONS
        self.train_part = conf.TRAIN_PCT
        self.shuffle = conf.SHUFFLE
        self.flip = conf.AUG_TO_FLIP
        self.rot = conf.AUG_TO_ROT
        self.to_split_data = conf.SPLIT_DATA
        self.data_set_type = conf.DATA_SET_TYPE
        self.is_aug_inplace = conf.IS_AUG_INPLACE
        self.reps = conf.REPS
        

        self.conf = conf
        
        self.X_paths, self.y, self.unique_markers = self._load_data_paths()  
        
        ### FOR SYNTHETIC (020724)
        
        # self.unique_classes = self.unique_ordered_list(['_'.join(m.split('_')[1:]) for m in self.unique_markers])
        # self.unique_markers = self.unique_ordered_list([m.split('_')[0] for m in self.unique_markers])
        # x_paths = self.X_paths.copy()
        # np.random.shuffle(x_paths)
        # _tmp = self.organize_paths(x_paths)
        # self.X_paths, self.y = self.create_index_mapping(_tmp)
        
        ###
        
        # PATCH...
        self.label = self.y      
        
    
    ## FOR SYNTHETIC (020724)
    
    def unique_ordered_list(self, original_list):
        seen = set()
        unique_list = [x for x in original_list if not (x in seen or seen.add(x))]
        unique_list = np.asarray(unique_list)
        return unique_list
    
    def organize_paths(self, root_paths):
        organized_paths = defaultdict(lambda: defaultdict(list))
        
        ##
        # Shuffling the labels
        # root_paths_shuffled = root_paths.copy()
        # np.random.shuffle(root_paths_shuffled)
        
        # for i, path in enumerate(root_paths):
        #     parts = root_paths_shuffled[i].split(os.sep)
        #     root = os.path.join(parts[-4], parts[-3])
        #     subfolder = parts[-2]
        #     organized_paths[root][subfolder].append(path)
        # return organized_paths
        
        ##
        
        ##
        # Duplicating the same single marker
        # for path in root_paths:
        #     parts = path.split(os.sep)
        #     root = os.path.join(parts[-4], parts[-3])
        #     subfolder = parts[-2]
        #     for i in range(4):
        #         organized_paths[root][subfolder].append(path)
        # return organized_paths
        ##
        
        for path in root_paths:
            parts = path.split(os.sep)
            root = os.path.join(parts[-4], parts[-3])
            subfolder = parts[-2]
            organized_paths[root][subfolder].append(path)
        return organized_paths


    def create_index_mapping(self, organized_paths):
        to_sort = True
        print(f"to_sort = {to_sort}")
        
        index_mapping = []
        root_mapping = []
        for root, subfolders in organized_paths.items():
            num_files = min(len(files) for files in subfolders.values())
            for i in range(num_files):
                combo = []
                for subfolder, files in subfolders.items(): 
                    # for l in range(4): # SAGY _ REPLICATE THE SAME MARKER for testing
                    #     combo.append(files[i % len(files)])
                    combo.append(files[(i) % len(files)])
                if to_sort:
                    combo.sort() # SAGY _ KEEP THE ORDER OF MARKERS THE SAME!
                
                # SAGY _ CHANGING THE ORDER OF THE MARKERSR for testing
                # combo = combo[::-1]
                ###
                
                if i >= len(index_mapping):
                    index_mapping.append([])
                    root_mapping.append([])
                index_mapping[i].append(combo)
                root_mapping[i].append(root.replace(os.sep,'_'))
                
        index_mapping = utils.flat_list_of_lists(index_mapping)
        root_mapping = utils.flat_list_of_lists(root_mapping)
        
        index_mapping = np.asarray(index_mapping)
        root_mapping = np.asarray(root_mapping)
        
        return index_mapping, root_mapping

    ###
    
        
    @abstractmethod
    def _load_data_paths(self):
        pass
    
    """
    # TODO: (take from improvement/preprocessing)
    def get_variance():
        from src.common.lib.image_sampling_utils import sample_images_all_markers_all_lines
    
        paths = sample_images_all_markers_all_lines(50)
        
        images = np.concatenate([np.load(path) for path in paths])
        
        return np.var(images)
    """
    
    def __len__(self):
        return len(self.y)
    
    @staticmethod
    def get_subset(dataset, indexes):
        subset = deepcopy(dataset)
        
        subset.X_paths, subset.y = dataset.X_paths[indexes], dataset.y[indexes]
        
        subset.label = subset.y
        
        return subset
    
    # Function to apply the transform to each sample in the batch
    def __apply_transform_per_sample(self, batch, transform):
        batch_size = batch.size(0)
        transformed_batch = []
        for i in range(batch_size):
            transformed_batch.append(transform(batch[[i]]))
        transformed_batch = torch.vstack(transformed_batch)
        return transformed_batch
    
    def __apply_paired_transform_batch(self, batch, transform):
        batch_size = batch.size(0)
        batch_global = torch.empty_like(batch)
        batch_local = torch.empty_like(batch)
        
        # Apply transform in parallel
        for i in range(batch_size):
            img_global, img_local = transform(batch[[i]])
            batch_global[i] = img_global
            batch_local[i] = img_local
            
        return batch_global, batch_local
    
    def __getitem__(self, index):
        # logging.info(f"\n\n\n\n __getitem__ {index}")
        'Generate one batch of data'
        # print(f"index={index}")
        X_batch, y_batch, paths_batch = self.get_batch(index, return_paths=True)
    
        y_batch = self.__label_converter(y_batch, label_format='index')
        paths_batch = paths_batch.reshape(-1,1)
        
        is_dual_inputs = utils.get_if_exists(self.conf, 'IS_DUAL_INPUTS', False)
        if is_dual_inputs:
            if self.transform:
                
                X_batch = torch.from_numpy(X_batch)  
                X_batch_global, X_batch_local = self.__apply_paired_transform_batch(X_batch, self.transform)
                
                return {'image_global': X_batch_global, 'image_local': X_batch_local, 'label': y_batch, 'image_path': paths_batch}        

        
        if self.transform:
            X_batch = torch.from_numpy(X_batch)
            X_batch = self.__apply_transform_per_sample(X_batch, self.transform)
            
        
        # if self.transform is not None:
        #     X_batch_global, X_batch_local = self.transform(X_batch)
        #     y_batch = np.repeat(y_batch, len(X_batch_global)//len(y_batch))
        #     return {'image_global': X_batch_global, 'image_local': X_batch_local, 'label': y_batch, 'image_path': paths_batch}
            
        return {'image': X_batch, 'label': y_batch, 'image_path': paths_batch}        

    def get_batch(self, indexes, return_paths=False):
        if not isinstance(indexes, list):
            indexes = [indexes]
        # logging.info(f"Indexes: {indexes}, {self.X_paths[indexes]}")
        # print(indexes)
        X_paths_batch = self.X_paths[indexes]
        # print(X_paths_batch)
        y_batch = self.y[indexes]

        return self.__load_batch(X_paths_batch, y_batch, return_paths=return_paths)

    # Synthetic ############# 

    # def __load_batch(self, paths, labels, return_paths=False):
    #     'Generates data containing batch_size samples' 
    #     X_batch = []
    #     y_batch = []
    #     paths_batch = []
        
        
    #     # SAGY_ TESING DIFFERENT ORDERS OF CONCAT - does it make any change?
    #     # n_markers = len(paths[0])
    #     # order = np.arange(n_markers)
    #     # np.random.shuffle(order)
    #     # order = [0,2,1,3]#[1,2,3,0]
    #     # print(f"order: {order}")
    #     # ran = False
    #     ##
        
        
    #     # if self.changable_channel_count:
    #     #     # SAGY - TRAINING WITH various channels
    #     #     markers_count = len(paths[0])
    #     #     _optional_markers_indx = np.arange(markers_count)
    #     #     n_markers_to_choose = np.random.randint(1, markers_count+1,1)[0]
    #     #     markers_chosen = np.random.choice(_optional_markers_indx, size=n_markers_to_choose, replace=False)
    #     #     markers_chosen.sort()
    #     # ####
        
    #     # Generate data
    #     for i, path in enumerate(paths):
    #         imgs =[]
            
    #         # if self.changable_channel_count:
    #         #     # SAGY - TRAINING WITH various channels
    #         #     for j in markers_chosen:
    #         #         # print(path[j])
    #         #         im = np.load(path[j])
    #         #         imgs.append(im)
    #         #     # print('\n')
    #         #     #####
    #         # else:
    #         ## ORIGINAL
    #         for p in path:
    #             # print(path)
    #             im = np.load(p)
    #             imgs.append(im)
    #         # print('\n')
    #         # ###
                
                
    #         # # SAGY_ TESING DIFFERENT ORDERS OF CONCAT - does it make any change?
    #         # for j in order:
    #         #     # if i <= 1:
    #         #     #     print(path[j])
                    
    #         #     im = np.load(path[j])
    #         #     imgs.append(im)
    #         # print('\n')
    #         # ran = True
    #         ####
            
    #         min_samples = min([len(im) for im in imgs])
            
    #         imgs = [im[:min_samples] for im in imgs]
    #         # imgs = np.concatenate(imgs, axis=1) # N,100+*|markers|,100,2
    #         imgs = np.concatenate(imgs, axis=-1) # N,100,100,2*|markers|
            
    #         n_tiles = imgs.shape[0]
            
    #         if min_samples > 0:
    #             X_batch.append(imgs)
    #             y_batch.append([labels[i]]*n_tiles)
            
    #     X_batch = np.concatenate(X_batch)
    #     y_batch = np.asarray(utils.flat_list_of_lists(y_batch))
    #     if return_paths:
    #         paths_batch = np.asarray([]) # TODO: FIX IT
    #     #     paths_batch = np.asarray(utils.flat_list_of_lists(paths_batch))

    #     # If the channel axis is the last one, move it to be the second one
    #     # (#tiles, 100, 100, #channel) -> (#tiles, #channel, 100, 100)
    #     if np.argmin(X_batch.shape[1:]) == len(X_batch.shape) - 2:
    #         X_batch = np.moveaxis(X_batch, -1, 1)

    #     # logging.info(f"y_batch shape: {y_batch.shape}")
        
    #     # logging.info(f"\n\n [load_batch]  X_batch: {X_batch.shape}, y [{np.unique(y_batch)}]: {y_batch.shape}")

    #     #############################################
    #     # NOTE! ************** [WARNING] Repeating the target channel 3 times (dropping the nucleus channel) ************
    #     # logging.warn("[WARNING] Repeating the target channel 3 times (dropping the nucleus channel)")
    #     # X_batch = X_batch[:,[0], ...]
    #     # X_batch = np.repeat(X_batch, 3, axis=1)
    #     # X_batch = torch.from_numpy(X_batch)
    #     # logging.info(f"Repeated (3) X_batch: {X_batch.shape}")
        
    #     ###############################################################
        
    #     # res = utils.apply_for_all_gpus(utils.getfreegpumem)
    #     # logging.info(f"Resources (Free, Used, Total): {res}")
    #     # nvidia_smi_info = utils.apply_for_all_gpus(utils.get_nvidia_smi_output)
    #     # logging.info(f"nvidia_smi: {nvidia_smi_info}")
        
    #     if return_paths:
    #         return X_batch, y_batch, paths_batch
        
    #     return X_batch, y_batch

    #################


    def __load_batch(self, paths, labels, return_paths=False):
        'Generates data containing batch_size samples' 
        X_batch = []
        y_batch = []
        paths_batch = []
        
        # Generate data
        for i, path in enumerate(paths):
            # logging.info(f"Path: {path}")
            imgs = np.load(path)
            
            n_tiles = imgs.shape[0]
            
            augmented_images = []
            
            # if self.flip or self.rot:
            #     for j in range(n_tiles): 
            #         # Augmentations
            #         img = deepcopy(imgs[j]) if not self.is_aug_inplace else imgs[j]
            #         if self.flip:
            #             img = random_horizontal_flip(img) 
            #             img = random_vertical_flip(img) 
            #         if self.rot:
            #             img = random_choice_rotate(img) 
                    
            #         if self.is_aug_inplace:
            #             imgs[j] = img
            #         elif not np.array_equal(img, imgs[j]): 
            #             augmented_images.append(img) 
        
            # Store sample - all the tiles in site
            X_batch.append(imgs)
            y_batch.append([labels[i]]*n_tiles)
            if return_paths:
                paths_batch.append([path]*n_tiles)
            
            # if not self.is_aug_inplace:
            #     # Append augmented images
            #     if len(augmented_images) > 0: 
            #         augmented_images = np.stack(augmented_images) 
            #         X_batch.append(augmented_images) 
            #         y_batch.append([labels[i]]*len(augmented_images)) 
            #         if return_paths:
            #             paths_batch.append([path]*len(augmented_images))
        
        X_batch = np.concatenate(X_batch)
        y_batch = np.asarray(utils.flat_list_of_lists(y_batch))
        if return_paths:
            paths_batch = np.asarray(utils.flat_list_of_lists(paths_batch))

        # If the channel axis is the last one, move it to be the second one
        # (#tiles, 100, 100, #channel) -> (#tiles, #channel, 100, 100)
        if np.argmin(X_batch.shape[1:]) == len(X_batch.shape) - 2:
            X_batch = np.moveaxis(X_batch, -1, 1)

        # logging.info(f"y_batch shape: {y_batch.shape}")
        
        # logging.info(f"\n\n [load_batch]  X_batch: {X_batch.shape}, y [{np.unique(y_batch)}]: {y_batch.shape}")

        # res = utils.apply_for_all_gpus(utils.getfreegpumem)
        # logging.info(f"Resources (Free, Used, Total): {res}")
        # nvidia_smi_info = utils.apply_for_all_gpus(utils.get_nvidia_smi_output)
        # logging.info(f"nvidia_smi: {nvidia_smi_info}")
        
        if return_paths:
            return X_batch, y_batch, paths_batch
        
        return X_batch, y_batch
        
    def id2label(self, y_id):
        # y_label = self.unique_classes[y_id.flatten().astype(int)]
        y_label = self.unique_markers[y_id.flatten().astype(int)]
        
        return y_label
    
    def __label_converter(self, y, label_format='index'):
        # if self.unique_classes is None:
        if self.unique_markers is None:
            # raise ValueError('unique_classes is empty.')
            raise ValueError('unique_markers is empty.')
        else:
            y = y.reshape(-1,)
            # onehot = y[:, None] == self.unique_classes
            onehot = y[:, None] == self.unique_markers
            if label_format == 'onehot':
                output = onehot
            elif label_format == 'index':
                output = onehot.argmax(1)
            else:
                output = y

            output = output.reshape(-1,1)
            return output
    
    @staticmethod
    def get_collate_fn(shuffle=False):
        def collate_fn(batch):
            if 'image_global' in batch[0]:
            
                res = utils.apply_for_all_gpus(utils.getfreegpumem)
                logging.info(f"Resources (Free, Used, Total): {res}")
                
                images_global = [b['image_global'] for b in batch]
                images_local = [b['image_local'] for b in batch]
                labels = [b['label'] for b in batch]
                labels = np.asarray(utils.flat_list_of_lists(labels))
                # labels = np.asarray(labels).reshape(-1,)

                # images_paths = [b['image_path'] for b in batch]
                
                # print(len(images), len(images[0]), images[0][0].shape)
                # print(len(images_global), images_global[0][0].shape, images_global[1][0].shape)
                output_images_global = torch.from_numpy(np.vstack(images_global))
                output_images_local = torch.from_numpy(np.vstack(images_local))
                # output_images = images
                # output_labels_global = [torch.from_numpy(np.vstack(labels).reshape(-1,))]
                # output_labels_global = np.asarray(output_labels_global)
                # output_labels_global = np.vstack(labels).reshape(-1,)
                
                output_labels_global = labels
                
                # output_paths = [np.vstack(images_paths).reshape(-1,)] * len(output_images)
                
                # logging.warn("----------------NO SHUFFLING IS APPLIED CURRENTLY!----------")
                if shuffle:
                    indexes = np.arange(len(output_images_global))
                    np.random.shuffle(indexes)
                    output_images_global = output_images_global[indexes]
                    output_images_local = output_images_local[indexes]
                    output_labels_global = output_labels_global[indexes]
                    
                # SAGY - TRAINING WITH various channels
                # changable_channel_count = batch[0]['changable_channel_count']
                # if changable_channel_count:
                #     channels_count = output_images_global.shape[1] 
                #     _optional_markers_indx = np.arange(0, channels_count  , 2) # 2 since we have target and nucleus per marker
                #     n_markers_to_choose = np.random.randint(1, len(_optional_markers_indx)+1,1)[0]
                #     markers_chosen = np.random.choice(_optional_markers_indx, size=n_markers_to_choose, replace=False)
                #     markers_chosen.sort()
                #     channels_chosen = [[m, m+1] for m in markers_chosen]
                #     channels_chosen = utils.flat_list_of_lists(channels_chosen)
                #     channels_chosen = np.asarray(channels_chosen)
                #     output_images_global = output_images_global[:,channels_chosen,...]
                #     output_images_local = output_images_local[:,channels_chosen,...]
                ####
                
                # if shuffle:
                #     # Ensure all tensors have the same number of samples
                #     n_samples = output_images[0].shape[0]

                #     # Generate a random permutation of indices
                #     permutation = torch.randperm(n_samples)
                #     # print(f"permutation: {permutation}")
                #     # Shuffle each tensor using the same permutation
                #     output_images = [tensor[permutation] for tensor in output_images]
                #     output_labels = [tensor[permutation] for tensor in output_labels]
                #     output_paths = [tensor[permutation] for tensor in output_paths]
                    
                    # indexes = np.arange(len(output_images[0]))
                    # np.random.shuffle(indexes)
                    # output_images = output_images[indexes]
                    # output_labels = output_labels[indexes]
                    # output_paths = output_paths[indexes]
                
                # print(f"Image shape: {len(output_images)}, label shape: {len(output_labels)} {output_labels[0].shape}, label unique: {np.unique(output_labels[0])}")
                # assert output_images.shape[0] == output_labels.shape[0] == output_paths.shape[0] 
                # print(f"Image global shape: {output_images_global.shape}, Image local shape: {output_images_local.shape}")
                print(f"Image global shape: {output_images_global.shape}, Image local shape: {output_images_local.shape}, output_labels_global global shape: {output_labels_global.shape}")
                
                # print(f"Image shape: {output_images.shape}, label shape: {output_labels.shape}, label unique: {np.unique(output_labels)}")
                # return {'image': output_images, 'label': output_labels, 'image_path': output_paths}
                # return {'image_global': output_images_global, 'image_local': output_images_local}
                print(f"Unique labels: {np.unique(output_labels_global)}")
                return {'image_global': output_images_global, 'image_local': output_images_local, 'labels_global': output_labels_global}

            else:
                # For inference
                res = utils.apply_for_all_gpus(utils.getfreegpumem)
                logging.info(f"Resources (Free, Used, Total): {res}")

                images = [b['image'] for b in batch]
                labels = [b['label'] for b in batch]
                # print(f"labels={labels}")
                images_paths = [b['image_path'] for b in batch]
                
                output_images = torch.from_numpy(np.vstack(images))
                output_labels = torch.from_numpy(np.vstack(labels).reshape(-1,))
                output_paths = np.vstack(images_paths).reshape(-1,)
                
                if shuffle:
                    indexes = np.arange(len(output_images))
                    np.random.shuffle(indexes)
                    output_images = output_images[indexes]
                    output_labels = output_labels[indexes]
                    output_paths = output_paths[indexes]
                    
                # SAGY - TRAINING WITH various channels
                # channels_count = output_images.shape[1] 
                # _optional_markers_indx = np.arange(0, channels_count  , 2) # 2 since we have target and nucleus per marker
                # n_markers_to_choose = np.random.randint(1, len(_optional_markers_indx)+1,1)[0]
                # markers_chosen = np.random.choice(_optional_markers_indx, size=n_markers_to_choose, replace=False)
                # markers_chosen.sort()
                # channels_chosen = [[m, m+1] for m in markers_chosen]
                # channels_chosen = utils.flat_list_of_lists(channels_chosen)
                # channels_chosen = np.asarray(channels_chosen)
                # output_images = output_images[:,channels_chosen,...]
                ####
                
                assert output_images.shape[0] == output_labels.shape[0]# == output_paths.shape[0] 

                # print(f"Image shape: {output_images.shape}, label shape: {output_labels.shape}")
                logging.info(f"Image shape: {output_images.shape}, label shape: {output_labels.shape}")

                # logging.info(f"Image shape: {output_images.shape}, label shape: {output_labels.shape}, label unique: {np.unique(output_labels)}")
                return {'image': output_images, 'label': output_labels, 'image_path': output_paths}
                # return {'image': output_images, 'label': output_labels}


        
        return collate_fn
    
    def split(self):
        """
        Split data by set (train/val/test)
        """
        # Dummy: df = pd.DataFrame([[0, '../A/Site1#2', 'A'], [1, "../A/Site1#3", 'A'], [100, '../B/Site3#4', 'B'], [101, '../B/Site4#6', 'B']], columns=['index', 'path', 'label'])
        # df.loc[:,'paths_sites'] = df['paths'].apply(lambda x: x.split('#')[0])
        # df.groupby('paths_sites').agg({"index": lambda x: list(x), "labels": lambda x: np.unique(x)})
        X_indexes, y = np.arange(len(self.X_paths)), self.y
        train_size = self.train_part
        random_state = self.conf.SEED
        shuffle = self.shuffle
        
        # # For stratification - check train_size is valid
        # n_classes = len(np.unique(y))
        # min_train_size = 1 - n_classes/len(X_indexes)
        # if train_size < min_train_size:
        #     logging.warning(f"'train_size' is lower than allowed. Setting it to be the minimum possible: {min_train_size}")
        #     train_size = min_train_size
    
        # First, split the data in training and remaining dataset
        logging.info(f"Split data by set (train/val/test): {train_size}")
        # logging.info(f"!!! test size is set to 0.3 !!!!")
        X_train_indexes, X_temp_indexes, y_train, y_temp = train_test_split(X_indexes, y,
                                                                            train_size=train_size,
                                                                            random_state=random_state,
                                                                            shuffle=shuffle,
                                                                            stratify=y)

        # The valid and test size to be equal (that is 50% of remaining data)
        X_valid_indexes, X_test_indexes, y_valid, y_test = train_test_split(X_temp_indexes, y_temp,
                                                                            test_size=0.5,
                                                                            random_state=random_state,
                                                                            shuffle=shuffle,
                                                                            stratify=y_temp)
        
        logging.info(f"Train set: {len(X_train_indexes)} {len(y_train)}")
        logging.info(f"Validation set: {len(X_valid_indexes)} {len(y_valid)}")
        logging.info(f"Test set: {len(X_test_indexes)} {len(y_test)}")
        # Markers order
        train_markers_order = [label[0].split('_')[0] for label in y_train]
        valid_markers_order = [label[0].split('_')[0] for label in y_valid]
        test_markers_order = [label[0].split('_')[0] for label in y_test]
        
        logging.warning("\n\nTODO: Sagy, to implment _labels_changepoints (if needed) + return markers_order!")
        
        # self.train_data, self.train_label, self.train_markers_order = None = X_train, y_train, train_markers_order
        # self.val_data, self.val_label, self.valid_markers_order = X_valid, y_valid, valid_markers_order
        # self.test_data, self.test_label, self.test_markers_order = X_test, y_test, test_markers_order
        
        # self.X_train, self.y_train = X_train, y_train
        # self.X_valid, self.y_valid = X_valid, y_valid
        # self.X_test, self.y_test = X_test, y_test
        # self.unique_markers = unique_markers
        
        # self.train_markers_order = train_markers_order
        # self.valid_markers_order = valid_markers_order
        # self.test_markers_order = test_markers_order
        
        return X_train_indexes, X_valid_indexes, X_test_indexes
