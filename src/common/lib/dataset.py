
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
    def __init__(self, conf: DatasetConfig):
        self.__set_params(conf)
        
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
        
        self.conf = conf
        
        self.X_paths, self.y, self.unique_markers = self._load_data_paths()  
        
        self.unique_markers = ['ANXA11_FUSHeterozygous_Untreated', 'CD41_FUSHeterozygous_Untreated',
                            'CLTC_FUSHeterozygous_Untreated', 'Calreticulin_FUSHeterozygous_Untreated',
                            'DCP1A_FUSHeterozygous_Untreated', 'FMRP_FUSHeterozygous_Untreated',
                            'FUS_FUSHeterozygous_Untreated', 'G3BP1_FUSHeterozygous_Untreated',
                            'GM130_FUSHeterozygous_Untreated', 'KIF5A_FUSHeterozygous_Untreated',
                            'LAMP1_FUSHeterozygous_Untreated', 'NCL_FUSHeterozygous_Untreated',
                            'NEMO_FUSHeterozygous_Untreated', 'NONO_FUSHeterozygous_Untreated',
                            'PEX14_FUSHeterozygous_Untreated', 'PML_FUSHeterozygous_Untreated',
                            'PSD95_FUSHeterozygous_Untreated', 'PURA_FUSHeterozygous_Untreated',
                            'Phalloidin_FUSHeterozygous_Untreated', 'SCNA_FUSHeterozygous_Untreated',
                            'SQSTM1_FUSHeterozygous_Untreated', 'TDP43_FUSHeterozygous_Untreated',
                            'TIA1_FUSHeterozygous_Untreated', 'TOMM20_FUSHeterozygous_Untreated',
                            'mitotracker_FUSHeterozygous_Untreated', 'ANXA11_FUSHomozygous_Untreated',
                            'CD41_FUSHomozygous_Untreated', 'CLTC_FUSHomozygous_Untreated',
                            'Calreticulin_FUSHomozygous_Untreated', 'DCP1A_FUSHomozygous_Untreated',
                            'FMRP_FUSHomozygous_Untreated', 'FUS_FUSHomozygous_Untreated',
                            'G3BP1_FUSHomozygous_Untreated', 'GM130_FUSHomozygous_Untreated',
                            'KIF5A_FUSHomozygous_Untreated', 'LAMP1_FUSHomozygous_Untreated',
                            'NCL_FUSHomozygous_Untreated', 'NEMO_FUSHomozygous_Untreated',
                            'NONO_FUSHomozygous_Untreated', 'PEX14_FUSHomozygous_Untreated',
                            'PML_FUSHomozygous_Untreated', 'PSD95_FUSHomozygous_Untreated',
                            'PURA_FUSHomozygous_Untreated', 'Phalloidin_FUSHomozygous_Untreated',
                            'SCNA_FUSHomozygous_Untreated', 'SQSTM1_FUSHomozygous_Untreated',
                            'TDP43_FUSHomozygous_Untreated', 'TIA1_FUSHomozygous_Untreated',
                            'TOMM20_FUSHomozygous_Untreated', 'mitotracker_FUSHomozygous_Untreated',
                            'ANXA11_FUSRevertant_Untreated', 'CD41_FUSRevertant_Untreated',
                            'CLTC_FUSRevertant_Untreated', 'Calreticulin_FUSRevertant_Untreated',
                            'DCP1A_FUSRevertant_Untreated', 'FMRP_FUSRevertant_Untreated',
                            'FUS_FUSRevertant_Untreated', 'G3BP1_FUSRevertant_Untreated',
                            'GM130_FUSRevertant_Untreated', 'KIF5A_FUSRevertant_Untreated',
                            'LAMP1_FUSRevertant_Untreated', 'NCL_FUSRevertant_Untreated',
                            'NEMO_FUSRevertant_Untreated', 'NONO_FUSRevertant_Untreated',
                            'PEX14_FUSRevertant_Untreated', 'PML_FUSRevertant_Untreated',
                            'PSD95_FUSRevertant_Untreated', 'PURA_FUSRevertant_Untreated',
                            'Phalloidin_FUSRevertant_Untreated', 'SCNA_FUSRevertant_Untreated',
                            'SQSTM1_FUSRevertant_Untreated', 'TDP43_FUSRevertant_Untreated',
                            'TIA1_FUSRevertant_Untreated', 'TOMM20_FUSRevertant_Untreated',
                            'mitotracker_FUSRevertant_Untreated', 'ANXA11_OPTN_Untreated',
                            'CD41_OPTN_Untreated', 'CLTC_OPTN_Untreated', 'Calreticulin_OPTN_Untreated',
                            'DCP1A_OPTN_Untreated', 'FMRP_OPTN_Untreated', 'FUS_OPTN_Untreated',
                            'G3BP1_OPTN_Untreated', 'GM130_OPTN_Untreated', 'KIF5A_OPTN_Untreated',
                            'LAMP1_OPTN_Untreated', 'NCL_OPTN_Untreated', 'NEMO_OPTN_Untreated',
                            'NONO_OPTN_Untreated', 'PEX14_OPTN_Untreated', 'PML_OPTN_Untreated',
                            'PSD95_OPTN_Untreated', 'PURA_OPTN_Untreated', 'Phalloidin_OPTN_Untreated',
                            'SCNA_OPTN_Untreated', 'SQSTM1_OPTN_Untreated', 'TDP43_OPTN_Untreated',
                            'TIA1_OPTN_Untreated', 'TOMM20_OPTN_Untreated',
                            'mitotracker_OPTN_Untreated', 'ANXA11_SCNA_Untreated',
                            'CD41_SCNA_Untreated', 'CLTC_SCNA_Untreated', 'Calreticulin_SCNA_Untreated',
                            'DCP1A_SCNA_Untreated', 'FMRP_SCNA_Untreated', 'FUS_SCNA_Untreated',
                            'G3BP1_SCNA_Untreated', 'GM130_SCNA_Untreated', 'KIF5A_SCNA_Untreated',
                            'LAMP1_SCNA_Untreated', 'NCL_SCNA_Untreated', 'NEMO_SCNA_Untreated',
                            'NONO_SCNA_Untreated', 'PEX14_SCNA_Untreated', 'PML_SCNA_Untreated',
                            'PSD95_SCNA_Untreated', 'PURA_SCNA_Untreated', 'Phalloidin_SCNA_Untreated',
                            'SCNA_SCNA_Untreated', 'SQSTM1_SCNA_Untreated', 'TDP43_SCNA_Untreated',
                            'TIA1_SCNA_Untreated', 'TOMM20_SCNA_Untreated',
                            'mitotracker_SCNA_Untreated', 'ANXA11_TBK1_Untreated',
                            'CD41_TBK1_Untreated', 'CLTC_TBK1_Untreated', 'Calreticulin_TBK1_Untreated',
                            'DCP1A_TBK1_Untreated', 'FMRP_TBK1_Untreated', 'FUS_TBK1_Untreated',
                            'G3BP1_TBK1_Untreated', 'GM130_TBK1_Untreated', 'KIF5A_TBK1_Untreated',
                            'LAMP1_TBK1_Untreated', 'NCL_TBK1_Untreated', 'NEMO_TBK1_Untreated',
                            'NONO_TBK1_Untreated', 'PEX14_TBK1_Untreated', 'PML_TBK1_Untreated',
                            'PSD95_TBK1_Untreated', 'PURA_TBK1_Untreated', 'Phalloidin_TBK1_Untreated',
                            'SCNA_TBK1_Untreated', 'SQSTM1_TBK1_Untreated', 'TDP43_TBK1_Untreated',
                            'TIA1_TBK1_Untreated', 'TOMM20_TBK1_Untreated',
                            'mitotracker_TBK1_Untreated', 'ANXA11_TDP43_Untreated',
                            'CD41_TDP43_Untreated', 'CLTC_TDP43_Untreated',
                            'Calreticulin_TDP43_Untreated', 'DCP1A_TDP43_Untreated',
                            'FMRP_TDP43_Untreated', 'FUS_TDP43_Untreated', 'G3BP1_TDP43_Untreated',
                            'GM130_TDP43_Untreated', 'KIF5A_TDP43_Untreated', 'LAMP1_TDP43_Untreated',
                            'NCL_TDP43_Untreated', 'NEMO_TDP43_Untreated', 'NONO_TDP43_Untreated',
                            'PEX14_TDP43_Untreated', 'PML_TDP43_Untreated', 'PSD95_TDP43_Untreated',
                            'PURA_TDP43_Untreated', 'Phalloidin_TDP43_Untreated',
                            'SCNA_TDP43_Untreated', 'SQSTM1_TDP43_Untreated', 'TDP43_TDP43_Untreated',
                            'TIA1_TDP43_Untreated', 'TOMM20_TDP43_Untreated',
                            'mitotracker_TDP43_Untreated', 'ANXA11_WT_Untreated', 'CD41_WT_Untreated',
                            'CLTC_WT_Untreated', 'Calreticulin_WT_Untreated', 'DCP1A_WT_Untreated',
                            'FMRP_WT_Untreated', 'FUS_WT_Untreated', 'G3BP1_WT_Untreated',
                            'GM130_WT_Untreated', 'KIF5A_WT_Untreated', 'LAMP1_WT_Untreated',
                            'NCL_WT_Untreated', 'NEMO_WT_Untreated', 'NONO_WT_Untreated',
                            'PEX14_WT_Untreated', 'PML_WT_Untreated', 'PSD95_WT_Untreated',
                            'PURA_WT_Untreated', 'Phalloidin_WT_Untreated', 'SCNA_WT_Untreated',
                            'SQSTM1_WT_Untreated', 'TDP43_WT_Untreated', 'TIA1_WT_Untreated',
                            'TOMM20_WT_Untreated', 'mitotracker_WT_Untreated', 'ANXA11_WT_stress',
                            'CD41_WT_stress', 'CLTC_WT_stress', 'Calreticulin_WT_stress',
                            'DCP1A_WT_stress', 'FMRP_WT_stress', 'FUS_WT_stress', 'G3BP1_WT_stress',
                            'GM130_WT_stress', 'KIF5A_WT_stress', 'LAMP1_WT_stress', 'NCL_WT_stress',
                            'NEMO_WT_stress', 'NONO_WT_stress', 'PEX14_WT_stress', 'PML_WT_stress',
                            'PSD95_WT_stress', 'PURA_WT_stress', 'Phalloidin_WT_stress',
                            'SCNA_WT_stress', 'SQSTM1_WT_stress', 'TDP43_WT_stress', 'TIA1_WT_stress',
                            'TOMM20_WT_stress', 'mitotracker_WT_stress']
        self.unique_markers = np.asarray(self.unique_markers)
        
        # PATCH...
        self.label = self.y      
        
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
    
    def __getitem__(self, index):
        logging.info(f"\n\n\n\n __getitem__ {index}")
        'Generate one batch of data'
        
        X_batch, y_batch, paths_batch = self.get_batch(index, return_paths=True)
    
        y_batch = self.__label_converter(y_batch, label_format='index')
        paths_batch = paths_batch.reshape(-1,1)
        
        return {'image': X_batch, 'label': y_batch, 'image_path': paths_batch}
        

    def get_batch(self, indexes, return_paths=False):
        if not isinstance(indexes, list):
            indexes = [indexes]
        logging.info(f"Indexes: {indexes}, {self.X_paths[indexes]}")
        X_paths_batch = self.X_paths[indexes]
        y_batch = self.y[indexes]

        return self.__load_batch(X_paths_batch, y_batch, return_paths=return_paths)

    def __load_batch(self, paths, labels, return_paths=False):
        'Generates data containing batch_size samples' 
        X_batch = []
        y_batch = []
        paths_batch = []
        
        # Generate data
        for i, path in enumerate(paths):
            logging.info(f"Path: {path}")
            imgs = np.load(path)
            
            n_tiles = imgs.shape[0]
            
            augmented_images = []
            
            if self.flip or self.rot:
                for j in range(n_tiles): 
                    # Augmentations
                    img = np.copy(imgs[j]) if not self.is_aug_inplace else imgs[j]
                    if self.flip:
                        img = random_horizontal_flip(img) 
                        img = random_vertical_flip(img) 
                    if self.rot:
                        img = random_choice_rotate(img) 
                    
                    if self.is_aug_inplace:
                        imgs[j] = img
                    elif not np.array_equal(img, imgs[j]): 
                        augmented_images.append(img) 
        
            # Store sample - all the tiles in site
            X_batch.append(imgs)
            y_batch.append([labels[i]]*n_tiles)
            if return_paths:
                paths_batch.append([path]*n_tiles)
            
            if not self.is_aug_inplace:
                # Append augmented images
                if len(augmented_images) > 0: 
                    augmented_images = np.stack(augmented_images) 
                    X_batch.append(augmented_images) 
                    y_batch.append([labels[i]]*len(augmented_images)) 
                    if return_paths:
                        paths_batch.append([path]*len(augmented_images))
        
        X_batch = np.concatenate(X_batch)
        y_batch = np.asarray(utils.flat_list_of_lists(y_batch))
        if return_paths:
            paths_batch = np.asarray(utils.flat_list_of_lists(paths_batch))

        # If the channel axis is the last one, move it to be the second one
        # (#tiles, 100, 100, #channel) -> (#tiles, #channel, 100, 100)
        if np.argmin(X_batch.shape[1:]) == len(X_batch.shape) - 2:
            X_batch = np.moveaxis(X_batch, -1, 1)

        logging.info(f"y_batch shape: {y_batch.shape}")
        
        logging.info(f"\n\n [load_batch]  X_batch: {X_batch.shape}, y [{np.unique(y_batch)}]: {y_batch.shape}")

        
        ###############################################################
        
        res = utils.apply_for_all_gpus(utils.getfreegpumem)
        logging.info(f"Resources (Free, Used, Total): {res}")
        nvidia_smi_info = utils.apply_for_all_gpus(utils.get_nvidia_smi_output)
        logging.info(f"nvidia_smi: {nvidia_smi_info}")
        
        if return_paths:
            return X_batch, y_batch, paths_batch
        
        return X_batch, y_batch
        
    def id2label(self, y_id):
        y_label = self.unique_markers[y_id.flatten().astype(int)]
        
        return y_label
    
    def __label_converter(self, y, label_format='index'):
        if self.unique_markers is None:
            raise ValueError('unique_markers is empty.')
        else:
            y = y.reshape(-1,)
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
            images = [b['image'] for b in batch]
            labels = [b['label'] for b in batch]
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
            
            assert output_images.shape[0] == output_labels.shape[0] == output_paths.shape[0] 

            logging.info(f"Image shape: {output_images.shape}, label shape: {output_labels.shape}, label unique: {np.unique(output_labels)}")
            return {'image': output_images, 'label': output_labels, 'image_path': output_paths}
        
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