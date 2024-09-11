import logging
import os
import random
import sys
import torch
from typing import Dict, List, Tuple
import numpy as np
from info_nce import InfoNCE

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.lib.models.NOVA_model import NOVAModel
from src.common.lib.models.trainers.trainer_base import TrainerBase
from src.common.configs.trainer_config import TrainerConfig
from src.common.lib.utils import flat_list_of_lists, get_if_exists
from src.common.configs.dataset_config import DatasetConfig
from src.datasets import label_utils


class _LabelInfo:
    """Holds all the needed info on the label for generating pairs for the contrastive loss
    """
    def __init__(self, label:str, index:int, dataset_config:DatasetConfig):
        __labels_np = np.asarray([label])
        self.batch:str = label_utils.get_batches_from_labels(__labels_np, dataset_config)[0]
        self.cell_line_cond:str = label_utils.get_cell_lines_conditions_from_labels(__labels_np, dataset_config)[0]
        self.marker:str = label_utils.get_markers_from_labels(__labels_np)[0]
        self.rep:str = label_utils.get_reps_from_labels(__labels_np, dataset_config)[0]
        self.index:int = index

class TrainerContrastive(TrainerBase):
    def __init__(self, trainer_config:TrainerConfig, nova_model:NOVAModel):
        """Get an instance

        Args:
            conf (TrainerConfig): The trainer configuration
            nova_model (NOVAModel): The NOVA model to train
        """
        super().__init__(trainer_config, nova_model)
        
        self.negative_count:int = get_if_exists(self.trainer_config, 'NEGATIVE_COUNT', 5, verbose=True)
        self.pretrained_model_path:str = get_if_exists(self.trainer_config, 'PRETRAINED_MODEL_PATH', None, verbose=True)
        
        self.loss_infoNCE:InfoNCE = InfoNCE(negative_mode = 'paired')
        
        if self.pretrained_model_path is not None:
            # Handle fine-tuning
            self.__load_weights_from_pretrained_model()
            self.__try_freeze_layers()
        
    def loss(self, embeddings:torch.Tensor, anchor_idx:List[int], positive_idx:List[int], negative_idx:List[int])->float:
        """Calculating the loss value

        Args:
            embeddings (torch.Tensor): The embeddings
            anchor_idx (List[int]): The indexes for the anchors
            positive_idx (List[int]): The indexes for the positive samples
            negative_idx (List[int]): The indexes for the negative samples

        Returns:
            float: The loss value
        """
        
        embeddings_size = embeddings.shape[1]
        
        query = embeddings[anchor_idx]
        positive = embeddings[positive_idx]
        negative = embeddings[torch.as_tensor(negative_idx)]

        assert query.shape[1] == positive.shape[1] == negative.shape[2] == embeddings_size
        assert query.shape[0] == positive.shape[0] == negative.shape[0]
        assert negative.shape[1] == self.negative_count

        
        return self.loss_infoNCE(query, positive, negative)
    
    def forward(self, X: torch.Tensor, y: torch.Tensor=None, paths: np.ndarray[str]=None) -> Dict:
        """Applying the forward pass (running the model on the given data)

        Args:
            X (torch.Tensor): The data to feed into the model
            y (torch.Tensor, optional): The ids for the labels. Defaults to None.
            paths (np.ndarray[str], optional): The paths to the files. Defaults to None.
        Returns:
            Dict: {
                    embeddings: The model outputs,
                    anchor_idx: The indexes for the anchors,
                    positive_idx: The indexes for the positives,
                    negative_idx: The indexes for the negatives
                }
        """
        logging.info(f"X shape: {X.shape}, paths shape: {y.shape}")

        # first we need to try and pair for each image (anchor), a positive and $negative_count negatives
        labels = self.data_loader_train.dataset.id2label(y)
        anchor_idx, positive_idx, negative_idx = self.__pair_labels(labels, self.negative_count) 
        logging.info(f'Found {len(anchor_idx)}/{X.shape[0]} anchors')

        # get all possible indexes
        all_idx = np.unique(anchor_idx + list(np.unique(flat_list_of_lists(negative_idx))) + positive_idx)
        
        # now we want to create embeddings only for the images that can be used as anchor/positive/negative
        X = X[all_idx]
        embeddings = self.nova_model.model(X)

        # because we took only the images that can be used as anchor/positive/negative, now the original indices are not true anymore and we need to convert them
        sorter = np.argsort(all_idx)
        anchor_idx = sorter[np.searchsorted(all_idx, anchor_idx, sorter=sorter)]
        positive_idx = sorter[np.searchsorted(all_idx, positive_idx, sorter=sorter)]
        negative_idx = sorter[np.searchsorted(all_idx, negative_idx, sorter=sorter)] 
        
        return {
            'embeddings': embeddings,
            'anchor_idx': anchor_idx,
            'positive_idx': positive_idx,
            'negative_idx': negative_idx
        }
    
    def __get_positives(self, anchor:_LabelInfo, labels_dicts: List[_LabelInfo])->List[int]:
        """given an anchor, we define positive as:
        # the same marker, batch, cell line, cond
        # different rep

        Args:
            anchor (_LabelInfo): The anchor
            labels_dicts (List[_LabelInfo]): List of the labels information

        Returns:
            List[int]: List of indexes for negative samples
        """

        positive_marker = anchor.marker
        positive_batch = anchor.batch
        positive_cell_line_cond = anchor.cell_line_cond

        positives = [i for i, lbl in enumerate(labels_dicts) if lbl.marker == positive_marker \
                and lbl.cell_line_cond == positive_cell_line_cond \
                and lbl.batch == positive_batch \
                and lbl.index != anchor.index]
        return positives
        
    def __get_negatives(self, anchor:_LabelInfo, labels_dicts: List[_LabelInfo])->List[int]:
        """given an anchor, we define negative as:
        the same marker, batch
        different cell line, cond

        Args:
            anchor (_LabelInfo): The anchor
            labels_dicts (List[_LabelInfo]): List of the labels information

        Returns:
            List[int]: List of indexes for negative samples
        """
        # 
        
        negative_marker = anchor.marker
        negative_batch = anchor.batch
        negatives = [i for i, lbl in enumerate(labels_dicts) if lbl.marker == negative_marker \
                and lbl.batch == negative_batch \
                and lbl.cell_line_cond != anchor.cell_line_cond]

        return negatives

    def __pair_labels(self, labels:List[str], negative_count:int=5)->Tuple[List[int], List[int], List[List[int]]]:
        """this function gets all the labels in the batch, and pairs all the anchor-positive-negatives it can.

        Args:
            labels (List[str]): The labels
            negative_count (int, optional): Number of negatives per pair. Defaults to 5.
            
        Returns:
            Tuple[List[int], List[int], List[int]]:\n
                - anchor_idx(List[int]) : indices of labels that can be used as anchors\n
                - positive_idx(List[int]) : indices of labels that can be used as positive to the anchors in the same location\n
                - negative_idx(List[List[int]]) : lists of indices of labels that can be used as negatives to the anchors in the same location\n
                \n
                for example:\n
                anchor_idx = [1]\n
                positive_idx = [10]\n
                negative_idx = [[3,6,8,12,20]]\n
                \n
                for the embeddings in index 1, the embeddings in index 10 can be used as positive. \n
                And, the embeddings in indices [3,6,8,12,20] can be used as negatives.
        """
        dataset_config = self.data_loader_train.dataset.get_configuration()
        labels_dicts = [_LabelInfo(label, i, dataset_config) for i,label in enumerate(labels)]        
        
        anchor_idx, positive_idx, negative_idx = [], [] , []
        for index, anchor in enumerate(labels_dicts):

            positives = self.__get_positives(anchor, labels_dicts)
            if len(positives)==0:
                # No positives were found
                continue
            
            # Sample one positive
            positive_random = random.sample(list(positives),1)[0]
            
            negatives = self.__get_negatives(anchor, labels_dicts)
            if len(negatives) < negative_count:
                # Not enough negatives were found
                continue
            
            # Sample negatives from the detected ones
            negatives_random = random.sample(list(negatives), negative_count)
            
            anchor_idx.append(index)
            positive_idx.append(positive_random)
            negative_idx.append(negatives_random)

        assert len(anchor_idx) == len(positive_idx) == len(negative_idx), "Mismatch in triplets of anchor, positive, negatives"
        assert len(negative_idx[0])==negative_count, f"Mismatch in number of negatives, expected {negative_count}, but got {len(negative_idx[0])}"

        return anchor_idx, positive_idx, negative_idx
    
    def __try_freeze_layers(self):
        """Trying to freeze layers based on the LAYERS_TO_FREEZE param in the config, if exists
        """
        layers_to_freeze = get_if_exists(self.trainer_config, 'LAYERS_TO_FREEZE', None)
        if layers_to_freeze is None or len(layers_to_freeze) == 0:
            # No layers to freeze
            return
        
        # Freezing the layers
        logging.info(f"Freeze layers: {layers_to_freeze}")
        _freezed_layers = self._freeze_layers(layers_to_freeze)
        logging.info(f"Layers freezed successfully : {_freezed_layers}")
            
    def __load_weights_from_pretrained_model(self):
        """Loads the weights from a given pretrained model path, while changing the output dimension of the head to the new output_dim
        """
        if self.pretrained_model_path is None:
            logging.warning("'pretrained_model_path' was set to None. Can't load pretrained model.")
            return
        
        logging.info(f"Loading pretrained model ({self.pretrained_model_path})")
        pretrained_model = NOVAModel.load_from_checkpoint(self.pretrained_model_path).model
        
        # Modifying the head's output dim 
        logging.info(f"Changing the head output dim from {pretrained_model.head.out_features} to {self.nova_model.model_config.OUTPUT_DIM}")
        pretrained_model.head = torch.nn.Linear(pretrained_model.head.in_features, self.nova_model.model_config.OUTPUT_DIM)
        
        # Set the modified pretrained model to be the starting point for our model
        self.nova_model.model = pretrained_model
        logging.info(f"The updated head is: {self.nova_model.model.head}")