import logging
import os
import random
import sys
import torch
from typing import Dict, List, Tuple
import numpy as np
from info_nce import InfoNCE

sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.models.architectures.NOVA_model import NOVAModel
from src.models.trainers.trainer_base import TrainerBase
from src.models.trainers.trainer_config import TrainerConfig
from src.common.utils import flat_list_of_lists, get_if_exists
from src.datasets.label_utils import LabelInfo

class TrainerContrastive(TrainerBase):
    def __init__(self, trainer_config:TrainerConfig, nova_model:NOVAModel):
        """Get an instance

        Args:
            conf (TrainerConfig): The trainer configuration
            nova_model (NOVAModel): The NOVA model to train
        """
        super().__init__(trainer_config, nova_model)
        
        self.negative_count:int = get_if_exists(self.trainer_config, 'NEGATIVE_COUNT', 5, verbose=True)
        
        self.loss_infoNCE:InfoNCE = InfoNCE(negative_mode = 'paired')
        
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
    
    def __get_positives(self, anchor:LabelInfo, labels_dicts: List[LabelInfo])->List[int]:
        """given an anchor, we define positive as:
        # the same marker, batch, cell line, cond
        # different rep

        Args:
            anchor (LabelInfo): The anchor
            labels_dicts (List[LabelInfo]): List of the labels information

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
        
    def __get_negatives(self, anchor:LabelInfo, labels_dicts: List[LabelInfo])->List[int]:
        """given an anchor, we define negative as:
        the same marker, batch
        different cell line, cond

        Args:
            anchor (LabelInfo): The anchor
            labels_dicts (List[LabelInfo]): List of the labels information

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
        labels_dicts = [LabelInfo(label, dataset_config, i) for i,label in enumerate(labels)]        
        
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
