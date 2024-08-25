import logging
import os
import random
import sys
from common.lib.models.trainers import fine_tuning_utils
import torch
from typing import Dict, List, Self
import numpy as np
from info_nce import InfoNCE

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from src.common.lib.models.trainers.trainer_base import TrainerBase
from src.common.configs.trainer_config import TrainerConfig
from src.common.lib.utils import flat_list_of_lists, get_if_exists

class _LabelInfo:
        def __init__(self,
                     batch:str,
                     cell_line_cond:str,
                     marker:str,
                     rep:str,
                     site:str,
                     index:int):
            self.batch:str = batch
            self.cell_line_cond:str = cell_line_cond
            self.marker:str = marker
            self.rep:str = rep
            self.site:str = site
            self.index:int = index

class TrainerContrastive(TrainerBase):
    def __init__(self, conf:TrainerConfig)->Self:
        super().__init__(conf)
        
        self.negative_count = get_if_exists(self.training_config, 'NEGATIVE_COUNT', 5)
        self.loss_infoNCE = InfoNCE(negative_mode = 'paired')
        
        self.__try_freezing_layers()
        
    def loss(self, embeddings:torch.Tensor[float], anchor_idx:List[int], positive_idx:List[int], negative_idx:List[int])->float:
        """Calculating the loss value

        Args:
            embeddings (torch.Tensor[float]): The embeddings
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
    
    def forward(self, X:torch.Tensor) -> Dict:
        """Applying the forward pass
        """
        with torch.cuda.amp.autocast():
            images = X['image'].to(torch.float).cuda()
            paths = X['image_path']
            logging.info(f"images shape: {images.shape}, paths shape: {paths.shape}")

            # first we need to try and pair for each image (anchor), a positive and $negative_count negatives
            anchor_idx, positive_idx, negative_idx = self.__pair_labels(paths, self.negative_count) 
            logging.info(f'[InfoNCE] found {len(anchor_idx)}/{images.shape[0]} anchors]')

            all_idx = np.unique(anchor_idx + list(np.unique(flat_list_of_lists(negative_idx))) + positive_idx)
            
            # now we want to create embeddings only for the images that can be used as anchor/positive/negative
            embeddings = self.nova_model(images[all_idx])

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

    def __pair_labels(self, paths:List[str], negative_count:int=5):
        """
        this function gets all the paths (=labels) in the batch, 
        and pairs all the anochr-positive-negatives it can.

        output:
            anchor_idx(list) : indices of labels that can be used as anchors
            positive_idx(list) : indices of labels that can be used as positive to the anchors in the same location
            negative_idx(list of lists) : lists of indices of labels that can be used as negatives to the anchors in the same location

            for example:
            anchor_idx = [1]
            positive_idx = [10]
            negative_idx = [[3,6,8,12,20]]

            for the embeddings in index 1, the embeddings in index 10 can be used as positive. 
            And, the embeddings in indices [3,6,8,12,20] can be used as negatives.
        """
        labels_list = [p.split(os.sep)[-5:] for p in paths]
        labels_dicts = [_LabelInfo(
                            batch=l[0],
                            cell_line_cond='_'.join(l[1:3]),
                            marker=l[3],
                            rep=l[4].split('_')[0],
                            site=l[4].split('_')[3],
                            index=i)
                        for i,l in enumerate(labels_list)]
        anchor_idx, positive_idx, negative_idx = [], [] , []
        for index, anchor in enumerate(labels_dicts):

            positives = self.__get_positives(anchor, labels_dicts)
            if len(positives)==0:
                continue
            else:
                positive_random = random.sample(list(positives),1)[0]
            
            negatives = self.__get_negatives(anchor, labels_dicts)
            if len(negatives) < negative_count:
                continue
            else:
                negatives_random = random.sample(list(negatives), negative_count)
                anchor_idx.append(index)
                positive_idx.append(positive_random)
                negative_idx.append(negatives_random)

        assert len(anchor_idx) == len(positive_idx) == len(negative_idx)
        assert len(negative_idx[0])==negative_count

        return anchor_idx, positive_idx, negative_idx
    
    def __try_freezing_layers(self):
        """Trying to freeze layers based on the LAYERS_TO_FREEZE param in the config, if exists
        """
        layers_to_freeze = get_if_exists(self.training_config, 'LAYERS_TO_FREEZE', None)
        if layers_to_freeze is None or len(layers_to_freeze) == 0:
            # No layers to freeze
            return
        
        # Freezing the layers
        logging.info(f"Freezing layers: {layers_to_freeze}")
        _freezed_layers = fine_tuning_utils.freeze_layers(self.nova_model, layers_to_freeze)
        logging.info(f"Layers freezed successfully : {_freezed_layers}")
            