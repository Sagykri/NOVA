import datetime
import logging
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer
from cytoself.analysis.analysis_opencell import AnalysisOpenCell
from cytoself.trainer.utils.plot_history import plot_history_cytoself
from cytoself.datamanager.base import DataManagerBase

from src.common.lib import metrics 
from src.common.lib.utils import get_if_exists

from src.common.configs.model_config import ModelConfig

# TODO: (210823) Clean plot_umap!

class Model():

    def __init__(self, conf:ModelConfig):
        self.set_params(conf)
        
        self.train_data = None
        self.train_label = None
        self.train_labels_changepoints = None
        self.train_markers_order = None
        
        self.val_data = None
        self.val_label = None
        self.val_labels_changepoints = None
        self.val_markers_order = None
        
        self.test_data = None
        self.test_label = None
        self.test_labels_changepoints = None
        self.test_markers_order = None
        
        self.train_loader = None
        self.valid_loader = None
        self.test_loader  = None
        
        
        self.model = None
        self.analytics = None
        self.num_class = None
        
    def generate_model_visualization(self, num_class=None, savepath=None):
        savepath = savepath if savepath is not None else os.path.join(self.conf.MODEL_OUTPUT_FOLDER,'model_viz')
        
        logging.info(f"Constructing trainer with num_class={num_class}")
        trainer = self.__construct_trainer(num_class)
        dummy_input = torch.randn(10, 2, 100, 100, device="cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Writing graph to {savepath}")
        writer = torch.utils.tensorboard.SummaryWriter(savepath)
        writer.add_graph(trainer.model, dummy_input)
        writer.close()
        
    def __construct_trainer(self, num_class=None, load_pretrained_model=False):
        pretrained_model_path   = self.pretrained_model_path
        early_stop_patience     = self.early_stop_patience
        learn_rate              = self.learn_rate 
        batch_size              = self.batch_size
        max_epoch               = self.max_epoch
        model_output_folder     = self.conf.MODEL_OUTPUT_FOLDER
        input_shape             = self.conf.INPUT_SHAPE
        emb_shapes              = self.conf.EMB_SHAPES
        output_shape            = self.conf.OUTPUT_SHAPE
        fc_args                 = self.conf.FC_ARGS
        fc_output_idx           = self.conf.FC_OUTPUT_IDX
        vq_args                 = self.conf.VQ_ARGS
        fc_input_type           = self.conf.FC_INPUT_TYPE
        reducelr_patience       = self.conf.REDUCELR_PATIENCE
        reducelr_increment      = self.conf.REDUCELR_INCREMENT
        
        if num_class is None:
            num_class = self.num_class
        
        model_args = {
            'input_shape': input_shape,
            'emb_shapes': emb_shapes, 
            'output_shape': output_shape,
            'fc_args': fc_args, #New
            'fc_output_idx': fc_output_idx,
            'vq_args': vq_args,# NEW
            'num_class': num_class,
            'fc_input_type': fc_input_type
        }
        train_args = {
            'lr': learn_rate,
            'max_epoch': max_epoch,
            'reducelr_patience': reducelr_patience,
            'reducelr_increment': reducelr_increment,
            'earlystop_patience': early_stop_patience,
        }
        
                
        logging.info("Creating the model")
        logging.info(
            f"early_stop_patience={early_stop_patience}, learn_rate={learn_rate},\
                batch_size={batch_size}, max_epoch={max_epoch}")

        logging.info(f"Init model object (fc output: {num_class})")
        trainer = CytoselfFullTrainer(train_args,
                                      homepath=model_output_folder,
                                      model_args=model_args)
        
        if load_pretrained_model and pretrained_model_path is not None and os.path.exists(pretrained_model_path):
            logging.info(f"Loading pretrained model: {pretrained_model_path}")
            pretrained_trainer = self.load_model(pretrained_model_path,
                                                 num_fc_output_classes=1311,
                                                 load_pretrained_model=False)
            logging.info(f"Copy weights")
            self.__copy_weights(pretrained_trainer.model, trainer.model)
        
        return trainer
            
    def __copy_weights(self, model_from, model_to):
        from_state_dict = model_from.state_dict()
        to_state_dict = model_to.state_dict()
        
        for name, param in from_state_dict.items():
            if name not in to_state_dict or to_state_dict[name].shape != param.shape:
                continue
            
            to_state_dict[name] = param

        model_to.load_state_dict(to_state_dict)
              
    def set_params(self, conf:ModelConfig):
        """Set the parameters

        Args:
            conf (ModelConfig): The new values
        """
        try:
            self.pretrained_model_path = conf.PRETRAINED_MODEL_PATH
            
            self.early_stop_patience = conf.EARLY_STOP_PATIENCE
            self.learn_rate = conf.LEARN_RATE
            self.batch_size = conf.BATCH_SIZE
            self.max_epoch = conf.MAX_EPOCH
            
            self.conf = conf
        except Exception as ex:
            logging.error(f"Error with the configuration file. {ex}")
    
    def load_with_dataloader(self, train_loader:DataLoader=None,
                             valid_loader:DataLoader=None, test_loader:DataLoader=None):
        """
        Load data generators for train, val and test
        """
        
        
        # data_loader_factory = DataLoader(self.conf, dataset)
        
        if train_loader is None and valid_loader is None and test_loader is None:
            raise Exception("All loaders are None")
        
        def __set_num_class(loader):
            if self.num_class is None:
                self.num_class = len(loader.dataset.unique_markers)
        
        if train_loader is not None:
            self.train_loader = train_loader    
            __set_num_class(self.train_loader)
        if valid_loader is not None:
            self.valid_loader = valid_loader
            __set_num_class(self.valid_loader)
        if test_loader is not None:
            self.test_loader = test_loader
            __set_num_class(self.test_loader)
        
        data_var = self.conf.DATA_VAR
        self.__init_datamanager_dummy(self.train_loader, self.valid_loader, self.test_loader,
                                      data_var, data_var, data_var)
            
    def __init_datamanager_dummy(self, train_loader, val_loader, test_loader,
                                 train_variance, val_variance, test_variance):
        dm = DataManagerBase(None, None, None)
        dm.train_loader = train_loader
        dm.val_loader = val_loader
        dm.test_loader = test_loader
        dm.train_variance = train_variance
        dm.val_variance = val_variance
        dm.test_variance = test_variance
        
        self.data_manager = dm
    
    def __try_load_trainer(self):
        last_checkpoint_path = get_if_exists(self.conf, 'LAST_CHECKPOINT_PATH', None)
        
        if last_checkpoint_path is None or not os.path.exists(last_checkpoint_path):
            logging.info(f"LAST_CHECKPOINT_PATH is None. Couldn't load trainer from file. Skipping.")
            return 
            
        logging.info(f"LAST_CHECKPOINT_PATH has been detected: {last_checkpoint_path}")
        logging.info("Loading checkpoint to continue from there.")
        
        # If path is folder, load the last checkpoint in the folder
        if os.path.isdir(last_checkpoint_path):
            logging.info(f"LAST_CHECKPOINT_PATH is a folder, hence loading the last created checkpoint inside it")
            __checkpoints = [os.path.join(last_checkpoint_path, f) for f in os.listdir(last_checkpoint_path) if f.endswith('.chkp')]
            
            if len(__checkpoints) == 0:
                logging.warning(f"The folder {last_checkpoint_path} is empty. Couldn't load trainer from file. Skipping.")
                return
            
            __checkpoints.sort(key=os.path.getctime)
            last_checkpoint_path = __checkpoints[-1]
            logging.info(f"Last checkpoint detected is: {last_checkpoint_path}")

        if self.model is not None:
            logging.warning(f"Overriding currently loaded model with {last_checkpoint_path}")
        logging.info(f"Loading checkpoint: {last_checkpoint_path}")
        trainer = self.load_model(last_checkpoint_path, load_pretrained_model=True)
        # Moving to the next epoch
        trainer.current_epoch += 1
        logging.info(f"Checkpoint loaded successfully. Current epoch is: {trainer.current_epoch}")
    
        return trainer
    
    def train_with_dataloader(self):
        """ 
        Train a model on given data
        
        """
        
        trainer = self.__try_load_trainer()
        if trainer is None:
            logging.info("Constructing trainer...")
            trainer = self.__construct_trainer(load_pretrained_model=True)
            
        
        logging.info("Loading params from configuration")
        
        model_output_folder     = self.conf.MODEL_OUTPUT_FOLDER
        
        labels_savepath = os.path.join(model_output_folder, "unique_labels")
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)
        logging.info(f"Saving unique labels to file: {labels_savepath}.npy")
        np.save(labels_savepath, np.unique(self.train_loader.dataset.y))
        
        logging.info("Training the model...")
        trainer.fit(self.data_manager,
                    initial_epoch=trainer.current_epoch,
                    tensorboard_path=f'tb_logs_{datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")}')
        
        logging.info("Plot history...")
        plot_history_cytoself(trainer.history, savepath=os.path.join(model_output_folder, 'visualization'))
        
        logging.info("Finished training the model...")

        self.model = trainer
        
        return self.model

    def load_model(self, model_path=None, num_fc_output_classes=None, load_pretrained_model=False):
        
        """Load model

        Args:
            model_path (string, Optional): 
            num_fc_output_classes (bool, Optional): Number ouf outputs for the fully connected model (number of classes) (Default is number of unique values in labels)

        Raises:
            ValueError: No input image shape
            ValueError: No num_fc_output_classes

        Returns:
            _type_: The model
        """
        
        model_path = model_path if model_path is not None else self.conf.MODEL_PATH
        
        if self.model is not None:
            logging.warning(f"[load_model] Overriding currently loaded model with {model_path}")
        self.model = self.__construct_trainer(num_fc_output_classes, 
                                              load_pretrained_model=load_pretrained_model)
            
        # if os.path.isdir(model_path):
        if os.path.splitext(model_path)[1] == '.chkp':
            # Load checkpoint
            logging.info(f"Loading model from checkpoint {model_path}")
            self.model.load_checkpoint(model_path)
        else:
            # Load model
            logging.info(f"Loading model {model_path}")
            self.model.load_model(model_path, by_weights=False)
        
        return self.model
        
    def load_analytics(self):
        # TODO: WIP!
        """Load Analytics - an API object to cytoself

        Raises:
            ValueError: Model not loaded

        Returns:
            Analytics: The analytics object
        """

        # # Generate ground truth from labels
        # lbl = np.unique(test_label)
        # gt_table = pd.DataFrame([[l, l] for l in lbl], columns=["gene_name", "localization"])

        # logging.info("Ground truth:")
        # logging.info(gt_table['gene_name'].values)

        if self.model is None:
            raise Exception("model is None")
        if self.data_manager is None:
            raise Exception("data_manager is None")

        analytics = AnalysisOpenCell(self.data_manager, self.model)#, gt_table=gt_table)
        
        self.analytics = analytics
        
        return analytics
    
    def generate_dummy_analytics(self):
        analytics = AnalysisOpenCell(None, self.model)
        
        self.analytics = analytics
        
        return analytics

    def generate_reconstructed_image(self, dataloader=None, savepath=None):
        """
        Args:
            savepath (str, Optional): Path to save the images. The default is model_output/visualization/reconstructed_images.png
            dataloader (str, Optional): The default is self.test_loader
        """
        
        if dataloader is None:
            dataloader = self.test_loader
        
        data_ch = ['target', 'nucleus']
        img = next(iter(dataloader))['image'].detach().cpu().numpy()
        torch.cuda.empty_cache()
        reconstructed = self.model.infer_reconstruction(img)
        fig, ax = plt.subplots(2, len(data_ch), figsize=(5 * len(data_ch), 5), squeeze=False)
        for ii, ch in enumerate(data_ch):
            t0 = np.zeros((2 * 100, 5 * 100))
            for i, im in enumerate(img[:10, ii, ...]):
                i0, i1 = np.unravel_index(i, (2, 5))
                t0[i0 * 100 : (i0 + 1) * 100, i1 * 100 : (i1 + 1) * 100] = im
            t1 = np.zeros((2 * 100, 5 * 100))
            for i, im in enumerate(reconstructed[:10, ii, ...]):
                i0, i1 = np.unravel_index(i, (2, 5))
                t1[i0 * 100 : (i0 + 1) * 100, i1 * 100 : (i1 + 1) * 100] = im
            ax[0, ii].imshow(t0, cmap='gray')
            ax[0, ii].axis('off')
            ax[0, ii].set_title('input ' + ch)
            ax[1, ii].imshow(t1, cmap='gray')
            ax[1, ii].axis('off')
            ax[1, ii].set_title('output ' + ch)
        fig.tight_layout()
        
        if savepath is None:
            savepath = os.path.join(self.conf.MODEL_OUTPUT_FOLDER, self.model.savepath_dict['visualization'], 'reconstructed_images.png')
        
        fig.savefig(savepath, dpi=300)
        
        # Calculate MSE
        logging.info(f"Calculating MSE")
        mses = metrics.calculate_mse(img, reconstructed)
        for k,v in mses.items():
            logging.info(f"MSE (ch={k}) = {v}")
            
        return savepath

    # TODO: Move here the feature specturm generation
    def generate_feature_spectrum(self):
        raise NotImplementedError()
        
    def load_embeddings(self, embeddings_type='testset', config_data=None):
        from src.common.lib import embeddings_utils
        
        if config_data is None:
            allowed_embeddings_types = ['trainset', 'valtest', 'testset', 'all']
            assert embeddings_type is not None \
                    and embeddings_type in allowed_embeddings_types,\
                    f"embeddings_type must be one of the following: {allowed_embeddings_types}" 
            
            if embeddings_type == 'trainset':
                config_data = self.train_loader.dataset.conf
            elif embeddings_type == 'valtest':
                config_data = self.valid_loader.dataset.conf
            elif embeddings_type == 'testset':
                config_data = self.test_loader.dataset.conf
            else: #all    
                config_data = self.test_loader.dataset.conf
        
        embeddings, labels = embeddings_utils.load_embeddings(config_model=self.conf,
                                                            config_data=config_data,
                                                            embeddings_type=embeddings_type)

        sample_pct =  get_if_exists(config_data, 'SAMPLE_PCT', None)
        if sample_pct is not None and sample_pct < 1 and sample_pct > 0:
            logging.info(f"[load_embeddings] A valid 'SAMPLE_PCT' has been identified. Sampling embeddings with SAMPLE_PCT={sample_pct}")
            
            labels_indexes = np.arange(len(labels))
            _, labels_indexes_sample = train_test_split(labels_indexes,
                                                        test_size=sample_pct,
                                                        random_state=self.conf.SEED,
                                                        shuffle=config_data.SHUFFLE,
                                                        stratify=labels)
        
            embeddings, labels = embeddings[labels_indexes_sample], labels[labels_indexes_sample]

        
        return embeddings, labels
    
    def set_mode(self, train):
        self.model.model.train(train)
    
    def plot_umap(self,
                  calc_embeddings=False,
                  embeddings_type=None,
                  embedding_data=None,
                  label_data=None,
                  data_loader=None,
                  id2label=None,
                  is_3d=False,
                  title='UMAP',
                  s=0.3,
                  alpha=0.5,
                  reset_umap=False,
                  **kwargs):
        """
        Args:
            Plot (and save to file in model_output/umap_figures/{title}.png) a UMAP plot
            data_loader(DataLoader, Optional): The default is self.test_loader
            calc_embeddings (boolean, Optional): Calculate embeddings (instead of loading them). Defauls to False
            embedding_data (nparray, Optional): Precalculated embeddings.
            label_data (nparray,Optional): Precalculated labels.
            embeddings_type ('trainset'|'testset'|'valtest'|'all', Optional). Must have if calc_embeddings=False. Defaults to testset.
            id2label (function(string[])->string[], Optional): Needed if label_data is None to convert labels from onehot index to actual label. Unneeded if calc_embeddings=False. Default to None
            
        """
        if self.analytics is None:
            raise Exception("Analytics is None. Please call load_analytics() beforehand")

        if reset_umap:
            self.analytics.reset_umap()

        if data_loader is None:
            data_loader = self.test_loader
                
        logging.info(f"[plot_umap] calc_embeddings={calc_embeddings}")

        if embedding_data is None and label_data is None:
            if not calc_embeddings:
                if embeddings_type is None:
                    embeddings_type='testset' if data_loader.dataset.conf.SPLIT_DATA else 'all'
                    logging.warn("embeddings_type is None. Setting to 'testset' if SPLIT_DATA=True, 'all' otherwise")

                embedding_data, label_data = self.load_embeddings(embeddings_type)
                
                if len(embedding_data) == 0:
                    logging.info("Couldn't find embeddings to load. Calculating them instead. (without saving)")
                    embedding_data, label_data = None, None
                    
            else:
                logging.warn("embedding_data & labe_data aren't None, but calc_embeddings==True, then calculating new embeddings")
                embedding_data, label_data = None, None
            
        
        umap_data = self.analytics.plot_umap_of_embedding_vector(
            data_loader=data_loader,
            id2label=id2label,
            group_col=0,
            title=title,
            xlabel='UMAP1',
            ylabel='UMAP2',
            s=s,
            alpha=alpha,
            show_legend=True,
            is_3d=is_3d,
            random_state=self.conf.SEED,
            embedding_data=embedding_data,
            label_data=label_data,
            **kwargs)
        
        return umap_data
    