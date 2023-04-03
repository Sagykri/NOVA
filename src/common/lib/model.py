import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from cytoself.models import CytoselfFullModel
from cytoself.data_loader.data_manager import DataManager
from cytoself.analysis.analytics import Analytics
from tensorflow.compat.v1.keras.callbacks import CSVLogger
import logging

from common.configs.model_config import ModelConfig


class Model():

    def __init__(self, conf:ModelConfig):
        self.set_params(conf())
        
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
        
        
        self.model = None
        self.analytics = None
        
        
    def set_params(self, conf:ModelConfig):
        """Set the parameters

        Args:
            conf (ModelConfig): The new values
        """
        try:
            self.input_folders = conf.INPUT_FOLDERS
            self.add_condition_to_label = conf.ADD_CONDITION_TO_LABEL
            self.add_line_to_label = conf.ADD_LINE_TO_LABEL
            self.add_type_to_label = conf.ADD_TYPE_TO_LABEL
            self.add_batch_to_label = conf.ADD_BATCH_TO_LABEL
            self.markers = conf.MARKERS    
            self.markers_to_exclude = conf.MARKERS_TO_EXCLUDE
            self.markers_for_downsample = conf.MARKERS_FOR_DOWNSAMPLE   
            self.cell_lines_include= conf.TRAIN_CELL_LINES
            self.split_by_set = conf.SPLIT_DATA
            self.data_set_type = conf.DATA_SET_TYPE
            self.train_part = conf.TRAIN_PCT
            self.shuffle = conf.SHUFFLE
            self.cell_lines = conf.CELL_LINES 
            self.conditions = conf.CONDITIONS
            self.split_by_set_for = conf.SPLIT_BY_SET_FOR
            self.split_by_set_for_batch = conf.SPLIT_BY_SET_FOR_BATCH
            
            self.pretrained_model_path = conf.PRETRAINED_MODEL_PATH
            
            self.early_stop_patience = conf.EARLY_STOP_PATIENCE
            self.learn_rate = conf.LEARN_RATE
            self.batch_size = conf.BATCH_SIZE
            self.max_epoch = conf.MAX_EPOCH
            
            self.conf = conf
        except Exception as ex:
            logging.error(f"Error with the configuration file. {ex}")
    
    def load_data(self):
        """ Load images from given folders """
        
        input_folders           =   self.input_folders
        condition_l             =   self.add_condition_to_label
        line_l                  =   self.add_line_to_label
        batch_l                 =   self.add_batch_to_label
        cell_type_l             =   self.add_type_to_label
        markers                 =   self.markers
        markers_to_exclude      =   self.markers_to_exclude
        markers_for_downsample  =   self.markers_for_downsample
        cell_lines_include      =   self.cell_lines_include
        conds_include           =   self.conditions
        set_type                =   self.data_set_type
        split_by_set            =   self.split_by_set
        split_by_set_for        =   self.split_by_set_for
        split_by_set_for_batch  =   self.split_by_set_for_batch
        train_part              =   self.train_part
        shuffle                 =   self.shuffle
        

        labels_changepoints = [0]
        labels = []
        images_concat = None
        markers_order = []

        if split_by_set:
            logging.info("#########################################################")
            logging.info(f"########### Splitting by set! ({set_type}) #############")
            logging.info("#########################################################")
            np.random.seed(self.conf.SEED)

        for i, input_folder in enumerate(input_folders):
            logging.info(f"Input folder: {input_folder}")
            if cell_type_l:
                if "microglia" in input_folder:
                    cur_cell_type = "microglia"
                else:
                    cur_cell_type = "neurons"
            for cell_line_folder in sorted(os.listdir(input_folder)):
                if cell_lines_include is not None and cell_line_folder not in cell_lines_include:
                    continue
                cell_line_folder_fullpath = os.path.join(input_folder, cell_line_folder)
                for j, cond_folder in enumerate(sorted(os.listdir(cell_line_folder_fullpath))):
                    #labels_counts.append(0)
                    
                    if conds_include is not None and cond_folder not in conds_include:
                        continue
                    cond_folder_fullpath = os.path.join(cell_line_folder_fullpath, cond_folder)
                    
                    
                    # if advanced_selection is not None:
                    #     if not isinstance(advanced_selection, list):
                    #         advanced_selection = [advanced_selection]
                    #     if tuple((cell_line_folder, cond_folder)) not in advanced_selection:
                    #         if verbose:
                    #             logging.info(f"Skipping (advanced selection): {cell_line_folder}/{cond_folder}")
                    #             continue
                    
                    for subfolder_name in sorted(os.listdir(cond_folder_fullpath)):
                        logging.info(f"Input subfolder: {subfolder_name}")

                        if markers is not None and subfolder_name not in markers:
                            logging.info(f"Skipping. {subfolder_name}")
                            continue

                        if markers_to_exclude is not None and subfolder_name in markers_to_exclude:
                            logging.info(f"Skipping (in markers to exclude). {subfolder_name}")
                            continue

                        if subfolder_name not in markers_order:
                            markers_order.append(subfolder_name)
                        subfolder = os.path.join(cond_folder_fullpath, subfolder_name)

                        if not os.path.isdir(subfolder) or ".ipynb_checkpoints" in subfolder:
                            continue

                        for filename in sorted(os.listdir(subfolder)):
                            file_path = os.path.join(subfolder, filename)

                            if os.path.isdir(file_path) or ".ipynb_checkpoints" in file_path or filename == 'desktop.ini':
                                continue

                            f_no_ext = os.path.splitext(filename)[0]
                            tpe = f_no_ext[f_no_ext.rindex('_') + 1:]

                            if cell_lines_include is not None and tpe not in cell_lines_include:
                                continue

                            logging.info(f"Filepath: {file_path}")

                            data = np.load(file_path)

                            split_by_set_include_current = True

                            # Cell line, condition
                            if split_by_set_for is not None:
                                if not isinstance(split_by_set_for, list):
                                    split_by_set_for = [split_by_set_for]
                                if tuple((cell_line_folder, cond_folder)) not in split_by_set_for:
                                    split_by_set_include_current = False

                            # Batch
                            if split_by_set_for_batch is not None:
                                if not isinstance(split_by_set_for_batch, list):
                                    split_by_set_for_batch = [split_by_set_for_batch]
                                if input_folder not in split_by_set_for_batch:
                                    split_by_set_include_current = False

                            # Downsample all data
                            if markers_for_downsample:  # TODO: ask Sagy!
                                if subfolder_name in markers_for_downsample.keys():
                                    to_sample = int(data.shape[0] * markers_for_downsample[subfolder_name])
                                    rng = np.random.default_rng()
                                    data = rng.choice(data, size=to_sample, replace=False)

                            # Split data by set (train/val/test)
                            if split_by_set and split_by_set_include_current:

                                np.random.shuffle(data)
                                train_size = int(len(data) * train_part)
                                val_size = int((len(data) - train_size) * train_part)

                                if set_type == 'train':
                                    data = data[:train_size]
                                elif set_type == 'val':
                                    data = data[train_size: train_size + val_size]
                                elif set_type == 'test':
                                    data = data[train_size + val_size:]
                                else:
                                    raise "ERROR: Bad set_type"

                            if images_concat is None:
                                images_concat = data
                            else:
                                images_concat = np.vstack((images_concat, data))

                            # Save when there is change between markers/conditions
                            labels_changepoints.append(len(images_concat))
                            
                            cond = cond_folder
                            
                            lbl = subfolder_name
                            if line_l:
                                lbl += f"_{tpe}"
                            if condition_l:
                                lbl += f"_{cond}"
                            if cell_type_l:
                                lbl += f"_{cur_cell_type}"
                            
                            labels += [lbl] * len(data)

                            if batch_l:
                                batch_postfix = f"_{os.path.basename(input_folder)}"
                                batch_size = len(data)

                                labels[-batch_size:] = [l + batch_postfix for l in labels[-batch_size:]]

        labels = np.asarray(labels).reshape(-1, 1)

        logging.info(f"{images_concat.shape}, {labels.shape}")

        if shuffle:
            p = np.random.permutation(len(images_concat))
            return images_concat[p], labels[p], None, None



        
        if not split_by_set or (split_by_set and set_type == 'test'):
            self.test_data, self.test_label,self.test_labels_changepoints, self.test_markers_order = images_concat, labels, labels_changepoints, markers_order
        else:
            if set_type == 'train':
                self.train_data, self.train_label,self.train_labels_changepoints, self.train_markers_order = images_concat, labels, labels_changepoints, markers_order
            elif set_type == 'val':
                self.val_data, self.val_label,self.val_labels_changepoints, self.val_markers_order = images_concat, labels, labels_changepoints, markers_order

        return images_concat, labels, labels_changepoints, markers_order


    def train(self, continue_training=False):
        """ 
        Train a model on given data
        
        Args:
            continue_training (bool): Whether to continue training or start training from the pretrained model
            
        """
        
        if continue_training:
            if self.model is None:
                raise Exception("Cannot continue training the model. Model is undefined.")
            else:
                model_to_train = self.model
        
        logging.info("Loading params from configuration")
        
        train_data, train_label = self.train_data, self.train_label
        val_data, val_label     = self.val_data, self.val_label
        test_data, test_label   = self.test_data, self.test_label
        pretrained_model_path   = self.pretrained_model_path
        early_stop_patience     = self.early_stop_patience
        learn_rate              = self.learn_rate 
        batch_size              = self.batch_size
        max_epoch               = self.max_epoch
        model_output_folder     = self.conf.MODEL_OUTPUT_FOLDER
        
        
        logging.info("Creating the model")
        logging.info(
            f"early_stop_patience={early_stop_patience}, learn_rate={learn_rate}, batch_size={batch_size}, max_epoch={max_epoch}")

        if not continue_training:
            model_to_train = CytoselfFullModel(input_image_shape=train_data.shape[1:],
                                    num_fc_output_classes=len(np.unique(train_label)),
                                    early_stop_patience=early_stop_patience, learn_rate=learn_rate,
                                    output_dir=model_output_folder)

        data_manager = DataManager(
            train_data=train_data,
            train_label=train_label,
            val_data=val_data,
            val_label=val_label,
            test_data=test_data,
            test_label=test_label
        )

        # Load pre-trained model

        logging.info("Loading a pretrained model's weights...")
        pretrained_model = CytoselfFullModel(input_image_shape=train_data.shape[1:])
        pretrained_model.load_model(pretrained_model_path)

        if not continue_training:
            # Copying weights (except the last two because of a different num_fc_output_classes)
            for i in range(len(pretrained_model.model.layers) - 2):
                model_to_train.model.layers[i].set_weights(pretrained_model.model.layers[i].get_weights())

        logging.info("Compiling the model...")
        # Compile the model with data_manager
        model_to_train.compile_with_datamanager(data_manager)

        logging.info("Training the model...")

        model_to_train.init_callbacks()
        csv_logger = CSVLogger(f"./logs/training_log_hist.csv", append=True, separator=',')

        model_to_train.callbacks += [csv_logger]
        model_to_train.train_with_datamanager(data_manager, batch_size=batch_size, max_epoch=max_epoch, reset_callbacks=False)

        logging.info("Finished training the model...")

        self.model = model_to_train
        
        return self.model

    def load_model(self, num_fc_output_classes=None, input_image_shape=None):
        
        """Load model

        Args:
            num_fc_output_classes (bool, Optional): Number ouf outputs for the fully connected model (number of classes) (Default is number of unique values in labels)
            input_image_shape (bool, Optional): Size of input image (Default is based on test_data)

        Raises:
            ValueError: No input image shape
            ValueError: No num_fc_output_classes

        Returns:
            _type_: The model
        """
        
        model_path = self.conf.MODEL_PATH
        
        if input_image_shape is None:
            if self.test_data is None:
                raise ValueError("No input image shape")
            else:
                input_image_shape = self.test_data.shape[1:]
        
        if num_fc_output_classes is None:
            if self.test_label is None:
                raise ValueError("No num_fc_output_classes")
            else:
                num_fc_output_classes = len(np.unique(self.test_label))
                
        model = CytoselfFullModel(input_image_shape=input_image_shape,
                                  num_fc_output_classes=num_fc_output_classes,
                                  output_dir=self.conf.MODEL_OUTPUT_FOLDER)
        
        logging.info("Loading weights")

        model.load_model(model_path)
        
        self.model = model
        
        return model
        

    def load_analytics(self):
        """Load Analytics - an API object to cytoself

        Raises:
            ValueError: Model not loaded

        Returns:
            Analytics: The analytics object
        """

        if self.model is None:
            raise ValueError("Model not loaded")

        test_data   = self.test_data
        test_label  = self.test_label


        logging.info("X, y:")
        test_label_pd = pd.Series(test_label.reshape(-1, ))
        logging.info(f"{test_data.shape}, {test_label_pd.shape}")
        logging.info(test_label_pd.value_counts())

        
        data_manager = DataManager(
            test_data=test_data,
            test_label=test_label
        )

        # Generate ground truth from labels
        lbl = np.unique(test_label)
        gt_table = pd.DataFrame([[l, l] for l in lbl], columns=["gene_name", "localization"])

        logging.info("Ground truth:")
        logging.info(gt_table['gene_name'].values)

        analytics = Analytics(self.model, data_manager, gt_table=gt_table)
        
        self.analytics = analytics
        
        return analytics


    def generate_reconstructed_images(self, images_indexes=None, embvecs=None,\
                        reset_embvec=True, only_second_layer=False, show=True):
        """Generate reconstructed images for a given input images via the autoencoder

        Args:
            images_indexes ([int], optional): Indexes of images to use as input from the test_data. Defaults to all indexes.
            embvecs (_type_, optional): Use an existing embedded vectors. Defaults to calculating them.
            only_second_layer (bool, optional): Use only the second vq layer. Defaults to False.
            show (bool, optional): Show reconstructed images. Defaults to True.

        Raises:
            ValueError: Analytic not loaded

        Returns:
            np array: reconstructed images
        """
        
        if self.analytics is None:
            raise ValueError("Analytics not loaded")
        
        analytics = self.analytics
        
        vq_layer = 1 if only_second_layer else 0 
        
        # Create decoder model
        dec_model = analytics.model.construct_decoder_model(dec_idx=vq_layer+1)

        # Load embedding vectors
        if embvecs is None:
            if analytics.model.embvec is None or reset_embvec:
                analytics.model.calc_embvec(analytics.data_manager.test_data)
                embvec0 = analytics.model.embvec[0].copy()
                embvec1 = analytics.model.embvec[1].copy()
        else:
            embvec0 = embvecs[0].copy()
            embvec1 = embvecs[1].copy()
            
            
        # Load data
        if images_indexes is None:
            # Take all
            images_indexes = np.arange(len(analytics.data_manager.test_data))
            
        train_raw = analytics.data_manager.test_data[images_indexes]
        train_labels = analytics.data_manager.test_label[images_indexes].flatten()
        num_img = len(images_indexes)
        train_raw = train_raw[:num_img]
        
        embvec0 = embvec0[images_indexes]
        embvec1 = embvec1[images_indexes]
        logging.info(f"embvec0 shape: {embvec0.shape}, embvec1 shape: {embvec1.shape}")
        
        if vq_layer == 1:
            embvecs = embvecs[0]
        embvecs = [embvec1.reshape(-1, 4, 4, 576), embvec0]
        
        # Generate images
        train_gen = dec_model.predict(embvecs)
        train_gen = train_gen[:num_img]

        if not show:
            return train_gen

        ################ Plot generated images ################
        
        fig, ax = plt.subplots(num_img,5, figsize=(8,8), dpi=1000)
        fig.patch.set_facecolor('white')
        
        # Set titles
        titles = ['Target_Org', 'Nucleus_Org', 'Target', 'Nucleus']
        current_ax = ax[0] if num_img > 1 else ax
        for i, t in enumerate(titles):
            current_ax[i].set_title(t, fontsize=3)
            
        # Show images
        cax = fig.add_axes([0, 0.5, 0.01, 0.1])
        for i in range(num_img):
            current_ax = ax[i] if num_img > 1 else ax
            
            ##### ORIGINAL IMAGES ######
            
            # Target channel - original image
            img = train_raw[i,...,0]
            # Increase contrast with min-max scaling
            X_std = (img - img.min(axis=0)) / (img.max(axis=0) - img.min(axis=0))
            img = X_std * (1 - 0) + 0
            im = current_ax[0].imshow(img, cmap = 'rainbow', interpolation = 'sinc')
            current_ax[0].set_axis_off()

            # Nucleus channel - original image
            img = train_raw[i,...,1]
            # Increase contrast with min-max scaling
            X_std = (img - img.min(axis=0)) / (img.max(axis=0) - img.min(axis=0))
            img = X_std * (1 - 0) + 0
            im = current_ax[1].imshow(img, cmap = 'rainbow', interpolation = 'sinc')
            current_ax[1].set_axis_off()
            
            ###############################
            
            ##### GENERATED IMAGES ######
            
            # Target
            img = train_gen[i,...,0]
            X_std = (img - img.min(axis=0)) / (img.max(axis=0) - img.min(axis=0))
            img = X_std * (1 - 0) + 0
            im = current_ax[2].imshow(img, cmap = 'rainbow', interpolation = 'sinc')
            current_ax[2].set_axis_off()
            
            # Nucleus
            img = train_gen[i,...,1]
            X_std = (img - img.min(axis=0)) / (img.max(axis=0) - img.min(axis=0))
            img = X_std * (1 - 0) + 0
            im = current_ax[3].imshow(img, cmap = 'rainbow', interpolation = 'sinc')
            current_ax[3].set_axis_off()
            
            ###############################
            
            # Label
            current_ax[4].set_axis_off()
            current_ax[4].text(0,0.5,train_labels[i], fontsize='xx-small')
        
        # Add colorbar
        fig.colorbar(im, cax=cax, orientation='vertical')
        
        # Reduce the space between axes
        plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.3, 
                        top=0.9)
        plt.show()
        
        return train_gen
        
        
    # TODO: Move here the feature specturm generation
    def generate_feature_spectrum(self):
        raise NotImplementedError()
        
    # TODO: load embedding vectors
    def load_embedding_vectors(self):
        raise NotImplementedError()
    
    