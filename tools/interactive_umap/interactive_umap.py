import os
import shutil
import time
import ctypes
import gc
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import ipywidgets as widgets
from ipywidgets import Button, Output, Text, Layout
from IPython.display import display, clear_output

from src.figures.umap_plotting import __format_UMAP_axes as format_UMAP_axes
from src.figures.umap_plotting import __format_UMAP_legend as format_UMAP_legend
from tools.interactive_umap.interactive_umap_utils import *
from tools.show_images_utils import show_processed_tif, extract_image_metadata


class InteractiveUMAPPipeline:
    def __init__(self,  config={}, hover=False):
        """
        Initializes the InteractiveUMAPPipeline.

        Args:
            config (dict): Configuration dictionary containing:
                - 'paths': dict with keys:
                    - 'umaps_folder' (str): Path to the UMAPs folder.
                    - 'csv_path' (str): Path to the Brenner CSV file.
                    - 'images_dir' (list of str): List of directories containing raw images.
                - 'layouts': dict containing predefined FOV layouts for heatmap visualization.
            hover (bool): Whether to enable interactive hover annotations on UMAP plots (can slow performance).
        """
        # --- Memory management ---
        self.dispose()
        # --- General setup ---
        self.filter_checkboxes = {}  # Stores dynamically generated filter checkboxes
        self.advanced_filter_dropdowns = {}  # Stores advanced filter dropdowns, e.g. 'Cell_Line_Condition', 'Panel', 'Marker'
        self.selected_indices_global = []  # Stores the selected point indices from rectangle selection
        self.rect_selector = None  # Persistent RectangleSelector object for UMAP plot
        
        # --- Data containers (start empty, filled during pipeline) ---
        self.df_site_meta = None  # Metadata for all raw image files (sites) from the directory
        self.df_brenner = None  # Brenner scores table for sites (sharpness/quality scores)
        self.df_umap_meta = None  # Metadata describing available UMAPs (e.g., type, batch, coloring) in the UMAP directory
        self.df_umap_tiles = None  # Metadata of the tiles in the displayed UMAP
        self.umap_embeddings = None  # 2D UMAP embeddings for the plot
        self.label_data = None  # Labels associated with each UMAP point (e.g., cell type, condition)
        self.config_data = None  # Raw configuration data saved with the UMAP
        self.config_plot = None  # Plotting configuration loaded with the UMAP (size, alpha, color maps)
        self.df_umap_tiles_filt = None  # Filtered version after applying checkbox filters
        self.umap_embeddings_filt = None  # Filtered version of embeddings
        self.label_data_filt = None  # Filtered version of labels
        self.current_umap_figure = None  # Currently displayed UMAP figure

        # --- Settings ---
        self.hover = hover  # Whether to enable interactive hover-over annotations on points

        # --- UI Setup ---
        paths = config.get('paths', {})
        layouts = config.get('layouts', {})
        self.exp_name = config.get('name', '')  # Experiment name for saving
        self._create_widgets(paths)  # Create all the ipywidgets (sliders, text fields, buttons)
        self._setup_callbacks()  # Connect buttons and actions to their event handlers
        self._display_ui()  # Display the complete layout of the app

        # --- Extra Config ---
        self.fov_layouts = layouts  # Predefined FOV maps for heatmap plotting
        self.format_UMAP_legend = format_UMAP_legend  # External helper function for formatting legend
        self.format_UMAP_axes = format_UMAP_axes  # External helper function for formatting axes
        check_memory_status()  # Initial memory check

    def _create_widgets(self, default_paths):
        # --- Path configuration widgets ---
        self.umaps_dir_widget = self.make_text_widget(
            default_paths.get('umaps_folder', ''), 'UMAPs Dir:',
            tooltip='Path to the folder containing UMAPs'
        )
        self.csv_path_widget = self.make_text_widget(
            default_paths.get('csv_path', ''), 'CSV Brenner Path:',
            tooltip='Path to a CSV file with image sharpness scores (Brenner)'
        )
        images_dirs = default_paths.get('images_dir', '')
        if isinstance(images_dirs, list):
            images_dirs = ', '.join(images_dirs)

        self.images_dir_widget = self.make_text_widget(
            images_dirs, 'Raw Images Dirs:',
            tooltip='Comma-separated list of directories containing raw images'
        )

        # --- Output display areas ---
        self.output_area = Output(layout={'height': '50px', 'margin': '0px 0px'})
        self.umap_output = Output(layout={'height': 'auto', 'margin': '10px 0px'})
        self.selected_images_output, self.selected_images_output_inner = self.create_scrollable_output(height='400px')
        self.selected_tiles_output, self.selected_tiles_output_inner = self.create_scrollable_output(height='350px')
        self.fov_output = Output(layout={'height': '1000px', 'margin': '0 auto', 'display': 'block'})
        self.warnings_output = Output()

        # --- Control buttons ---
        self.run_button = Button(description="Run", layout=Layout(width='200px', margin='5px 250px', ), tooltip="Search for UMAPs and load available options",)
        self.create_umap_button = Button(description="Create UMAP", layout=Layout(width='200px', margin='5px 10px', display = 'none'),
                                         tooltip="Create UMAP with selected parameters")
        self.show_images_button = Button(description="Show Selected Points", layout=Layout(width='200px', margin='0px 10px'), tooltip="Show sites and tiles images corresponding to selected points")
        self.apply_filter_button = Button(description="Apply Filters", layout=Layout(width='180px', margin='10px 0px 10px 5px', display = 'none'))
        self.save_button = Button(description="💾 Save", layout=Layout(width='150px', margin='0px 10px'), tooltip="Save displayed umap and images to folder saved_umaps")
        
        self.save_status_label = widgets.HTML(value="", layout=Layout(margin="0 0 0 10px"))

        self.num_images_slider = widgets.IntSlider(
            value=10, min=1, max=30, step=1,
            description='Num images:',
            layout=Layout(width='250px'),
            style={'description_width': '100px'}
        )
        self.image_display_controls = widgets.HBox([
            self.num_images_slider,
            self.show_images_button,
            self.save_button,
        ], layout=Layout(display='none'))

        # --- UMAP dropdowns ---
        dropdown_style = {'description_width': '150px'}
        dropdown_width = '400px'

        # Create dropdowns with consistent layout and style
        self.umap_type_dropdown = widgets.Dropdown(description='1. UMAP Type:', layout=Layout(width=dropdown_width), style=dropdown_style)
        self.batch_dropdown = widgets.Dropdown(description='2. Batch:', layout=Layout(width=dropdown_width), style=dropdown_style)
        self.reps_dropdown = widgets.Dropdown(description='Reps:', layout=Layout(width=dropdown_width), style=dropdown_style)
        self.coloring_dropdown = widgets.Dropdown(description='Coloring:', layout=Layout(width=dropdown_width), style=dropdown_style)
        self.marker_dropdown = widgets.Dropdown(description='Marker:', layout=Layout(width=dropdown_width), style=dropdown_style)
        self.cell_line_dropdown = widgets.Dropdown(description='Cell Line:', layout=Layout(width=dropdown_width), style=dropdown_style)
        self.condition_dropdown = widgets.Dropdown(description='Condition:', layout=Layout(width=dropdown_width), style=dropdown_style)

        # Arrange in two columns
        dropdown_col1 = widgets.VBox([
            self.umap_type_dropdown,
            self.batch_dropdown,
            
        ])
        dropdown_col2 = widgets.VBox([
            self.marker_dropdown,
            self.cell_line_dropdown,
            self.coloring_dropdown,
            self.condition_dropdown,
            self.reps_dropdown,
        ])

        def labeled_checkbox(description, tooltip, value=False):
            checkbox = widgets.Checkbox(value=value, layout=Layout(width='auto'))
            checkbox.layout.flex = '0 0 auto'  # don’t shrink

            label = widgets.HTML(
                f"<span title='{tooltip}' style='font-size:13px;'>{description}</span>",
                layout=Layout(margin='0 0 0 8px')
            )
            return widgets.HBox([checkbox, label], layout=Layout(align_items='center', display='flex')), checkbox

        # --- Additional plot controls ---
        self.pickle_status_label = widgets.HTML()
        self.mix_groups_box, self.mix_groups_checkbox = labeled_checkbox('Mix Groups', 'Shuffle and plot instead of plotting each group on top of each other', value=True)
        self.recolor_box, self.recolor_checkbox = labeled_checkbox('Recolor by Brenner','Recolor points based on Brenner scores (if available)', value=False)
        self.dilute_slider = widgets.SelectionSlider(
            options=[1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100],
            value=1,
            description='Downsample',
            style={'description_width': '100px'},
            layout=Layout(width='300px'),
            continuous_update=False,
            tooltip='Dilute the number of points plotted (1 = no dilution, 2 = every second point, etc.)'
        )
        self.bins_slider = widgets.IntSlider(value=5, min=1, max=20, step=1, description='Colors (Brenner)', style={'description_width': '100px'},
                                             tooltip='Number of colors for Brenner score recoloring')

        # --- Layout containers ---
        self.umap_params = widgets.VBox([
            widgets.HTML("<b>Select UMAP Parameters (not all combinations are available):</b>"),
            widgets.HBox([dropdown_col1, dropdown_col2], layout=Layout(gap='20px')),
            self.mix_groups_box,
            self.recolor_box,
            self.bins_slider,
            self.dilute_slider,
            self.pickle_status_label,
            self.create_umap_button
        ], layout=Layout(display='none', gap='40px'))

        self.right_box = widgets.VBox(layout=Layout(display='none', width='320px', margin='0 0 0 10px'))  # Populated later with filters

        # --- Filter checkboxes (populated dynamically) ---
        self.filter_checkboxes = {}

        # --- Section labels ---
        self.selected_images_label = widgets.HTML(value="<b>1. Selected Images:</b>", layout=widgets.Layout(display='none'))
        self.selected_tiles_label = widgets.HTML(value="<b>2. Corresponding Tiles:</b>", layout=widgets.Layout(display='none'))
        self.fov_label = widgets.HTML(value="<b>3. FOV Map:</b>", layout=widgets.Layout(display='none'))
        
    def create_scrollable_output(self, height='300px'):
        inner = Output()
        outer = widgets.Box(
            [inner],
            layout=widgets.Layout(
                height=height,
                overflow='auto',
                border='1px solid #ccc',
                padding='5px',
                flex='1 1 auto'
            )
        )
        inner.layout.width = '100%'
        return outer, inner

    def _setup_callbacks(self):
        self.run_button.on_click(self.run_pipeline)
        self.show_images_button.on_click(self.show_selected_images)
        self.create_umap_button.on_click(self.create_umap)
        self.apply_filter_button.on_click(self.apply_filters_and_update_plot)
        self.save_button.on_click(self.save_all)

    def _display_ui(self):
        self.ui = widgets.VBox([
            self.umaps_dir_widget,
            self.csv_path_widget,
            self.images_dir_widget,
            self.run_button,
            widgets.HTML("<hr>"),
            self.output_area,
            self.umap_params,
            widgets.HTML("<hr>"),
            widgets.HBox([self.umap_output, self.right_box]),
            self.image_display_controls,
            self.save_status_label,
            widgets.HTML("<hr>"),
            self.selected_images_label,
            self.selected_images_output,
            widgets.HTML("<hr>"),
            self.selected_tiles_label,
            self.selected_tiles_output,
            widgets.HTML("<hr>"),
            self.fov_label,
            self.fov_output,
            self.warnings_output,
        ], layout=widgets.Layout(padding='10px', min_height='2500px', height='auto', overflow_y='visible'))

    def show(self):
        display(self.ui)

    def reset(self, reset_metadata=False):
        """Reset the pipeline state. If reset_metadata=True, also clear UMAP metadata, image folders, and Brenner info."""

        # 1. Close all figures
        plt.close('all')

        # 2. Clear output widgets
        for out in (
            self.output_area,
            self.umap_output,
            self.selected_images_output_inner,
            self.selected_tiles_output_inner,
            self.fov_output,
            self.warnings_output
        ):
            out.clear_output(wait=True)

        # 3. Disconnect interactive elements
        if getattr(self, 'rect_selector', None):
            self.rect_selector.set_active(False)
            try: self.rect_selector.disconnect_events()
            except: pass
            self.rect_selector = None

        if getattr(self, 'hover_cursor', None):
            try: self.hover_cursor.disconnect()
            except: pass
            del self.hover_cursor

        # 4. Clear special filter dropdowns (if exist)
        for attr in ('combination_dropdown', 'panel_dropdown'):
            if hasattr(self, attr):
                delattr(self, attr)

        # 5. Clear current UMAP data
        for attr in (
            'umap_embeddings', 'label_data', 'df_umap_tiles',
            'config_data', 'config_plot',
            'umap_embeddings_filt', 'label_data_filt', 'df_umap_tiles_filt'
        ):
            setattr(self, attr, None)

        # 6. Clear saving status
        self.save_status_label.value = ''

        # 7. Optionally clear metadata and Brenner info
        if reset_metadata:
            for attr in ('df_umap_meta', 'df_site_meta', 'df_brenner'):
                setattr(self, attr, None)

        # 8. Clear dynamic filters
        self.filter_checkboxes.clear()
        self.right_box.children = ()
        self.applied_filters = {} 

        # 9. Force garbage collection
        gc.collect()

    def run_pipeline(self, btn):
        self.clear_outputs()
        self.reset(reset_metadata = True)
        self.umap_params.layout.display = 'none'
        self.pickle_status_label.value = ''
        with self.output_area:
            clear_output()
            print('Searching for all UMAPs in folder... (up to 30 seconds)')

            umaps_dir = self.umaps_dir_widget.text_input.value.strip()
            csv_path = self.csv_path_widget.text_input.value.strip()
            image_dirs_raw = self.images_dir_widget.text_input.value.strip()

            # --- UMAPs dir must exist ---
            if not umaps_dir or not os.path.isdir(umaps_dir):
                print("❌ UMAP folder not defined or does not exist.")
                return

            # --- Brenner CSV: optional, but if defined must exist ---
            if csv_path:
                if not os.path.isfile(csv_path):
                    print(f"❌ Brenner CSV file not found at:\n{csv_path}")
                    return
                else:
                    self.df_brenner = pd.read_csv(csv_path)
                    self.df_brenner["Image_Name"] = self.df_brenner["Path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
                    ## Find the panel (necessary for DAPI)  
                    self.df_brenner["Panel"] = self.df_brenner["Path"].str.extract(r"/(panel\w+)/")  # extract panelD etc.
                    self.df_brenner["Panel"] = self.df_brenner["Panel"].str.replace("panel", "", case=False)  # get just the letter, e.g., D
            else:
                print("ℹ️ No Brenner CSV provided. Continuing without Brenner scores.")
                self.df_brenner = None

            # --- Images dir: optional, but if defined must exist ---
            if image_dirs_raw:
                image_dirs = [d.strip() for d in image_dirs_raw.split(",") if d.strip()]
                valid_dirs = [d for d in image_dirs if os.path.isdir(d)]
                if not valid_dirs:
                    print("❌ No valid image directories found.")
                    return

                # Concatenate metadata from all valid dirs
                metadata_list = []
                for d in valid_dirs:
                    meta = extract_image_metadata(d, FILE_EXTENSION='.tiff', KEY_BATCH='Batch')
                    meta["__source_dir__"] = d
                    metadata_list.append(meta)

                self.df_site_meta = pd.concat(metadata_list, ignore_index=True)
            else:
                print("ℹ️ No image directories provided. Continuing without image metadata.")
                self.df_site_meta = None

            self.df_umap_meta = extract_umap_data(base_dir=umaps_dir)
            clear_output()
            print(len(self.df_umap_meta), 'UMAPs were located')

            self.populate_dropdowns_from_umaps()
            self.umap_params.layout.display = 'flex'
            self.create_umap_button.layout.display = 'inline-block'
            check_memory_status()
         
    def create_umap(self, btn): 
        self.reset()
        with self.umap_output:
            clear_output()
        with self.umap_output:
            pickle_file_path = get_umap_pickle_path(
                self.df_umap_meta,
                batch=self.batch_dropdown.value,
                umap_type=self.umap_type_dropdown.value,
                reps=self.reps_dropdown.value,
                coloring=self.coloring_dropdown.value,
                marker=self.marker_dropdown.value,
                cell_line=self.cell_line_dropdown.value,
                condition=self.condition_dropdown.value
            )

            # Check existence
            if pickle_file_path != -1:
                self.pickle_status_label.value = f"<span style='color: green;'>✅ Pickle file found:</span><br><code>{pickle_file_path}</code>"
            else:
                self.pickle_status_label.value = f"<span style='color: red;'>❌ Pickle file not found:</span><br><code></code>"
                return  # Stop execution if file doesn't exist

            self.umap_embeddings, self.label_data, self.config_data, self.config_plot, self.df_umap_tiles = load_and_process_data(
                self.umaps_dir_widget.text_input.value, pickle_file_path, self.df_brenner)
            
            self.apply_dilution() 
     
            self.config_plot['MIX_GROUPS'] = self.mix_groups_checkbox.value

            self.plot_interactive_umap()

            # Dynamically populate checkbox filters
            self.update_filter_widgets()

        self.apply_filter_button.layout.display = 'inline-block'
        self.right_box.layout.display = 'flex'
        self.image_display_controls.layout.display = 'flex'
        self.clear_outputs(umaps=False)
        check_memory_status()

    def apply_dilution(self):
        """Apply downsampling to embeddings, labels, and tiles."""
        dilute = self.dilute_slider.value
        self.umap_embeddings = self.umap_embeddings[::dilute]
        self.label_data = self.label_data[::dilute]
        self.df_umap_tiles = self.df_umap_tiles.iloc[::dilute]
        self.df_umap_tiles.index = list(range(len(self.df_umap_tiles)))

    def update_filter_widgets(self):
        """Update and display dynamic filter checkboxes and dropdowns."""
        self.right_box.children = [
            widgets.HTML("<b>Filter Settings:</b>")
        ] + [
            self.create_checkbox_group(col) for col in ['Batch', 'Condition', 'Rep', 'CellLine']
        ] + [
            self.create_more_filters()
        ] + [self.apply_filter_button]

    def create_more_filters(self):
        def make_dropdown(series, label_text, key):
            counts = series.dropna().value_counts()
            options = [f"{val} ({counts[val]})" for val in sorted(counts.index)]
            height = f"{max(min(30 * len(options), 180), 50)}px"
            dropdown = widgets.SelectMultiple(
                options=options,
                layout=Layout(width='95%', height=height)
            )
            self.advanced_filter_dropdowns[key] = dropdown
            return widgets.Label(label_text), dropdown

        label1, _ = make_dropdown(self.df_umap_tiles['Cell_Line_Condition'], "CellLine + Condition", "Cell_Line_Condition")
        label2, _ = make_dropdown(self.df_umap_tiles['Panel'], "Panel", "Panel")
        label3, _ = make_dropdown(self.df_umap_tiles['Marker'], "Marker", "Marker")

        inner = widgets.VBox([
            widgets.HTML("<span style='font-size:11px;'>Hold Ctrl to select/deselect multiple</span>"),
            label1, self.advanced_filter_dropdowns["Cell_Line_Condition"],
            label2, self.advanced_filter_dropdowns["Panel"],
            label3, self.advanced_filter_dropdowns["Marker"]
        ])

        acc = widgets.Accordion(children=[inner])
        acc.set_title(0, "More Filters (Advanced)")
        acc.selected_index = None
        return acc
        
    def apply_filters_and_update_plot(self, btn): 
        with self.umap_output:
            clear_output()
    
            # Build filters dictionary
            filters = self.build_filters_from_checkboxes()
            self.applied_filters = filters 
            self.filter_umap_data(filters=filters)

            self.plot_interactive_umap()
            self.clear_outputs(umaps=False)

    def filter_umap_data(self, filters: dict):
        """
        Filters umap_embeddings, label_data, and df_umap_tiles based on values in filters.

        Args:
            umap_embeddings (np.ndarray): 2D array of shape (N, 2) containing UMAP embeddings.
            label_data (np.ndarray): 1D array of shape (N,) containing labels for each embedding.
            df_umap_tiles (pd.DataFrame): DataFrame containing image statistics.
            filters (dict): Dictionary where keys are column names in df_umap_tiles and values are lists of allowed values.

        Returns:
            np.ndarray: Filtered umap_embeddings.
            np.ndarray: Filtered label_data.
            pd.DataFrame: Filtered df_umap_tiles.
        """
        # Apply all filters to df_umap_tiles
        mask = np.ones(len(self.df_umap_tiles), dtype=bool)  # Start with all True
        for column, values in filters.items():
            # Checkbox filters    
            # Strip counts (e.g., 'Batch4 (4384)' -> 'Batch4')
            cleaned_values = [v.split(' (')[0] for v in values]

            mask &= self.df_umap_tiles[column].apply(
                lambda x: any(str(x).startswith(prefix) for prefix in cleaned_values)
            )
        # Apply mask to all data
        self.umap_embeddings_filt = self.umap_embeddings[mask]
        self.label_data_filt = self.label_data[mask]
        self.df_umap_tiles_filt = self.df_umap_tiles.iloc[list(mask)].copy().reset_index(drop=True)

    def get_active_filter_values(self):
        """
        Returns a list of all applied UMAP filters.

        Each item in the list is a list of selected values for a specific filter group (e.g., Cell Line, Condition).
        The returned values reflect the user's current selections and are used to annotate saved UMAPs.
        """
        active_filter_groups = []

        filters = getattr(self, "applied_filters", {})

        # Get all checkbox options per filter column
        all_options = {
            col: [cb.description for cb in cbs]
            for col, cbs in self.filter_checkboxes.items()
        }
        # Get all options per filter column in the more filters section
        for name, dropdown in self.advanced_filter_dropdowns.items():
            if dropdown is not None:
                all_options[name] = list(dropdown.options)

        for col, selected_values in filters.items():
            # Clean both selected and full options to ignore counts like "WT (42)"
            selected_clean = sorted([v.split(" (")[0] for v in selected_values])
            full_clean = sorted([v.split(" (")[0] for v in all_options.get(col, [])])

            # Only keep the filter if the user selected a subset
            if selected_clean != full_clean:
                active_filter_groups.append(selected_clean)

        return active_filter_groups

    def get_save_folder_path(self, root_folder="saved_umaps"):
        """
        Creates a timestamped folder in the specified root folder to save UMAP outputs.
        Args:
            root_folder (str): The base folder where the timestamped folder will be created.
        Returns:
            str: Full path to the created folder.
        """
        os.makedirs(root_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        base = self.exp_name.strip().replace(" ", "_") if getattr(self, "exp_name", "") else "experiment"
        folder_name = f"{base}_{timestamp}"
        full_path = os.path.join(root_folder, folder_name)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def write_umap_params(self, folder_path):
        """
        Writes the current UMAP parameters and filter settings to a text file in the specified folder.
        Args:
            folder_path (str): Path to the folder where the parameters will be saved.
        """
        dropdown_parts = [
            ("umap_type", self.umap_type_dropdown.value),
            ("batch", f"batch{self.batch_dropdown.value}"),
            ("rep", self.reps_dropdown.value),
            ("coloring", self.coloring_dropdown.value),
            ("marker", self.marker_dropdown.value),
            ("cell_line", self.cell_line_dropdown.value),
            ("condition", self.condition_dropdown.value),
        ]
        filters = self.get_active_filter_values()
        filter_str = "\n".join([f"filter_{i}: {','.join(vals)}" for i, vals in enumerate(filters) if vals])

        param_lines = [f"{k}: {str(v)}" for k, v in dropdown_parts if v]
        
        raw_html = self.pickle_status_label.value
        match = re.search(r"<code>(.*?)</code>", raw_html)
        pickle_path = match.group(1) if match else "N/A"
        param_lines.append(f"pickle_file: {pickle_path}")    

        param_lines.append("filters:\n" + filter_str)
        param_lines.append(f"MIX_GROUPS: {True if self.mix_groups_checkbox.value else False}")
        param_lines.append(f"RECOLOR_BY_BRENNER: {True if self.recolor_checkbox.value else False}")
        param_lines.append(f"BRENNER_bins: {self.bins_slider.value}")
        param_lines.append(f"DILUTE: {self.dilute_slider.value}")
        param_lines.append(f"Selected Points: {len(self.selected_indices_global)}")

        with open(os.path.join(folder_path, "params.txt"), "w") as f:
            f.write("\n".join(param_lines))

    def save_umap(self, folder_path, dpi=300):
        """
        Saves the current UMAP figure to the specified folder.
        Args:
            folder_path (str): Path to the folder where the UMAP figure will be saved.
            dpi (int): Dots per inch for the saved figure.
        """
        if not hasattr(self, "current_umap_figure") or self.current_umap_figure is None:
            print("⚠️ No UMAP figure found to save.")
            return
        self.current_umap_figure.savefig(os.path.join(folder_path, "umap.png"), dpi=dpi, bbox_inches='tight')

    def save_selected_images(self, folder_path, img_path = 'images_and_tiles'):
        """
        Saves the selected images and tiles to the specified folder.
        Args:
            folder_path (str): Path to the folder where images will be saved.
            img_path (str): Subfolder name for images and tiles.
        """
        if not self.selected_indices_global:
            print("⚠️ No selected points to save images.")
            return

        df_to_use = self.df_umap_tiles_filt.copy() if self.df_umap_tiles_filt is not None else self.df_umap_tiles.copy()

        image_folder = os.path.join(folder_path, img_path)
        os.makedirs(image_folder, exist_ok=True)

        for idx in self.selected_indices_global[:self.num_images_slider.value]:
            if self.df_site_meta is not None:
                target_path = construct_target_path(df_to_use, idx, self.df_site_meta)
                if target_path != -1:
                    save_processed_tif_image(target_path, idx, image_folder)
            else:
                print("❌ df_site_meta missing — skipping TIFF save.")

            # Save tile visualizations
            save_processed_tile(df_to_use, idx, image_folder)
        
        self.zip_saved_outputs(image_folder) ## Zip the saved images and tiles folder for easy download
        # Clean up the folder after zipping
        shutil.rmtree(image_folder)

    def create_fov_grid(self):
        """
        Create a grid layout for FOV visualization based on batch, panel and self.fov_layouts definition.
        """
        batch = self.df_umap_tiles["Batch"].unique()
        panel = self.df_umap_tiles["Panel"].unique()

        if len(batch) != 1 or len(panel) != 1:
            print("⚠️ FOV grid requires exactly one batch and one panel.")
            ## TODO: concider combining grid from multiple batches and panels
            return None
        
        batch = batch[0]
        panel = panel[0]

        if batch not in self.fov_layouts or panel not in self.fov_layouts[batch]:
            print(f"❌ Unknown Batch/Panel: {batch}, {panel}")
            return None

        fov_grid = self.fov_layouts[batch][panel]
        return fov_grid

    def save_fov_visualizations(self, folder_path):            
        """
        Saves FOV visualizations (histogram and heatmap) to the specified folder.
        Args:
            folder_path (str): Path to the folder where FOV visualizations will be saved.
        """
        try:
            fig1 = plot_fov_histogram(self.df_umap_tiles, self.selected_indices_global, return_fig=True)
            fig1.savefig(os.path.join(folder_path, "fov_histogram.png"))
            plt.close(fig1)

            if not self.fov_layouts:
                print("⚠️ No FOV layouts available.")
            else:
                fov_grid = self.create_fov_grid()
                if fov_grid is not None:
                    fig2 = plot_fov_heatmaps(self.df_umap_tiles, self.selected_indices_global, fov_grid, return_fig=True)
                    fig2.savefig(os.path.join(folder_path, "fov_heatmap.png"))
                    plt.close(fig2)
        except Exception as e:
            print(f"❌ Failed to save FOV plots: {e}")

    def save_all(self, btn=None, folder="saved_umaps", dpi=300):
        """
        Saves all outputs (UMAP, images, FOV visualizations) to a specified folder.
        Args:
            btn: Button that triggered the save action (optional).
            folder (str): Base folder name where outputs will be saved.
            dpi (int): Dots per inch for the saved UMAP figure.
        """
        self.save_status_label.value = "⏳ Saving outputs..."
        folder_path = self.get_save_folder_path(folder)

        with self.warnings_output:
            try:
                self.write_umap_params(folder_path)
                self.save_umap(folder_path, dpi)
                self.save_selected_images(folder_path)
                self.save_fov_visualizations(folder_path)
                self.save_status_label.value = f"<span style='color: green;'>✅ Outputs saved in:</span> <code>{folder_path}</code>"
            except Exception as e:
                self.save_status_label.value = f"<span style='color: red;'>❌ Error saving outputs:</span> {e}"
            clear_output()

    def zip_saved_outputs(self, folder_path):
        """
        Zips the specified folder containing saved images and tiles.
        Args:
            folder_path (str): Path to the folder to be zipped.
        """
        zip_path = folder_path + ".zip"
        try:
            shutil.make_archive(base_name=folder_path, format="zip", root_dir=folder_path)
        except Exception as e:
            print(f"❌ Failed to zip folder: {e}")

    def show_selected_images(self, btn):
        """
        Displays images and tiles corresponding to the selected UMAP points.
        Args:
            btn: Button that triggered the action (optional).
        """
        self.clear_outputs(umaps=False)
        if not self.selected_indices_global:
            with self.selected_images_output_inner:
                print("⚠️ No points selected. Please choose points on the UMAP first.")
            return
        # Set labels above each section
        self.selected_images_label.layout.display = 'inline-block'
        self.selected_tiles_label.layout.display = 'inline-block'
        self.fov_label.layout.display = 'inline-block'
        
        df_to_use = self.df_umap_tiles_filt.copy() if self.df_umap_tiles_filt is not None else self.df_umap_tiles.copy()

        # Section 1: Images only
        with self.selected_images_output_inner:
            if self.df_site_meta is not None:
                for ind in self.selected_indices_global[:self.num_images_slider.value]:
                    # print(ind, df_to_use.iloc[ind]['Target_Sharpness_Brenner'])
                    target_path = construct_target_path(df_to_use, ind, self.df_site_meta)
                    if target_path != -1:
                        show_processed_tif(target_path)
                        plt.show()
                        time.sleep(0.2)
                    else:
                        break
            else:
                print("❌ Please specify the image directory to enable image display.")

        time.sleep(2)
        # Section 2: Tiles only
        with self.selected_tiles_output_inner:
            for ind in self.selected_indices_global[:self.num_images_slider.value]:
                print(ind, df_to_use.Path.iloc[ind])
                show_processed_tile(df_to_use, ind)
                plt.show()
                time.sleep(0.2)

        time.sleep(3)
        # Section 3: FOV
        with self.fov_output:
            if self.fov_layouts:
                fov_grid = self.create_fov_grid()
                if fov_grid is not None:
                    plot_fov_heatmaps(self.df_umap_tiles, self.selected_indices_global, fov_grid)
            else:
                print("❌ Please specify the FOV layout to display FOV map.")
            plot_fov_histogram(self.df_umap_tiles, self.selected_indices_global)
        check_memory_status()
                
    def clear_outputs(self, selected_points=True, umaps=True): 
        if selected_points:
            with self.selected_images_output_inner: clear_output()
            with self.selected_tiles_output_inner: clear_output()
            with self.fov_output: clear_output()
            with self.warnings_output: clear_output()
        if umaps:
            with self.umap_output: clear_output()
            self.right_box.layout.display = 'none'
            self.image_display_controls.layout.display = 'none'

    def make_text_widget(self, value, description, tooltip=''):
        label = widgets.HTML(
            f"""
            <label style="display:inline-block; width:200px;" title="{tooltip}">
                {description}
            </label>
            """
        )
        text = Text(value=value, layout=Layout(width='700px'))
        container = widgets.HBox([label, text])
        container.text_input = text  # attach text widget for later access
        return container

    def create_checkbox_group(self, column): 
        value_counts = self.df_umap_tiles[column].value_counts()
        values = sorted(value_counts.index.dropna())
        checkboxes = [
            widgets.Checkbox(
                value=True, 
                description=f"{v} ({value_counts[v]})"
            ) for v in values
        ]
        self.filter_checkboxes[column] = checkboxes
        return widgets.VBox([widgets.Label(f"{column} Filter:")] + checkboxes)

    def populate_dropdowns_from_umaps(self):
        def safe_set(dropdown, options, default=None):
            options = sorted(list(options))
            dropdown.options = options
            if options:
                dropdown.value = default if default in options else options[0]

        def clear_dropdown(dropdown):
            dropdown.options = []
            dropdown.value = None

        # Step 0: clear all but umap_type
        for d in [
            self.batch_dropdown, self.reps_dropdown, self.coloring_dropdown,
            self.marker_dropdown, self.cell_line_dropdown, self.condition_dropdown
        ]:
            clear_dropdown(d)

        # Step 1: populate UMAP Type only
        umap_types = self.df_umap_meta['Umap_Type'].dropna().unique()
        safe_set(self.umap_type_dropdown, umap_types, 'SINGLE_MARKERS')

        # Step 2: when UMAP Type is selected → populate Batch and hook Batch callback
        def on_umap_type_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                filtered = self.df_umap_meta[self.df_umap_meta['Umap_Type'] == change['new']]
                batch_options = filtered['Batch'].dropna().unique()
                safe_set(self.batch_dropdown, batch_options, '4')

                # Inner function: populate rest based on batch + umap_type
                def populate_rest(batch_value):
                    subset = filtered[filtered['Batch'] == batch_value]
                    if subset.empty:
                        return

                    first_row = subset.iloc[0]
                    safe_set(self.reps_dropdown, subset['Rep'].dropna().unique(), first_row['Rep'])
                    safe_set(self.coloring_dropdown, subset['Coloring'].dropna().unique(), first_row['Coloring'])
                    safe_set(self.marker_dropdown, subset['Marker'].dropna().unique(), first_row['Marker'])
                    safe_set(self.cell_line_dropdown, subset['CellLine'].dropna().unique(), first_row['CellLine'])
                    safe_set(self.condition_dropdown, subset['Condition'].dropna().unique(), first_row['Condition'])

                # Step 3: hook Batch change → update rest
                def on_batch_change(bchange):
                    if bchange['type'] == 'change' and bchange['name'] == 'value':
                        populate_rest(bchange['new'])

                self.batch_dropdown.observe(on_batch_change, names='value')

                # Trigger batch logic once immediately using selected default
                populate_rest(self.batch_dropdown.value)

        self.umap_type_dropdown.observe(on_umap_type_change, names='value')

        # Trigger UMAP type callback once to start the flow
        self.umap_type_dropdown.value = self.umap_type_dropdown.value
        # Manually trigger once at startup
        on_umap_type_change({'type': 'change', 'name': 'value', 'new': self.umap_type_dropdown.value})

    def build_filters_from_checkboxes(self):
        filters = {}
        for col, checkboxes in self.filter_checkboxes.items():
            selected = [cb.description for cb in checkboxes if cb.value]
            if selected:
                filters[col] = selected

        for name, dropdown in self.advanced_filter_dropdowns.items():
            selected = list(dropdown.value)
            if selected:
                filters[name] = selected

        return filters
    
    def plot_interactive_umap(self, 
        title: str = None,
        dpi: int = 500,
        figsize: tuple = (6, 5),
        cmap: str = 'tab20',
        ari_score: float = None,
    ):
        self.selected_indices_global = []
        self.rect_selector = None

        df_umap_tiles = self.df_umap_tiles_filt.copy() if self.df_umap_tiles_filt is not None else self.df_umap_tiles.copy()
        umap_embeddings = self.umap_embeddings_filt.copy() if self.umap_embeddings_filt is not None else self.umap_embeddings.copy()
        label_data = self.label_data_filt.copy() if self.label_data_filt is not None else self.label_data.copy()
        RECOLOR_BY_BRENNER=self.recolor_checkbox.value
        dilute=self.dilute_slider.value
        bins=self.bins_slider.value

        """Plots UMAP embeddings with interactive hovering for labels, with optional data dilution."""
        if umap_embeddings.shape[0] != label_data.shape[0]:
            print("⚠️ The number of embeddings and labels must match.")
            return
        
        if len(df_umap_tiles) == 0:
            print("⚠️ No matching data found.              \nTry adjusting your filters.")
            return
        
        original_indices = np.arange(len(df_umap_tiles))#[::dilute]
        annotations_dict = {}; colors_dict = {}; scatter_mappings = {}

        if df_umap_tiles is not None:
            image_names_dict = {idx: row.Image_Name for idx, row in df_umap_tiles.iterrows()}
            if ("Target_Sharpness_Brenner" in df_umap_tiles.columns) and not (df_umap_tiles["Target_Sharpness_Brenner"].isna().all()):
                brenner_scores_dict = {idx: row.Target_Sharpness_Brenner for idx, row in df_umap_tiles.iterrows()}
                annotations_dict = {
                    idx: f"{idx}: {image_names_dict.get(idx, 'Unknown')}\nBrenner Score: {brenner_scores_dict.get(idx, 'N/A')}"
                    for idx in df_umap_tiles.index
                }
            else:
                annotations_dict = {idx: f"{idx}: {image_names_dict.get(idx, 'Unknown')}" for idx in df_umap_tiles.index}
            if RECOLOR_BY_BRENNER:
                if ('Path_List' not in df_umap_tiles) and ("Target_Sharpness_Brenner" in df_umap_tiles.columns) and not (df_umap_tiles["Target_Sharpness_Brenner"].isna().all()):
                    df_umap_tiles["Color"], percentiles, cmap = set_colors_by_brenners(df_umap_tiles["Target_Sharpness_Brenner"].fillna(0), bins=bins)
                    colors_dict = {idx: row.Color for idx, row in df_umap_tiles.iterrows()}
                elif ('Path_List' in df_umap_tiles):
                    print("❌ Recoloring by Brenner score is not possible for Multiplexed Embeddings.")
                else:
                    print("❌ Please specify the correct Brenner csv path to enable recoloring by Brenner score.")

        name_key, color_key = self.config_plot['MAPPINGS_ALIAS_KEY'], self.config_plot['MAPPINGS_COLOR_KEY']
        marker_size, alpha = self.config_plot['SIZE'], self.config_plot['ALPHA']
        mix_groups = self.config_plot['MIX_GROUPS']
        cmap = plt.get_cmap(cmap)
        unique_groups = np.unique(label_data)
        name_color_dict = self.config_plot['COLOR_MAPPINGS'] if self.config_plot['COLOR_MAPPINGS'] is not None else {
            group: {color_key: cmap(i / (len(unique_groups) - 1)), name_key: group} for i, group in enumerate(unique_groups)}

        fig, ax = plt.subplots(figsize=figsize)
        self.current_umap_figure = fig
        scatter_objects = []
        legend_labels = []
        indices = []
        colors = []
        for group in unique_groups:
            group_indices = np.where(label_data == group)[0]
            if RECOLOR_BY_BRENNER and (df_umap_tiles is not None) and ("Target_Sharpness_Brenner" in df_umap_tiles.columns) and not (df_umap_tiles["Target_Sharpness_Brenner"].isna().all()):
                rgba_colors = [colors_dict.get(idx, "#000000") for idx in original_indices[group_indices]]
            else:
                base_color = name_color_dict[group][color_key]
                rgba_colors = [mcolors.to_rgba(base_color, alpha=alpha)] * len(group_indices)
            if not mix_groups:
                scatter = ax.scatter(
                    umap_embeddings[group_indices, 0],
                    umap_embeddings[group_indices, 1],
                    s=marker_size,
                    alpha=alpha,
                    c=rgba_colors,
                    marker='o',
                    label=name_color_dict[group][name_key]
                )
                scatter_objects.append(scatter)
                legend_labels.append(name_color_dict[group][name_key])
                scatter_mappings[scatter] = group_indices.tolist()
            else:
                colors.append(rgba_colors)
                indices.append(group_indices)
        if mix_groups:
            colors = np.concatenate(colors)
            indices = np.concatenate(indices)
            shuffled_indices = np.random.permutation(len(indices))
            shuffled_colors = colors[shuffled_indices]
            scatter = ax.scatter(
                    umap_embeddings[indices][shuffled_indices,0],
                    umap_embeddings[indices][shuffled_indices,1],
                    s=marker_size,
                    alpha=alpha,
                    c=shuffled_colors,
                    marker='o',
                )    
            scatter_objects = [scatter]
            legend_labels = [name_color_dict[group][name_key] for group in unique_groups]
            scatter_mappings[scatter] = indices[shuffled_indices].tolist()

        if self.hover:
            # Enable interactive hovering with precomputed labels
            cursor = mplcursors.cursor(scatter_objects, hover=True)

            @cursor.connect("add")
            def on_hover(sel):
                scatter_obj = sel.artist
                scatter_index = sel.index

                if scatter_obj in scatter_mappings:
                    actual_index = original_indices[scatter_mappings[scatter_obj][scatter_index]] # Correct mapping
                    sel.annotation.set_text(annotations_dict.get(actual_index, "Unknown"))
                else:
                    sel.annotation.set_text("Unknown")

        # **Rectangle Selection Functionality**
        def on_select(eclick, erelease):
            """Handles rectangle selection and stores selected point indices."""

            if eclick.xdata is None or erelease.xdata is None:
                return  # Ignore invalid selections

            x_min, x_max = sorted([eclick.xdata, erelease.xdata])
            y_min, y_max = sorted([eclick.ydata, erelease.ydata])

            # Find selected points in the downsampled (diluted) dataset
            selected_indices_diluted = [
                i for i in range(len(umap_embeddings))
                if x_min <= umap_embeddings[i, 0] <= x_max and y_min <= umap_embeddings[i, 1] <= y_max
            ]

            # Map selected indices back to the original dataset
            self.selected_indices_global = original_indices[selected_indices_diluted].tolist()

        # Attach Rectangle Selector (global storage prevents garbage collection)
        self.rect_selector = RectangleSelector(ax, on_select, interactive=True, useblit=False)

        if RECOLOR_BY_BRENNER:
            if (self.df_brenner is not None) and ("Target_Sharpness_Brenner" in df_umap_tiles.columns) and not (df_umap_tiles["Target_Sharpness_Brenner"].isna().all()):
                # Create colorbar
                norm = mcolors.BoundaryNorm(percentiles, cmap.N)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, ticks=percentiles, fraction=0.046, pad=0.04)
                cbar.ax.set_yticklabels([
                    f'{int(p)} ({round(q)}%)' for p, q in zip(percentiles, np.linspace(0, 100, len(percentiles)))
                ])
                cbar.ax.tick_params(labelsize=6) 
                cbar.set_label('Target Sharpness (Brenner Score)', fontsize=8)
        else:
            if mix_groups:
                # Manually create handles and labels
                handles = []
                labels = []
                for group in unique_groups:
                    color = name_color_dict[group][color_key]
                    label = name_color_dict[group][name_key]
                    patch = mpatches.Patch(color=color, label=label)
                    handles.append(patch)
                    labels.append(label)
                self.format_UMAP_legend(ax, marker_size, handles=handles, labels=labels)
            else:
                self.format_UMAP_legend(ax, marker_size, scatter_objects, legend_labels)

        self.format_UMAP_axes(ax, title)
        fig.tight_layout()
        plt.show()
        print("UMAP plot ready. \nSelect points and then click the button below to show selected images.")
        return
    
    def dispose(self):
        # Clear widgets, outputs, and figures
        for attr in ['umap_output', 'selected_images_output_inner', 'selected_tiles_output_inner', 'fov_output']:
            if hasattr(self, attr):
                getattr(self, attr).clear_output()
        clear_output(wait=True)
        plt.close('all')
        gc.collect()  # Force garbage collection
        self.trim_malloc()
    
    def trim_malloc(self):
        """Ask glibc to return free heap pages to the OS."""
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
