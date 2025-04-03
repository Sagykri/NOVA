import os
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import Button, Output, Text, Layout
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.figures.umap_plotting import __format_UMAP_axes as format_UMAP_axes
from src.figures.umap_plotting import __format_UMAP_legend as format_UMAP_legend
from interactive_umaps_utils import *
from tools.show_images_utils import show_processed_tif, extract_image_metadata


class InteractiveUMAPPipeline:
    def __init__(self, fov_layouts={}, default_paths = {}):
        self.filter_checkboxes = {}
        self.selected_indices_global = []
        self.rect_selector = None  # Persistent RectangleSelector

        self.df_meta = None
        self.dfb = None ## Brenner scores
        self.df_umaps = None ## metadata on all umaps in umap folder
        self.df_image_stats = None ## metadata information related to the images in the displayed umap
        self.df_image_stats_filt = None ## After applying filters
        self.umap_embeddings = None ## The embeddings of the displayed umap
        self.umap_embeddings_filt = None ## After applying filters
        self.label_data = None ## The labels of the displayed umap
        self.label_data_filt = None ## After applying filters
        self.config_data = None ## Saved with the original umap
        self.config_plot = None ## Saved with the original umap

        self._create_widgets(default_paths)
        self._setup_callbacks()
        self._display_ui()
        self.fov_layouts = fov_layouts
        self.format_UMAP_legend = format_UMAP_legend
        self.format_UMAP_axes = format_UMAP_axes

    def _create_widgets(self, default_paths):
        # --- Path configuration widgets ---
        self.umaps_dir_widget = self.make_text_widget(default_paths['umaps_folder'], 'UMAPs Dir:')
        self.preprocessing_path_widget = self.make_text_widget(default_paths['preprocessing_path'],'Preprocessing/brenner Path:')
        self.csv_name1_widget = self.make_text_widget(default_paths['csv_name1'], 'CSV Brenner 1:')
        self.csv_name2_widget = self.make_text_widget(default_paths['csv_name2'], 'CSV Brenner 2 (if exists):')
        self.images_dir_widget = self.make_text_widget(default_paths['images_dir'],'Raw images Dir:')

        # --- Output display areas ---
        self.output_area = Output(layout={'height': '50px', 'margin': '0px 0px'})
        self.umap_output = Output(layout={'height': '1800px', 'margin': '10px 0px'})
        self.selected_images_output, self.selected_images_output_inner = self.create_scrollable_output(height='400px')
        self.selected_tiles_output, self.selected_tiles_output_inner = self.create_scrollable_output(height='350px')
        self.fov_output = Output(layout={'height': '1000px', 'margin': '0 auto', 'display': 'block'})

        # --- Control buttons ---
        self.run_button = Button(description="Run", layout=Layout(width='200px', margin='5px 250px', ))
        self.create_umap_button = Button(description="Create UMAP", layout=Layout(width='200px', margin='5px 10px'))
        self.create_umap_button.layout.display = 'none'
        self.show_images_button = Button(description="Show Selected Points", layout=Layout(width='200px', margin='0px 10px'))
        self.show_images_button.layout.display = 'none'
        self.apply_filter_button = Button(description="Apply Filters", layout=Layout(width='180px', margin='10px 0px 0px 0px'))
        self.apply_filter_button.layout.display = 'none'

        self.num_images_slider = widgets.IntSlider(
            value=10, min=1, max=30, step=1,
            description='Num images:',
            layout=Layout(width='250px', display='none'),
            style={'description_width': '100px'}
        )

        # --- UMAP dropdowns ---
        self.batch_dropdown = widgets.Dropdown(description='Batch:')
        self.umap_type_dropdown = widgets.Dropdown(description='UMAP Type:')
        self.reps_dropdown = widgets.Dropdown(description='Reps:')
        self.coloring_dropdown = widgets.Dropdown(description='Coloring:')
        self.marker_dropdown = widgets.Dropdown(description='Marker:')
        self.cell_line_dropdown = widgets.Dropdown(description='Cell Line:')
        self.condition_dropdown = widgets.Dropdown(description='Condition:')

        # --- Additional plot controls ---
        self.pickle_status_label = widgets.HTML()
        self.mix_groups_checkbox = widgets.Checkbox(value=True, description='Mix Groups')
        self.recolor_checkbox = widgets.Checkbox(value=False, description='Recolor by Brenner')
        self.dilute_slider = widgets.IntSlider(value=1, min=1, max=100, step=1, description='Dilute', style={'description_width': '100px'})
        self.bins_slider = widgets.IntSlider(value=5, min=1, max=20, step=1, description='Bins (Brenner)', style={'description_width': '100px'})

        # --- Layout containers ---
        self.umap_params = widgets.VBox([
            widgets.HTML("<b>Select UMAP Parameters (not all combinations are available):</b>"),
            self.batch_dropdown,
            self.umap_type_dropdown,
            self.reps_dropdown,
            self.coloring_dropdown,
            self.marker_dropdown,
            self.cell_line_dropdown,
            self.condition_dropdown,
            self.mix_groups_checkbox,
            self.recolor_checkbox,
            self.dilute_slider,
            self.bins_slider,
            self.pickle_status_label,
            self.create_umap_button
        ], layout=Layout(display='none', gap='40px'))

        self.right_box = widgets.VBox(layout=Layout(display='none', width='320px'))  # Populated later with filters

        # --- Filter checkboxes (populated dynamically) ---
        self.filter_checkboxes = {}

        # --- Section labels ---
        self.selected_images_label = widgets.HTML(value="<b>1. Selected Image Previews:</b>", layout=widgets.Layout(display='none'))
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

    def _display_ui(self):
        self.ui = widgets.VBox([
            self.umaps_dir_widget,
            self.preprocessing_path_widget,
            self.csv_name1_widget,
            self.csv_name2_widget,
            self.images_dir_widget,
            self.run_button,
            widgets.HTML("<hr>"),
            self.output_area,
            self.umap_params,
            widgets.HTML("<hr>"),
            widgets.HBox([self.umap_output, self.right_box]),
            widgets.HBox([self.num_images_slider, self.show_images_button]),
            widgets.HTML("<hr>"),
            self.selected_images_label,
            self.selected_images_output,
            widgets.HTML("<hr>"),
            self.selected_tiles_label,
            self.selected_tiles_output,
            widgets.HTML("<hr>"),
            self.fov_label,
            self.fov_output
        ], layout=widgets.Layout(padding='10px', min_height='2500px', height='auto', overflow_y='visible'))

    def show(self):
        display(self.ui)
        
    def reset(self):
        self.filter_checkboxes.clear()  # Clear old filters first
        self.umap_embeddings, self.label_data, self.config_data, self.config_plot, self.df_image_stats = None, None, None, None, None
        self.umap_embeddings_filt, self.label_data_filt, self.df_image_stats_filt = None, None, None

    def run_pipeline(self, btn):
        self.clear_outputs()
        self.reset()
        with self.output_area:
            clear_output()
            print('Searching for all UMAPs in folder... (~10 seconds)')
            umaps_dir = self.umaps_dir_widget.value
            preprocessing_path = self.preprocessing_path_widget.value
            csv_name1 = self.csv_name1_widget.value
            csv_name2 = self.csv_name2_widget.value
            images_dir = self.images_dir_widget.value

            df1 = pd.read_csv(os.path.join(preprocessing_path, csv_name1))
            if csv_name2.strip():  # check if not empty
                df2 = pd.read_csv(os.path.join(preprocessing_path, csv_name2))
                dfb = pd.concat([df1, df2], ignore_index=True)
            else:
                dfb = df1.copy()
            dfb["Image_Name"] = dfb["Path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            self.dfb = dfb

            self.df_meta = extract_image_metadata(images_dir, FILE_EXTENSION='.tiff', KEY_BATCH='Batch')
            self.df_umaps = extract_umap_data(base_dir=umaps_dir)
            clear_output()
            print(len(self.df_umaps), 'UMAPs were located')

            # Populate dropdowns from df_umaps
            self.populate_dropdowns_from_umaps()
            self.umap_params.layout.display = 'flex'
            self.create_umap_button.layout.display = 'inline-block'
            
    def create_umap(self, btn): 
        self.reset()
        with self.umap_output:
            clear_output()
        with self.umap_output:
            pickle_file_path = get_umap_pickle_path(
                self.df_umaps,
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

            self.umap_embeddings, self.label_data, self.config_data, self.config_plot, self.df_image_stats = load_and_process_data(
                self.umaps_dir_widget.value, pickle_file_path, self.dfb)
            
            # Apply dilution
            dilute=self.dilute_slider.value
            self.umap_embeddings = self.umap_embeddings[::dilute]
            self.label_data = self.label_data[::dilute]
            self.df_image_stats = self.df_image_stats.iloc[::dilute].copy()
            self.df_image_stats.index = list(range(len(self.df_image_stats)))            

            self.config_plot['MIX_GROUPS'] = self.mix_groups_checkbox.value

            self.plot_interactive_umap()

            # Dynamically populate checkbox filters
            self.right_box.children = [
                widgets.HTML("<b>Filter Settings:</b>")
            ] + [
                self.create_checkbox_group(col) for col in ['Batch', 'Condition', 'Rep', 'Panel', 'Cell_Line']
            ] + [self.apply_filter_button]

        self.apply_filter_button.layout.display = 'inline-block'
        self.right_box.layout.display = 'flex'
        self.show_images_button.layout.display = 'inline-block'
        self.num_images_slider.layout.display = 'flex'
        self.clear_outputs(umaps=False)
        
    def apply_filters_and_update_plot(self, btn): 
        with self.umap_output:
            clear_output()
    
            # Build filters dictionary
            filters = self.build_filters_from_checkboxes()
            self.filter_umap_data(filters=filters)

            self.plot_interactive_umap()
            self.clear_outputs(umaps=False)

    def show_selected_images(self, btn):
        self.clear_outputs(umaps=False)
        if not self.selected_indices_global:
            with self.selected_images_output_inner:
                print("⚠️ No points selected. Please choose points on the UMAP first.")
            return
        # Set labels above each section
        self.selected_images_label.layout.display = 'inline-block'
        self.selected_tiles_label.layout.display = 'inline-block'
        self.fov_label.layout.display = 'inline-block'
        
        df_to_use = self.df_image_stats_filt.copy() if self.df_image_stats_filt is not None else self.df_image_stats.copy()

        # Section 1: Images only
        with self.selected_images_output_inner:
            for ind in self.selected_indices_global[:self.num_images_slider.value]:
                print(ind, df_to_use.iloc[ind]['Target_Sharpness_Brenner'])
                target_path = construct_target_path(df_to_use, ind, self.df_meta)
                show_processed_tif(target_path)
                plt.show()

        time.sleep(2)
        # Section 2: Tiles only
        with self.selected_tiles_output_inner:
            for ind in self.selected_indices_global[:self.num_images_slider.value]:
                print(ind, df_to_use.Path.iloc[ind])
                show_processed_tile(df_to_use, ind)
                plt.show()

        time.sleep(2)
        # Section 3: FOV
        with self.fov_output:
            batch = df_to_use["Batch"].iloc[0]
            panel = df_to_use["Panel"].iloc[0].split('panel')[1]

            if batch not in self.fov_layouts or panel not in self.fov_layouts[batch]:
                raise ValueError(f"Unknown Batch/Panel: {batch}, {panel}")

            fov_grid = self.fov_layouts[batch][panel]
            plot_fov_histogram(df_to_use, self.selected_indices_global)
            plot_fov_heatmaps(df_to_use, self.selected_indices_global, fov_grid)
            
    def clear_outputs(self, selected_points=True, umaps=True): 
        if selected_points:
            with self.selected_images_output_inner: clear_output()
            with self.selected_tiles_output_inner: clear_output()
            with self.fov_output: clear_output()
        if umaps:
            with self.umap_output: clear_output()
        
    def make_text_widget(self, value, description):
        return Text(
            value=value,
            description=description,
            layout=Layout(width='900px'),
            style={'description_width': '200px'}
        )

    def create_checkbox_group(self, column):
        values = sorted(self.df_image_stats[column].dropna().unique())
        checkboxes = [widgets.Checkbox(value=True, description=str(v)) for v in values]
        self.filter_checkboxes[column] = checkboxes
        return widgets.VBox([widgets.Label(f"{column} Filter:")] + checkboxes)

    def populate_dropdowns_from_umaps(self):
        def safe_set(dropdown, options, default):
            dropdown.options = sorted(options)
            dropdown.value = default if default in dropdown.options else dropdown.options[0]

        safe_set(self.batch_dropdown, self.df_umaps['batch'].unique(), '4')
        safe_set(self.umap_type_dropdown, self.df_umaps['umap_type'].unique(), 0)
        safe_set(self.reps_dropdown, self.df_umaps['rep'].unique(), 'all_reps')
        safe_set(self.coloring_dropdown, self.df_umaps['coloring'].unique(), 'CONDITIONS')
        safe_set(self.marker_dropdown, self.df_umaps['markers'].unique(), 'TDP-43')
        safe_set(self.cell_line_dropdown, self.df_umaps['cell_line'].unique(), 'all_cell_lines')
        safe_set(self.condition_dropdown, self.df_umaps['condition'].unique(), 'all_conditions')

    def build_filters_from_checkboxes(self):
        filters = {}
        for col, checkboxes in self.filter_checkboxes.items():
            selected = [cb.description for cb in checkboxes if cb.value]
            if selected:
                filters[col] = selected
        return filters
    
    def filter_umap_data(self, filters: dict):
        """
        Filters umap_embeddings, label_data, and df_image_stats based on values in filters.

        Args:
            umap_embeddings (np.ndarray): 2D array of shape (N, 2) containing UMAP embeddings.
            label_data (np.ndarray): 1D array of shape (N,) containing labels for each embedding.
            df_image_stats (pd.DataFrame): DataFrame containing image statistics.
            filters (dict): Dictionary where keys are column names in df_image_stats and values are lists of allowed values.

        Returns:
            np.ndarray: Filtered umap_embeddings.
            np.ndarray: Filtered label_data.
            pd.DataFrame: Filtered df_image_stats.
        """
        # Apply all filters to df_image_stats
        mask = np.ones(len(self.df_image_stats), dtype=bool)  # Start with all True
        for column, values in filters.items():
            mask &= self.df_image_stats[column].apply(lambda x: any(str(x).startswith(prefix) for prefix in values))

        # Apply mask to all data
        self.umap_embeddings_filt = self.umap_embeddings[mask]
        self.label_data_filt = self.label_data[mask]
        self.df_image_stats_filt = self.df_image_stats.iloc[list(mask)].copy().reset_index()
            
    def plot_interactive_umap(self, 
        title: str = None,
        dpi: int = 500,
        figsize: tuple = (6, 5),
        cmap: str = 'tab20',
        ari_score: float = None,
    ):
        
        df_image_stats = self.df_image_stats_filt.copy() if self.df_image_stats_filt is not None else self.df_image_stats.copy()
        umap_embeddings = self.umap_embeddings_filt.copy() if self.umap_embeddings_filt is not None else self.umap_embeddings.copy()
        label_data = self.label_data_filt.copy() if self.label_data_filt is not None else self.label_data.copy()
        RECOLOR_BY_BRENNER=self.recolor_checkbox.value
        dilute=self.dilute_slider.value
        bins=self.bins_slider.value

        """Plots UMAP embeddings with interactive hovering for labels, with optional data dilution."""
        if umap_embeddings.shape[0] != label_data.shape[0]:
            raise ValueError("The number of embeddings and labels must match.")
        
        original_indices = np.arange(len(df_image_stats))#[::dilute]
        annotations_dict = {}; colors_dict = {}; scatter_mappings = {}

        if df_image_stats is not None:
            image_names_dict = {idx: row.Image_Name for idx, row in df_image_stats.iterrows()}
            if "Target_Sharpness_Brenner" in df_image_stats.columns:
                brenner_scores_dict = {idx: row.Target_Sharpness_Brenner for idx, row in df_image_stats.iterrows()}
                annotations_dict = {
                    idx: f"{idx}: {image_names_dict.get(idx, 'Unknown')}\nBrenner Score: {brenner_scores_dict.get(idx, 'N/A')}"
                    for idx in df_image_stats.index
                }
                if RECOLOR_BY_BRENNER:
                    df_image_stats["Color"] = set_colors_by_brenners(df_image_stats["Target_Sharpness_Brenner"].fillna(0), bins=bins)
                    colors_dict = {idx: row.Color for idx, row in df_image_stats.iterrows()}
            else:
                annotations_dict = {idx: f"{idx}: {image_names_dict.get(idx, 'Unknown')}" for idx in df_image_stats.index}

        name_key, color_key = self.config_plot['MAPPINGS_ALIAS_KEY'], self.config_plot['MAPPINGS_COLOR_KEY']
        marker_size, alpha = self.config_plot['SIZE'], self.config_plot['ALPHA']
        mix_groups = self.config_plot['MIX_GROUPS']
        cmap = plt.get_cmap(cmap)
        unique_groups = np.unique(label_data)
        name_color_dict = self.config_plot['COLOR_MAPPINGS'] if self.config_plot['COLOR_MAPPINGS'] is not None else {
            group: {color_key: cmap(i / (len(unique_groups) - 1)), name_key: group} for i, group in enumerate(unique_groups)}

        fig, ax = plt.subplots(figsize=figsize)
        scatter_objects = []
        legend_labels = []
        indices = []
        colors = []
        for group in unique_groups:
            group_indices = np.where(label_data == group)[0]
            if RECOLOR_BY_BRENNER and df_image_stats is not None and "Target_Sharpness_Brenner" in df_image_stats.columns:
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
            scatter_objects.append(scatter)
            legend_labels.append(name_color_dict[group][name_key])
            scatter_mappings[scatter] = indices[shuffled_indices].tolist()

        # Add legend
        ax.legend(scatter_objects, legend_labels, loc="upper right", title="Groups")

        # Enable interactive hovering with precomputed labels
        # cursor = mplcursors.cursor(scatter_objects, hover=True)

        # @cursor.connect("add")
        # def on_hover(sel):
        #     scatter_obj = sel.artist
        #     scatter_index = sel.index

        #     if scatter_obj in scatter_mappings:
        #         actual_index = original_indices[scatter_mappings[scatter_obj][scatter_index]] # Correct mapping
        #         sel.annotation.set_text(annotations_dict.get(actual_index, "Unknown"))
        #     else:
        #         sel.annotation.set_text("Unknown")

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

        ax.set_title(title if title else "UMAP Projection")
        self.format_UMAP_legend(ax, marker_size)
        self.format_UMAP_axes(ax, title)
        fig.tight_layout()
        plt.show()
        print("UMAP plot ready. Select points and then click the button below to show selected images.")
        return