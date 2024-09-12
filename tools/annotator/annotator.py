import os
import csv
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

class ImageAnnotator:
    def __init__(self, folder_path, savepath='annotations.csv'):
        self.folder_path = folder_path
        self.savepath = savepath
        self.files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
        self.current_file_index = 0
        self.current_image_index = 0
        self.images = None
        self.annotation_data = {}
        self.save_annotations() # Reset the existing file
        self.setup_gui()
        self.load_next_file()

    def _current_key(self):
        file_path = os.path.join(self.folder_path, self.files[self.current_file_index])
        return f"{file_path}_{self.current_image_index}"

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Image Annotator")
        
        self.path_label = tk.Label(self.root, text="", wraplength=400)  # Label for displaying the image path
        self.path_label.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.dead_button = tk.Button(self.root, text="Dead", command=lambda: self.record_response('dead'), bg='dark red', fg='white')
        self.dead_button.pack(side=tk.LEFT)

        self.alive_button = tk.Button(self.root, text="Alive", command=lambda: self.record_response('alive'), bg='light green', fg='black')
        self.alive_button.pack(side=tk.RIGHT)
        
        # Bottom Frame for the Back Button
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        container_frame = tk.Frame(bottom_frame)
        container_frame.pack(expand=True)
        
        # Back Button with left arrow in the bottom frame
        self.back_button = tk.Button(container_frame, text="←", command=self.go_back)
        self.back_button.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Forward Button with right arrow in the bottom frame
        self.forward_button = tk.Button(container_frame, text="→", command=self.go_forward)
        self.forward_button.pack(side=tk.LEFT, padx=5, pady=10)

    def load_next_file(self):
        if self.current_file_index < len(self.files):
            file_path = os.path.join(self.folder_path, self.files[self.current_file_index])
            self.images = np.load(file_path)
            self.current_image_index = 0
            self.load_image()
        else:
            self.root.destroy()

    def load_image(self):
        if self.current_image_index < len(self.images):
            img = self.images[self.current_image_index, :, :, 1]  # Selecting the second channel
            
             # Normalize the image
            vmin, vmax = 0, 1  # Replace with your chosen values
            img = np.clip(img, vmin, vmax)
            img = (img - vmin) / (vmax - vmin) * 255
            img = img.astype(np.uint8)

            # Apply colormap
            cmap = plt.get_cmap('gray')  # Replace 'gray' with your chosen colormap
            img_colored = cmap(img)  # This returns RGBA values
            img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit RGB
            
            
            # Convert to PIL image and resize
            img = Image.fromarray(img)
            base_width = 300  # Set the desired width
            w_percent = (base_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((base_width, h_size), Image.ANTIALIAS)
            
            img = ImageTk.PhotoImage(img)
            
            image_path = os.path.join(os.sep.join(self.folder_path.split(os.sep)[-4:]), self.files[self.current_file_index])
            self.path_label.config(text=f"Current Image: {image_path} - Image Index: {self.current_image_index}")

            self.image_label.configure(image=img)
            self.image_label.image = img
        else:
            self.current_file_index += 1
            self.load_next_file()
            
    def go_back(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()
        elif self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_next_file()
            self.current_image_index = len(self.images) - 1
            self.load_image()
            
    def go_forward(self):
        key = self._current_key()
        if key in self.annotation_data:
            self.current_image_index += 1
            self.load_image()
            
    def record_response(self, status):
        key = self._current_key()
        self.annotation_data[key] = status
            
        self.save_annotations()

        self.current_image_index += 1
        self.load_image()
        
    def save_annotations(self):
        with open(self.savepath, 'w', newline='') as file:
            writer = csv.writer(file)
            for key, status in self.annotation_data.items():
                writer.writerow([key, status])

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    images_folder = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/batch6_16bit_no_downsample/WT/Untreated/G3BP1"
    savepath = "/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/tools/annotator/annotations.csv"
    
    annotator = ImageAnnotator(images_folder, savepath)
    try:
        annotator.start()
    finally:
        annotator.save_annotations()
