import numpy as np
  
def calculate_var(image):
  
  return np.var(image)

def calculate_image_sharpness_brenner(image):
  def _brenners_gradient(image):
    # Calculate the squared difference
    shift = 2  # Typical distance between pixels for calculation
    diff = image[:, :-shift] - image[:, shift:]
    brenner = np.sum(diff ** 2)
    
    return brenner
  
  rows_brenner = _brenners_gradient(image)
  cols_brenner = _brenners_gradient(image.T)
  
  return rows_brenner + cols_brenner
