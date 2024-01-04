import numpy as np
import skimage
import cv2
from skimage.restoration import estimate_sigma
from skimage.measure import shannon_entropy
  
def calculate_snr(image):    
  # Signal is the mean value of the image
  signal = np.mean(image)
  
  # Noise is the standard deviation of the image
  noise = np.std(image)
  
  # Calculate SNR
  snr = signal / noise
  
  return snr


def calculate_image_sharpness_laplacian(image):
    """
    Higher Variance: Strong edges and, consequently, a sharper image.
    Lower Variance: The image is blurrier with fewer or softer edges.
    """
    from scipy import ndimage
    # Convert image to grayscale

    laplacian_var = ndimage.laplace(image).var()
    
    
    return laplacian_var

def calculate_var(image):
  
  return np.var(image)

def calculate_image_sharpness_brenner(image):
  """
  Low = blur
  High = sharp
  """
  def _brenners_gradient(image):
    # Calculate the squared difference
    shift = 2  # Typical distance between pixels for calculation
    diff = image[:, :-shift] - image[:, shift:]
    brenner = np.sum(diff ** 2)
    
    return brenner
  
  rows_brenner = _brenners_gradient(image)
  cols_brenner = _brenners_gradient(image.T)
  
  return rows_brenner + cols_brenner

def calculate_entropy(image):
  """
  High = Noise
  Low = Structure
  """
  entropy_value = shannon_entropy(image)
  
  return entropy_value

def calculate_sigma(image):
  ret = estimate_sigma(image)
  
  return ret

def calculate_high_freq_power(image, threshold=None):
    """
    "...if there are a low amount of high frequencies, then the image can be considered blurry." (https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Calculate the average power spectrum in the higher frequency bands
    # This can be tuned depending on the size and characteristics of the images
    if threshold is None:
        center = tuple(np.array(magnitude_spectrum.shape) // 2)
        threshold = center[0] + center[0] // 2
    high_freq_power = magnitude_spectrum[threshold:, threshold:].mean()

    return high_freq_power
  
def calc_image_blur_effect(image):
  """
  0 - no blur
  1 - maximal blur
  """
  from skimage.measure import blur_effect
  return blur_effect(image)

###############################################################################################

##########################################################################
#                                                                        #
#                           UNTESTED (WIP)                               #
#                                                                        #
########################################################################## 

def calculate_largest_uniform_area(image):
    """
    For detecting snow-like images - a snow-like image will have very few large uniform areas
    """
    from skimage.measure import label, regionprops
    
    
    gray_image = image
    # if len(image.shape) == 3:
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
    # Assuming the image has been thresholded or pre-processed appropriately
    labeled_img = label(gray_image)
    props = regionprops(labeled_img)
    largest_area = max(region.area for region in props)

    return largest_area

def calculate_correlation(image):
  """
  High = structure
  Low = Noise
  """
  gray_image = image
  # if len(image.shape) == 3:
  # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  correlation_matrix = np.corrcoef(gray_image.flatten(), np.roll(gray_image, 1).flatten())

  return correlation_matrix


def calculate_GLCM(image):
  """
  "GLCM - Gray-Level Co-occurrence Matrix for calculating texture.
  Blurry images may have lower contrast and higher homogeneity in the GLCM
  """
  gray_image = image
# if len(image.shape) == 3:
  # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  from skimage.feature import greycomatrix, greycoprops
  glcm = greycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
  contrast = greycoprops(glcm, 'contrast')
  
  return contrast


def calculate_number_of_edged(image, threshold1, threshold2):
  # Convert image to grayscale
  gray_image = image
  # if len(image.shape) == 3:
  # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray_image, threshold1, threshold2)
  edge_count = np.sum(edges > 0)

  return edge_count

def calculate_high_low_freq_ratio(image, threshold):
  gray_image = image
  # if len(image.shape) == 3:
  # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  f = np.fft.fft2(gray_image)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum = 20 * np.log(np.abs(fshift))
  high_freq_power = np.sum(magnitude_spectrum > threshold)
  total_power = np.sum(magnitude_spectrum)
  ratio = high_freq_power / total_power

  return ratio



#########################################
#                                       #
#           Haar wavelet                #
#                                       #
#########################################


def blur_detect(img, threshold):
    import pywt
    
    # Convert image to grayscale
    # Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Y = img
    
    M, N = Y.shape
    
    # Crop input image to be 3 divisible by 2
    Y = Y[0:int(M/16)*16, 0:int(N/16)*16]
    
    # Step 1, compute Haar wavelet of input image
    LL1,(LH1,HL1,HH1)= pywt.dwt2(Y, 'haar')
    # Another application of 2D haar to LL1
    LL2,(LH2,HL2,HH2)= pywt.dwt2(LL1, 'haar') 
    # Another application of 2D haar to LL2
    LL3,(LH3,HL3,HH3)= pywt.dwt2(LL2, 'haar')
    
    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8
    
    # Sliding windows size level 2
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    
    # Sliding windows size level 3
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    
    # Number of edge maps, related to sliding windows size
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    
    count = 0
    
    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0
    
    # Sliding windows limit on horizontal dimension
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        # Get the maximum value of slicing windows over edge maps 
        # in each level
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            
            x2 = x2 + sizeM2
            y2 = 0
            
            x3 = x3 + sizeM3
            y3 = 0
            
            count += 1
        
        # windows moves along horizontal dimension
        else:
                
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1
    
    # Step 3
    EdgePoint1 = Emax1 > threshold;
    EdgePoint2 = Emax2 > threshold;
    EdgePoint3 = Emax3 > threshold;
    
    # Rule 1 Edge Pojnts
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3
    
    n_edges = EdgePoint.shape[0]
    
    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint]);
    
    # Rule 3 Roof-Structure or Gstep-Structure
    
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
            
                RGstructure[i] = 1
                
    # Rule 4 Roof-Structure
    
    RSstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
            
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image 

    BlurC = np.zeros((n_edges))

    for i in range(n_edges):
    
        if RGstructure[i] == 1 or RSstructure[i] == 1:
        
            if Emax1[i] < threshold:
            
                BlurC[i] = 1                        
        
    # Step 6
    Per = np.sum(DAstructure)/np.sum(EdgePoint)
    
    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        
        BlurExtent = 100
    else:
        BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))
    
    return Per, BlurExtent

def find_images_haar(images, threshold=35, minZero=0.001):
    blur_images = []
    for i, image in enumerate(images):
        print(f"{i}/{len(images)}")
        per, blurext = blur_detect(image, threshold)
        if per < minZero:
            blur_images.append(image)
            print(f"{i}: per: {per}, blurext: {blurext}")
    
    blur_images = np.stack(blur_images)
    return blur_images


def improve_brightness(img, contrast_factor, brightness_factor):
    brightness_adjusted = contrast_factor * (img - img.min()) + brightness_factor
    return brightness_adjusted