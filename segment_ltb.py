import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure, morphology
from scipy.ndimage.morphology import binary_erosion, binary_closing
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation

def search(l, string):
    has = [item for item in l if item.find(string) > -1]
    mis = [item for item in l if item.find(string) < 0]
    return has, mis

def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return (~canvas | input_mask.astype(np.uint8)) 

def load_scan(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    read = reader.Execute()
    scan = sitk.GetArrayFromImage(read)
    return scan

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def get_mask(image, structure = 'lung'):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    if  structure == 'tissue':
        print('Doing a tissues based segmentation mask...')
        alpha = -600
        binary_image = np.array(image < alpha, dtype=np.int8)+1
        fill_structures=False
    elif structure == 'bone':
        print('Doing a bone based segmentation mask...')
        alpha = 250
        binary_image = np.array(image < alpha, dtype=np.int8)+1
        fill_structures=False
    else:
        print('Doing a lung based segmentation mask...')
        alpha = -320
        binary_image = np.array(image > alpha, dtype=np.int8)+1
        fill_structures=True
        
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

def normalize(x):
    x = (x + x.min() + 1.)
    x = ((x-x.min())/(x.max()-x.min()))
    return x

def get_segmentation(path, part=6, display=True):
    scan = load_scan(path)
    scan[scan < -1024] = -1024
    scan[scan > 3071] = 3071
    if scan.max() < 3071:
        scan[:,0,0] = 3071
    n = (scan.shape[0]//12) * part
    img = scan[n]
    img = normalize(img)
    min, max = img.min(), img.max()
    
    # Get the lung segmentation
    lung_mask = get_mask(scan, structure = 'lung')[n]
    lung_mask = lung_mask.astype(np.bool)
    lung_mask = morphology.remove_small_objects(lung_mask, min_size=32)
    #lung_mask = img * mask[n]
    
    #Get the bone mask
    mask = get_mask(scan, structure='bone')
    #mask = binary_erosion(mask, iterations=0)
    mask = binary_closing(mask, iterations=3)
    mask = binary_dilation(mask, iterations=2)
    mask = binary_erosion(mask, iterations=1)
    mask = mask[n]
    #mask3 = binary_fill_holes(mask3)
    mask = fill_hole(mask)
    bool_mask = mask.astype(np.bool)
    bone_mask = morphology.remove_small_objects(bool_mask, min_size=128)
    
    # Get the tissue mask    
    mask = get_mask(scan, structure='tissue')
    #mask = binary_dilation(mask, iterations=1)
    #mask = binary_erosion(mask[n], iterations=8)
    mask = mask[n]
    tissue_mask = mask *np.logical_not(bone_mask)
    tissue_mask = tissue_mask *np.logical_not(lung_mask)
    tissue_mask = tissue_mask.astype(np.bool)
    tissue_mask = morphology.binary_opening(tissue_mask)
    tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=512)
    
    if display:
        plt.figure(figsize=(18,12))
        plt.subplot(2,3,1)
        plt.imshow(lung_mask)        
        plt.subplot(2,3,2)
        plt.imshow(bone_mask)        
        plt.subplot(2,3,3)
        plt.imshow(tissue_mask)
        plt.subplot(2,3,4)
        plt.imshow(img*lung_mask, vmin=min, vmax=max)        
        plt.subplot(2,3,5)
        plt.imshow(img*bone_mask, vmin=min, vmax=max)        
        plt.subplot(2,3,6)
        plt.imshow(img*tissue_mask, vmin=min, vmax=max)
        plt.show()
        
    return lung_mask, bone_mask, tissue_mask
