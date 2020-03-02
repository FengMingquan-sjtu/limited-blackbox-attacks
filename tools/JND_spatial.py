# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:50:40 2019
JND in pixel space for grayscale images.
@author: y84145850
"""
import cv2
import numpy as np
from scipy import signal
import cv2
import numpy as np
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt

def avg_background_lum(img):
    """
    calculate the average of background luminance.
    """
    mask = np.array([[1, 1, 1, 1, 1],
                     [1, 2, 2, 2, 1],
                     [1, 2, 0, 2, 1],
                     [1, 2, 2, 2, 1],
                     [1, 1, 1, 1, 1]])
    output = signal.convolve2d(img, np.rot90(mask, 2), mode='same', boundary='fill', fillvalue=0)
    return output/32.


def edge_canny(img, edge_thre_min, edge_thre_max):
    if img.dtype != 'uint8':
        img = img.astype(np.uint8)
    smoothedInput = cv2.GaussianBlur(img, (7, 7), 2)
    edges = cv2.Canny(smoothedInput, edge_thre_min, edge_thre_max)
    return edges

def gaussian2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def max_weight_grad(img, JND_type='Yang'):
    """
    maximal weighted average of gradients.
    """    
    G1 = np.array([[0, 0, 0, 0, 0],
                   [1, 3, 8, 3, 1],
                   [0, 0, 0, 0, 0],
                   [-1, -3, -8, -3, -1],
                   [0, 0, 0, 0, 0]])
    
    G2 = np.array([[0, 0, 1, 0, 0],
                   [0, 8, 3, 0, 0],
                   [1, 3, 0, -3, -1],
                   [0, 0, -3, -8, 0],
                   [0, 0, -1, 0, 0]])
    
    G3 = np.array([[0, 0, 1, 0, 0],
                   [0, 0, 3, 8, 0],
                   [-1, -3, 0, 3, 1],
                   [0, -8, -3, 0, 0],
                   [0, 0, -1, 0, 0]])
    
    G4 = np.array([[0, 1, 0, -1, 0],
                   [0, 3, 0, -3, 0],
                   [0, 8, 0, -8, 0],
                   [0, 3, 0, -3, 0],
                   [0, 1, 0, -1, 0]])
    
    hei, wid = img.shape
    grad = np.zeros(shape=(hei, wid, 4))
    
    grad[:, :, 0] = signal.convolve2d(img, np.rot90(G1, 2), mode='same', boundary='fill', fillvalue=0)/16.
    grad[:, :, 1] = signal.convolve2d(img, np.rot90(G2, 2), mode='same', boundary='fill', fillvalue=0)/16.
    grad[:, :, 2] = signal.convolve2d(img, np.rot90(G3, 2), mode='same', boundary='fill', fillvalue=0)/16.
    grad[:, :, 3] = signal.convolve2d(img, np.rot90(G4, 2), mode='same', boundary='fill', fillvalue=0)/16.
    
    Gm = np.amax(np.abs(grad), axis=2)
    
    edge_thre_min = 50  #25, 50 respectively for min, max
    edge_thre_max = 127  #TODO threshold set as 0.5 in referred matlab code(min_val=0.4*max_val)
    img_edge = edge_canny(img, edge_thre_min, edge_thre_max)
    img_edge = img_edge / 255.  #convert to float format
    
    # define kernel
    kernel = np.ones((11, 11), dtype=float)  #strel('disk',6)
    
    img_edge = cv2.dilate(img_edge, kernel, iterations=1)
    img_supedge = 1.0 - 0.95*img_edge  #TODO check max value of img_edge in matlab code
    
    gaussian_kernel = gaussian2D(shape=(7, 7), sigma=0.8)
    
    img_supedge = signal.convolve2d(img_supedge, np.rot90(gaussian_kernel, 2), mode='same', boundary='fill', fillvalue=0)
    
    ## type to choose
    if JND_type == 'Chou':
        output = Gm
    else:
        output = Gm *img_supedge 
        
    return output 
    

def JND_pixel(img, JND_type='Yang'):
    """ JND in spatial pixel domain.
    img: grayscale uint8 image with intensity range [0,255].
    return: jnd. float values within [0,255], generally within 18.
    """
    # constants setup
    c0 = 17
    c1 = 0.3
    lamda = 1 / 2.
    gamma = 3/128.

    img = img.astype(float)
    hei, wid = img.shape

    jnd_lum = np.zeros(shape=(hei, wid))
    jnd_texture = np.zeros(shape=(hei, wid))
    jnd_tmp = np.zeros(shape=(hei, wid, 2))

    f = np.amax([1, round(np.amin([hei, wid])/256.)])
    if f > 1:
        lpf = np.ones(shape=(f, f))
        lpf = lpf / np.sum(lpf.ravel())
        img = cv2.filter2D(img, -1, lpf)
        img = img[::f, ::f]

    # average luminance, weighted average gradients
    avg_lum = avg_background_lum(img)
    avg_max_grad = max_weight_grad(img, JND_type='Yang')
    # plt.imshow(avg_max_grad, cmap='gray')
    # plt.show()

    # jnd luminance component
    for row in range(hei):
        for col in range(wid):
            if avg_lum[row, col] <= 127:
                jnd_lum[row, col] = c0 * (1 - np.sqrt(avg_lum[row, col]/127.)) + 3
            else:
                jnd_lum[row, col] = gamma * (avg_lum[row, col]-127) + 3


    # jnd texture component
    alpha = 0.0001 * avg_lum + 0.115
    beta = lamda - 0.01 * avg_lum
    jnd_texture = avg_max_grad * alpha + beta


    # final JND
    jnd_tmp[:, :, 0] = jnd_lum
    jnd_tmp[:, :, 1] = jnd_texture

    jnd = np.sum(jnd_tmp, axis=2) - c1 * np.amin(jnd_tmp, axis=2)
    return jnd

def run(img_name):
    #input image name
    #output 3-channel JND
    img = Image.open(img_name).convert('L', (0.2989, 0.5870, 0.1140, 0))
    img = np.array(img)
    JND = JND_pixel(img)
    return JND


if __name__ == '__main__':
    img_name="ILSVRC2012_val_00000019.png"
    #img_name = 'data/imagenetVal/images/000b7d55b6184b08.png'    #panda image
    # img_name = 'data/imagenetVal/images/0be391239ccba0f2.png'  #sunset image
    img = Image.open(img_name).convert('L', (0.2989, 0.5870, 0.1140, 0))
    img = np.array(img )
    # plt.imshow(img , cmap='gray')
    # plt.show()
    JND_spatial = JND_pixel(img)
    x=JND_spatial
    x_3d=np.stack((x,x,x),axis=2)
    print(x_3d.shape)
    plt.imshow(x_3d)
    #plt.imshow(JND_spatial, cmap='gray')

    plt.show()

    #plt.hist(JND_spatial.ravel())
    #plt.show()

    print(np.mean(np.abs(JND_spatial.ravel())))