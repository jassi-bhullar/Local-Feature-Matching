#-----------Author : Sanchit Jalan------------------

import cv2
import numpy as np
import sys
import getopt
import scipy
from scipy import ndimage
import math
import operator
import datetime
from numpy import zeros

def gradient(img):
    # img = img.astype('int32')
    # dx = ndimage.sobel(img, 0)  # horizontal derivative
    # dy = ndimage.sobel(img, 1)  # vertical derivative
    # grad_mag = np.hypot(dx, dy)  # magnitude
    # if dx==0:
    #     grad_orient=0
    # else :
    #     grad_orient=math.degrees(math.atan(dy/dx))
    img = np.sqrt(img)

    gx = cv2.Sobel(np.float32(img), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(img), cv2.CV_32F, 0, 1)
    grad_mag, grad_orient = cv2.cartToPolar(gx, gy)
    return grad_mag,grad_orient*57.2958

def calculate_bins(angle,mag):
    bin=zeros((8))
    for i in range(0,4):
        for j in range(0,4):
            bin[int(angle[i][j]/45)]=bin[int(angle[i][j]/45)] + mag[i][j]
    return bin

def hist_cal(angle_patch,mag_patch):
    y=np.array([0,4,8,12])
    x=np.array([0,4,8,12])
    hist_f=zeros(128)
    cnt=0
    for i in range(0,4):
        bin=zeros((8))
        for j in range(0,4):
            angle=angle_patch[y[i]:y[i]+4,x[j]:x[j]+4]
            mag=mag_patch[y[i]:y[i]+4,x[j]:x[j]+4]
            bin=calculate_bins(angle,mag)
            for k in range(0,8):
                hist_f[cnt]=bin[k]
                cnt=cnt+1
    # print "hist_f is"
    # print hist_f
    return hist_f


def get_features(image, x, y, feature_width, scales=None):
    
    height = image.shape[0]
    width = image.shape[1]
    features = zeros((x.shape[0], 128))
    impatch_or = zeros((feature_width, feature_width, x.shape[0]))
    image_filtered = zeros(image.shape)
    imgrad_mag = zeros(image.shape)
    # print "heyy " + str(imgrad_mag.shape)
    imgrad_orient = zeros(image.shape)
    imghist = zeros([x.shape[0], (feature_width / 4) ** 2 * 8])

    # Create Gaussian filter
    Sigma = 0.5
    Hshape = 3
    H=cv2.GaussianBlur(image,(Hshape,Hshape),Sigma)
    # H = fspecial(mstring('gaussian'), Hshape, Sigma)

    # Filter image with gaussian, and take gradient
    image_filtered=scipy.ndimage.convolve(image, H, mode='nearest') 
    # image_filtered = imfilter(image, H, mstring('symmetric'), mstring('same'), mstring('conv'))
    [imgrad_mag, imgrad_orient] = gradient(image_filtered)

    # Create blockproc function for histcounts  
    fun = lambda block_struct: histcounts(block_struct.data, mcat([mslice[-180:45:180]]))

    imgrad_orient1=zeros((height+16,width+16))
    imgrad_orient1[8:height+8,8:width+8]=imgrad_orient[:,:]
    imgrad_mag1=zeros((height+16,width+16))
    imgrad_mag1[8:height+8,8:width+8]=imgrad_mag[:,:]

    # print "the shape of imgrad_orient is " + str(imgrad_orient.shape)
    # outfile = open('lolololol.txt', 'w')
    # for i in range(0,imgrad_orient.shape[0]):
    #     for j in range(0,imgrad_orient.shape[1]):
    #         outfile.write(str(imgrad_orient1[i][j]) + ' ')
    #     outfile.write('2222\n')
    # outfile.close()
    
    #Create 16x16 image patches around interest points
    for i in range(0,x.shape[0]):
        yc=y[i]
        xc=x[i]
        xc=xc+8
        yc=yc+8
        impatch_or[:, :, i] = imgrad_orient1[(yc - 8):(yc + 8), (xc - 8):(xc+8)]
        hist_temp=hist_cal(impatch_or[:,:,i],imgrad_mag1[(yc - 8):(yc + 8), (xc - 8):(xc+8)])
        # imghist[i, :] = np.array([item for item in hist_temp[:]])
        imghist[i, :] = hist_temp[:]
        imghist[i, :] = imghist[i,:] / np.linalg.norm(imghist[i,:], 2)
        for j in range(0,128):
            if imghist[i][j]>0.2:
                imghist[i][j]=0.2
        # tempy = find(imghist(i, mslice[:]) > 0.2)
        # imghist(i, tempy).lvalue = 0.2
        imghist[i, :] = imghist[i,:] / np.linalg.norm(imghist[i,:], 2)
    
    features=imghist

    return features 

