from __future__ import print_function, division
import numpy as np
import numpy
import os
import time
import sys
import cv2
import scipy
import skimage
from PSNR import psnrVDSR ,PSNRTorch ,psnrNITRE,psnrSVLAB,im2double
#from keras.utils.visualize_util import plot
#from keras.utils.vis_utils import plot_model as plot
import models
import img_utils
from scipy.misc import imread, imresize, imsave
import skimage
import skimage.io as io
import skimage.transform
from skimage.measure import compare_ssim as ssim_ski
from skimage.measure import compare_psnr as psnr_ski
from skimage.color import rgb2ycbcr 
from models import psnr2 ,psnr3, psnr ,PSNRLossTest

import tensorflow as tf

def setimgrgb2ycbcr(im):
    im=rgb2ycbcr(im)
    #im=rgb2ycbcrCV(im)
    #im=im.astype(np.uint8)
    im=im[ :, :, 0]
    return im         


def rgb2ycbcrLocal(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def rgb2ycbcrTORCH(im):
    im=im2double(im)
    y = 16 + (65.481 * im[:,:,0]) + (128.553 * im[:,:,1]) + (24.966 * im[:,:,2])
    return y.astype(np.float32)
    #function util:rgb2ycbcr(img)
    #local y = 16 + (65.481 * img[1]) + (128.553 * img[2]) + (24.966 * img[3])
    #return y / 255

def rgb2ycbcrCV(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def ycbcr2rgb(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*255.0-16)/(235-16) #to [0, 1]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*255.0-16)/(240-16) #to [0, 1]
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb




def crop_border(imgage,bordr):
    init_width, init_height = imgage.shape[0], imgage.shape[1]
    croped=imgage[bordr : init_width-bordr , bordr : init_height-bordr]
    return croped
def im2double1(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    #return im.astype(np.float) / info.max
    return im.astype(np.float) / 255.0

if __name__ == "__main__":
    path = r""
    
    scorlist=[]
    scorssimy=[]
    scorski=[]
    scale = 2

    """
    Plot the models
    """
    suffix='vds'
    suffix='OUTminiBL'
    suffix='OUTBL'
    suffix='scaled'
    scale_factor=1
    path_dir="/home/www/Image-Super-Resolution/val_images/set14nitre/"
    path_dir="/home/www/imgsuper/val_images/set5nitre/"
    #path_dir="/home/www/imgsuper/val_images/set5png/"
    #path_dir="/home/www/imgsuper/val_images/nitre/"


    #path_dir="/home/www/Image-Super-Resolution/val_images/nitrevd/"
    for file in os.listdir(path_dir):    
        pathfile=path_dir + file
        #print(pathfile)

        path = os.path.splitext(pathfile)
        if suffix not in pathfile:            
            #print(path)
            fileOrig=path[0] + path[1]
            print(fileOrig)
            filenameNitre = path[0] + "_" + suffix + "(%dx)" % (scale_factor) + path[1]

            #filenameNitre = path[0] + "x4_" + suffix + "(%dx)" % (scale_factor) + path[1]
            filenameNitreNPY = path[0] + "_" + suffix + "(%dx)" % (scale_factor) + '.npy'
            filenameNitreSavediff = path[0] + "_DIFF" + suffix + "(%dx)" % (scale_factor) + path[1]
            #filenameNitre = path[0] + "_vdsr.png"
            print (filenameNitre)
            im1=imread(fileOrig, mode='RGB')
            #im1sim = tf.decode_png(fileOrig)
            #im1 = cv2.imread(fileOrig)
            #b,g,r = cv2.split(im1) 
            #im1 = cv2.merge([r,g,b])
            im1ski=im1

            #im1 = skimage.img_as_float(skimage.io.imread(fileOrig)).astype(np.float32)
            
            im2=imread(filenameNitre, mode='RGB')
            #im2sim = tf.decode_png(filenameNitre)
            #im2 = cv2.imread(filenameNitre)
            #b,g,r = cv2.split(im2) 
            #im2 = cv2.merge([r,g,b])
            im2ski=im2


            img_width, img_height = im1.shape[0], im1.shape[1]
            img_widtho=int(img_width/4)
            img_heighto=int(img_height/4)
            im1or = imresize(im1, (img_widtho, img_heighto),interp='bicubic')
            fileOrigsave=path[0]+'ORIG' + path[1]
            #imsave(fileOrigsave, im1or)
            im1orbig = imresize(im1or, (img_width, img_height),interp='nearest')
            fileOrigsave=path[0]+'ORIBIG' + path[1]
            #imsave(fileOrigsave, im1orbig)


            #im2=imread(filenameNitre)
            #im2npy=np.load(filenameNitreNPY);
            #print('MIN MAX NPY')
            #minVal=np.amin(im2npy)            
            #print(minVal)
            #NAXVal=np.amax(im2npy)
            #print(NAXVal)

            
            #factor=255.0/NAXVal
            #im2=im2npy*factor
            #im2=im2npy

            #im2 = im2.astype(np.float32) * 255.
            #im2npy = np.where(im2npy > 256., im2npy+1.*(255.-im2npy) , im2npy)            
            #im2 = np.rint(im2npy).astype('uint32')
            #im2 = np.rint(im2npy).astype(np.uint8)
            #im1=im1.astype('uint32')
            #im2=im2.astype('uint32')
            #im2 = np.clip(im2, 0, 255)
            #im2 = np.rint(im2 )
            #im2 = np.clip(im2, 0, 1)
            #im2 = im2.astype(np.float) * 255.
            #im1=im1.astype(np.float)
            #im1=im1/255
            #im1=im2double(im1)
            
            #im2 = skimage.img_as_float(skimage.io.imread(filenameNitre)).astype(np.float32)

            img_width, img_height = im1.shape[0], im1.shape[1] 
            #im2 = imresize(im2, (img_width, img_height),interp='bicubic')
            cropval=10

            im1=crop_border(im1,cropval)
            im2=crop_border(im2,cropval)
            im1ski=crop_border(im1ski,cropval)
            im2ski=crop_border(im2ski,cropval)
            #im1=im2double(im1)
            #im2=im2double(im2)


            #im1=setimgrgb2ycbcr(im1)
            #im2=setimgrgb2ycbcr(im2)
            
            #im1=rgb2ycbcrTORCH(im1)
            #im2=rgb2ycbcrTORCH(im2)

            im1=setimgrgb2ycbcr(im1)
            im2=setimgrgb2ycbcr(im2)

            #imy=ycbcr2rgb(im2)
            #imsave(fileOrigsave, imy)
            

            #im1=im1.astype(np.uint8)
            #im2=im2.astype(np.uint8) 
            #print('---')
            minVal=np.amin(im1)
            minVal2=np.amin(im2)
            #print(minVal)
            maxVal=np.amax(im1)
            maxVal2=np.amax(im2)
            #print(maxVal)
            #print(minVal2)
            #print(maxVal2)


            #im1=im2double(im1)
            #im2=im2double(im2)
            diffadd=im1-im2
            ski=1
            #diffadd=im1+diff
            #imsave(filenameNitreSavediff, diffadd)


            #scor=psnr_ski(im1  , im2 ,255.)
            #scor=psnrVDSR(im1  , im2,1)
            scor=PSNRTorch(im2  , im1,0)
            scor=psnrNITRE(im2  , im1 ,0)
            

            #scor=psnrSVLAB(im1  , im2 )
            #ski = tf.image.ssim(im1sim, im2sim, max_val=255)
            ski_y=ssim_ski(im1  , im2, data_range=255)

            ski=ssim_ski(im1ski  , im2ski, data_range=255, multichannel=True)
            #ski=ssim_ski(im1ski  , im2ski, multichannel=True)
            
            #print(ssim_ski(im1  , im2, multichannel=True))
            print("SCORs psnr_ski" )
            print (scor)
            scorlist.append(scor)
            scorski.append(ski)
            scorssimy.append(ski_y)
            #print("---")
            print("SCORs SSIM Y" )
            print (ski_y)



    meanPNSR=sum(scorlist) / float(len(scorlist))
    meanSKI=sum(scorski) / float(len(scorski))
    scorssimy=sum(scorssimy) / float(len(scorssimy))
    print("-------------------------------------------------------------------------------")
    print("---")
    print("---")
    print("SCOR MEAN psnr")
    print(meanPNSR)
    print("---")
    print("---")
    print("---")
    print("SCOR MEAN SSIM SKI")
    print(meanSKI)
    print("---")
    print("SCOR MEAN SSIM SKI y")
    print(scorssimy)

    #imb=imread('/home/www/Image-Super-Resolution/val_images/set5nitre/butterfly_GT_DIFFOUTBL(1x).bmp', mode='RGB')
    #print(imb)

#N = numel(E); % Assume the original signal is at peak (|F|=1)
#res = 10*log10( N / sum(E(:).^2) );




   
