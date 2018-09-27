from __future__ import print_function, division
import tensorflow as tf

from keras.models import Model
from keras.layers import Concatenate, Add,Subtract, Average, Input, Dense,Dropout, Flatten, BatchNormalization, Activation, LeakyReLU ,ELU,Lambda
from keras.layers.convolutional import Conv2D,Convolution2D, MaxPooling2D, UpSampling2D,ZeroPadding2D, Convolution2DTranspose ,SeparableConv2D,Conv2DTranspose
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l1,l2 ,L1L2
activity_l1 = l1
activity_l2 = l2
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from advanced import HistoryCheckpoint, SubPixelUpscaling ,SubpixelConv2D
from keras_subpixel import Subpixel
import img_utils

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import os ,gc
import time
import sys
import math
import scipy
import skimage
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure
from skimage.transform import rescale
from skimage.color import rgb2ycbcr , ycbcr2rgb ,gray2rgb ,rgb2gray
#from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
#                                 denoise_wavelet, estimate_sigma)


train_path = img_utils.output_path
validation_path = img_utils.validation_output_path
path_X = img_utils.output_path + "X/"
path_Y = img_utils.output_path + "y/"

print(K.image_dim_ordering())

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return K.mean(y_pred)
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))

def PSNRLossTest(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    #return K.mean(y_pred)
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))

def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))

def psnr2(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr3(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10((PIXEL_MAX ** 2)/ math.sqrt(mse))


class BaseSuperResolutionModel(object):

    def __init__(self, model_name, scale_factor):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None # type: Model
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.weight_path = None

        self.type_scale_type = "norm" # Default = "norm" = 1. / 255
        self.type_requires_divisible_shape = False
        self.type_true_upscaling = False

        self.evaluation_func = None
        self.uses_learning_phase = False

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=2) -> Model:
        """
        Subclass dependent implementation.
        """
        if self.type_requires_divisible_shape:
            assert height * img_utils._image_scale_multiplier % 4 == 0, "Height of the image must be divisible by 4"
            assert width * img_utils._image_scale_multiplier % 4 == 0, "Width of the image must be divisible by 4"

        if K.image_dim_ordering() == "th":
            shape = (channels, width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier)
        else:
            shape = (width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier, channels)

        print("shape")
        print(shape)
        print(img_utils._image_scale_multiplier)
        init = Input(shape=shape)

        return init

    def fit(self, batch_size=2, nb_epochs=100, save_history=True, history_fn="Model History.txt") -> Model:
        """
        Standard method to train any of the models.
        """

        samples_per_epoch = img_utils.image_count()
        #samples_per_epoch =10000
        val_count = img_utils.val_image_count()
        if self.model == None: self.create_model(batch_size=batch_size)

        callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=False,
                                                   mode='max', save_weights_only=True ,period=1)]
        if save_history: callback_list.append(HistoryCheckpoint(history_fn))

        print("Training model : %s" % (self.__class__.__name__))
        trainset=img_utils.image_generator(train_path, scale_factor=self.scale_factor,
                                                           small_train_images=self.type_true_upscaling,
                                                           batch_size=batch_size)
        self.model.fit_generator(trainset, steps_per_epoch=samples_per_epoch,
                                 epochs=nb_epochs, callbacks=callback_list,
                                 validation_data=img_utils.image_generator(validation_path,
                                                                           scale_factor=self.scale_factor,
                                                                           small_train_images=self.type_true_upscaling,
                                                                           batch_size=batch_size),
                                 validation_steps=val_count)

        return self.model

    def evaluate(self, validation_dir):
        if self.type_requires_divisible_shape:
            _evaluate_denoise(self, validation_dir)
        else:
            _evaluate(self, validation_dir)

    def upVideo(self, imgObj, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=8, mode="patch", verbose=False):
        print("SHAAAAAAAAAAAAAAAAA")
        print (imgObj.shape)

        img_width, img_height = imgObj.shape[0], imgObj.shape[1]
        images = np.expand_dims(imgObj, axis=0)

        # Transpose and Process images
        img_conv = images.astype(np.float32) / 255.

        model = self.create_model(img_height,img_width ,load_weights=True)
        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=128, verbose=verbose)        
        result = result[0, :, :, :] # Access the 3 Dimensional image vector
        #print(result.shape)
        result = np.clip(result, 0, 255).astype('uint8')
        return result

    def upscaleStepPatch(self, img_path, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=256,scalemulti=4,step_patch=64, mode="patch", verbose=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """
        import os
        from scipy.misc import imread, imresize, imsave
        print("IMAGEPATH")
        print(img_path)

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + path[1]
        filenameNumpy = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + '.npy'
        filenameM = path[0] + "_A" + suffix + "(%dx)" % (self.scale_factor) + path[1]
        filenameN = path[0] + "_N" + suffix + "(%dx)" % (self.scale_factor) + path[1]

        # Read image
        scale_factor = int(self.scale_factor)
        scale_factor = 1
        true_img = imread(img_path, mode='RGB')
        #true_img = gaussian_filter(true_img, sigma=0.5)







        #true_img =cv2.imread(img_path,1)
        init_height , init_width = true_img.shape[0], true_img.shape[1]
        orig_height , orig_width = true_img.shape[0], true_img.shape[1]

        bordersize=patch_size
        imgageRebuildini = np.zeros( (init_height+bordersize,init_width+bordersize,3))
        imgageRebuildini[0 : init_height , 0 : init_width]=true_img

        true_img=imgageRebuildini
        init_height , init_width = true_img.shape[0], true_img.shape[1]



        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_height * scale_factor, init_width * scale_factor))
        print (init_width)
        #return 
        #img_height, img_width = 0, 0

        #if mode == "patch" and self.type_true_upscaling:
            # Overriding mode for True Upscaling models
        #    mode = 'fast'
        #    print("Patch mode does not work with True Upscaling models yet. Defaulting to mode='fast'")
        #return
        if mode == 'patch':
            #arr_patch=np.zeros((2,)
            # Create patches
            step_patch=64
            #if init_width % scalemulti != 0 or init_height % scalemulti != 0 :
            if init_width % step_patch != 0 or init_height % step_patch != 0 :
                new_w=int((init_width/step_patch)+1)*step_patch
                new_h=int((init_height/step_patch)+1)*step_patch
                #Crop image to by multiple
                new_img = np.zeros((new_h,new_w,3))
                new_img[0 :init_height,0 : init_width] =true_img
                true_img=new_img

                #Reinitialise w ,h
                init_height , init_width = true_img.shape[0], true_img.shape[1]
                print (init_width)
                print("new SHAPEEEEE")
                print("new SHAPEEEEE : ", true_img.shape)
            



            if self.type_requires_divisible_shape:
                if patch_size % 4 != 0:
                    print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
                    patch_size = 8
            #step_patch=16
            images ,counterrebuildtople= img_utils.extract_patches_Step(true_img,  (patch_size, patch_size),step_patch)
            #images = img_utils.subimage_build_patch_global(true_img, patch_size,patch_size,5000)
            #images = img_utils.extract_patches_2dlocal(true_img,imagesfull, (patch_size, patch_size) ,step=step_patch)
            print (images.shape)
            
            

            nb_images = images.shape[0]

            patch_shape = (int(init_width * scale_factor), int(init_height * scale_factor), 3)
            


            #out_im=img_utils.rebuild_from_patches_Step(true_img,images, (patch_size, patch_size),counterrebuildtople, scale_factor,step_patch)

            #imsave('/home/www/imgsuper/val_images/testX4.png', out_im)

            x_width=int(images.shape[1]*scalemulti)
            x_height=int(images.shape[2]*scalemulti)
            patch_shape = (nb_images,int(patch_size *scalemulti), int(patch_size *scalemulti), 3)
            arr_patch_min=np.zeros(patch_shape).astype(np.float32)
            #for i in range( nb_images ):
            #    patchel=images[i]
                #w_img=img_utils.imresize(patchel, (x_width, x_height), interp='bicubic')
               # w_img=rescale(patchel,4)
                #arr_patch_min[i] = w_img
                #imsave('/home/www/imgsuper/val_images/batchmini/testX4_'+str(i)+'.png', patchel)
                #imsave('/home/www/imgsuper/val_images/batch/testX4_'+str(i)+'.png', w_img)
            #imagesset=arr_patch_min


            #out_im=img_utils.rebuild_from_patches_Step(true_img,imagesset, (patch_size, patch_size),counterrebuildtople, scalemulti,step_patch)

            #imsave('/home/www/imgsuper/val_images/testXX4.png', out_im)


            #images=arr_patch_min
            img_height ,img_width = images.shape[1], images.shape[2]
            print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_height, img_width))
            #return
        else:
            # Use full image for super resolution
            img_height ,img_width = self.__match_autoencoder_size(img_height, img_width, init_height,
                                                                  init_width, scale_factor)

            #if (img_height*4) >2100 or (img_width*4)>2100:
            #    sys.exit("Error message")
            #print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))
        # Save intermediate bilinear scaled image is needed for comparison.
        #save_intermediate=True
        #return
        intermediate_img = None
        if save_intermediate:
            if verbose: print("Saving intermediate image.")
            fn = path[0] + "_intermediate_" + path[1]
            intermediate_img = imresize(true_img, (init_width * scale_factor, init_height * scale_factor))
            #imsave(fn, intermediate_img)
            imsave(fn, images[0, :, :, :])
            #print(images[0, :, :, :].shape)

        # Transpose and Process images
        if K.image_dim_ordering() == "th":
            img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        else:
            img_conv = images.astype(np.float32) / 255.

        model = self.create_model(img_height, img_width, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=1, verbose=verbose)


        if verbose: print("De-processing images.")

         # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.
        for j in range(5):
            im=result[j]
            #imsave(filenameM, im)


        del model
        K.clear_session()
        gc.collect()

            
        #print(result.shape)
        # Output shape is (original_width * scale, original_height * scale, nb_channels)
        if mode == 'patch':
            scale_factor=1
            out_shape = (int( init_height* scale_factor), int(init_width * scale_factor), 3)
            out_shape_patch = (int( init_height), int(init_width ),3)
            #out_shape = (101,60,3)
            #out_shape = (60,101,3)
            #out_shape = (init_height+8,init_width+8,3)
            #out_shape = (init_width+patch_size,init_height+patch_size,3)
            print(out_shape)
            print("------")
            print(out_shape)
            #(93, 52, 3)

            print(result.shape)
            print(true_img.shape)
            #result = img_utils.combine_patches(result, out_shape, scale_factor)
            #result = img_utils.combine_patches(result, out_shape_patch, scale_factor)
            #result = img_utils.reconstruct_from_patches_2dlocal(imagesfull,result, out_shape_patch ,step=step_patch) 
            result=img_utils.rebuild_from_patches_Step(true_img,result, (patch_size, patch_size),counterrebuildtople, scalemulti,step_patch) 
            #result = img_utils.subimage_combine_patches_global(true_img , result, patch_size* 2, patch_size*2, 2)
        else:
            result = result[0, :, :, :] # Access the 3 Dimensional image vector
        #print(result.shape)
        if verbose: print("Shape out net.")
        print(result.shape)
        #np.save(filenameNumpy, result)
        #result = np.rint(result)
        result = np.clip(result, 0, 255).astype('uint8')
        #result =result.astype('uint8')
        #result=result[0 :orig_height,0 : orig_width]
        if verbose: print("Shape initial.")
        print(result.shape)
        if verbose: print("\nCompleted De-processing image.")
        #result=scipy.misc.imfilter(result,ftype='edge_enhance_more')
        #result=scipy.misc.imfilter(result,ftype='edge_enhance')
        #result=denoise_wavelet(result, multichannel=True, convert2ycbcr=True)

        #p2, p98 = np.percentile(result, (2,98))
        #result= exposure.rescale_intensity(result, in_range=(p2, p98))
        #result=scipy.misc.imfilter(result,ftype='sharpen')
        #result= scipy.ndimage.rotate(result, -1*angle, reshape=False)
        if return_image:
            # Return the image without saving. Useful for testing images.
            return result


          
        if verbose: print("Saving image.")
        outresult= result[0:orig_height*scalemulti , 0:orig_width*scalemulti ]  
        #result = imresize(result, (out_width, out_height),interp='bicubic')
        #imsave(filename, result)
        imsave(filename, outresult)



    def upscalePatch(self, img_path, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=32,scalemulti=4, mode="patch", verbose=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """
        import os
        from scipy.misc import imread, imresize, imsave

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + path[1]
        filenameNumpy = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + '.npy'
        filenameM = path[0] + "_A" + suffix + "(%dx)" % (self.scale_factor) + path[1]
        filenameN = path[0] + "_N" + suffix + "(%dx)" % (self.scale_factor) + path[1]

        # Read image
        scale_factor = int(self.scale_factor)
        scale_factor = 1
        true_img = imread(img_path, mode='RGB')
        #true_img =cv2.imread(img_path,1)
        init_height , init_width = true_img.shape[0], true_img.shape[1]
        orig_height , orig_width = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_height * scale_factor, init_width * scale_factor))
        print (init_width)
        #return 
        #img_height, img_width = 0, 0

        #if mode == "patch" and self.type_true_upscaling:
            # Overriding mode for True Upscaling models
        #    mode = 'fast'
        #    print("Patch mode does not work with True Upscaling models yet. Defaulting to mode='fast'")
        #return
        if mode == 'patch':
            #arr_patch=np.zeros((2,)
            # Create patches
            step_patch=4
            #if init_width % scalemulti != 0 or init_height % scalemulti != 0 :
            if init_width % step_patch != 0 or init_height % step_patch != 0 :
                new_w=int((init_width/step_patch)+1)*step_patch
                new_h=int((init_height/step_patch)+1)*step_patch
                #Crop image to by multiple
                new_img = np.zeros((new_h,new_w,3))
                new_img[0 :init_height,0 : init_width] =true_img
                true_img=new_img

                #Reinitialise w ,h
                init_height , init_width = true_img.shape[0], true_img.shape[1]
                print (init_width)
                print("new SHAPEEEEE")
                print("new SHAPEEEEE : ", true_img.shape)
            



            if self.type_requires_divisible_shape:
                if patch_size % 4 != 0:
                    print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
                    patch_size = 8
            #step_patch=16
            imagesfull = img_utils.make_patchesOrig(true_img, scale_factor, patch_size, verbose)
            #images = img_utils.subimage_build_patch_global(true_img, patch_size,patch_size,5000)
            images = img_utils.extract_patches_2dlocal(true_img,imagesfull, (patch_size, patch_size) ,step=step_patch)
            print (images.shape)
            
            #return

            nb_images = images.shape[0]

            patch_shape = (int(init_width * scale_factor), int(init_height * scale_factor), 3)


            x_width=int(images.shape[1]/scalemulti)
            x_height=int(images.shape[2]/scalemulti)
            patch_shape = (nb_images,int(patch_size /scalemulti), int(patch_size /scalemulti), 3)
            arr_patch_min=np.zeros(patch_shape).astype(np.float32)
            for i in range( nb_images ):
                patchel=images[i]
                arr_patch_min[i] = img_utils.imresize(patchel, (x_width, x_height), interp='bicubic')


            images=arr_patch_min
            img_height ,img_width = images.shape[1], images.shape[2]
            print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_height, img_width))
        else:
            # Use full image for super resolution
            img_height ,img_width = self.__match_autoencoder_size(img_height, img_width, init_height,
                                                                  init_width, scale_factor)

            #if (img_height*4) >2100 or (img_width*4)>2100:
            #    sys.exit("Error message")
            #print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))
        # Save intermediate bilinear scaled image is needed for comparison.
        #save_intermediate=True
        #return
        intermediate_img = None
        if save_intermediate:
            if verbose: print("Saving intermediate image.")
            fn = path[0] + "_intermediate_" + path[1]
            intermediate_img = imresize(true_img, (init_width * scale_factor, init_height * scale_factor))
            #imsave(fn, intermediate_img)
            imsave(fn, images[0, :, :, :])
            #print(images[0, :, :, :].shape)

        # Transpose and Process images
        if K.image_dim_ordering() == "th":
            img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        else:
            img_conv = images.astype(np.float32) / 255.

        model = self.create_model(img_height, img_width, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=2, verbose=verbose)

        if verbose: print("De-processing images.")

         # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.
        for j in range(5):
            im=result[j]
            #imsave(filenameM, im)
            
        #print(result.shape)
        # Output shape is (original_width * scale, original_height * scale, nb_channels)
        if mode == 'patch':
            scale_factor=1
            out_shape = (int( init_height* scale_factor), int(init_width * scale_factor), 3)
            out_shape_patch = (int( init_height), int(init_width ),3)
            #out_shape = (101,60,3)
            #out_shape = (60,101,3)
            #out_shape = (init_height+8,init_width+8,3)
            #out_shape = (init_width+patch_size,init_height+patch_size,3)
            print(out_shape)
            print("------")
            print(out_shape)
            #(93, 52, 3)

            print(result.shape)
            print(true_img.shape)
            #result = img_utils.combine_patches(result, out_shape, scale_factor)
            #result = img_utils.combine_patches(result, out_shape_patch, scale_factor)
            result = img_utils.reconstruct_from_patches_2dlocal(imagesfull,result, out_shape_patch ,step=step_patch)  
            #result = img_utils.subimage_combine_patches_global(true_img , result, patch_size* 2, patch_size*2, 2)
        else:
            result = result[0, :, :, :] # Access the 3 Dimensional image vector
        #print(result.shape)
        if verbose: print("Shape out net.")
        print(result.shape)
        #np.save(filenameNumpy, result)
        #result = np.rint(result)
        result = np.clip(result, 0, 255).astype('uint8')
        #result =result.astype('uint8')
        result=result[0 :orig_height,0 : orig_width]
        if verbose: print("Shape initial.")
        print(result.shape)
        if verbose: print("\nCompleted De-processing image.")
        #result=scipy.misc.imfilter(result,ftype='edge_enhance_more')
        #result=scipy.misc.imfilter(result,ftype='edge_enhance')
        #result=denoise_wavelet(result, multichannel=True, convert2ycbcr=True)

        #p2, p98 = np.percentile(result, (2,98))
        #result= exposure.rescale_intensity(result, in_range=(p2, p98))
        #result=scipy.misc.imfilter(result,ftype='sharpen')
        #result= scipy.ndimage.rotate(result, -1*angle, reshape=False)
        if return_image:
            # Return the image without saving. Useful for testing images.
            return result


          
        if verbose: print("Saving image.")
        #result = imresize(result, (out_width, out_height),interp='bicubic')
        imsave(filename, result)

    def upscale(self, img_path, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=32, mode="patch", verbose=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """
        import os
        from scipy.misc import imread, imresize, imsave
        flip_img=False

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + path[1]
        filenameNumpy = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + '.npy'
        filenameM = path[0] + "_A" + suffix + "(%dx)" % (self.scale_factor) + path[1]
        filenameN = path[0] + "_N" + suffix + "(%dx)" % (self.scale_factor) + path[1]

        # Read image
        scale_factor = int(self.scale_factor)
        true_img = imread(img_path, mode='RGB')
        #true_img =cv2.imread(img_path,1)
        init_width, init_height = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_height * scale_factor, init_width * scale_factor))

        img_height, img_width = 0, 0

        if mode == "patch" and self.type_true_upscaling:
            # Overriding mode for True Upscaling models
            mode = 'fast'
            print("Patch mode does not work with True Upscaling models yet. Defaulting to mode='fast'")

        if mode == 'patch':
            # Create patches
            if self.type_requires_divisible_shape:
                if patch_size % 4 != 0:
                    print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
                    patch_size = 8

            true_img = imresize(true_img, (init_width*4, init_height*4),interp='bicubic')
            print(patch_size)
            imsave(filenameM, true_img)
            images = img_utils.make_patches(true_img, scale_factor, patch_size, verbose)


            nb_images = images.shape[0]
            print(true_img.shape)
            print(nb_images)

            #return



            x_width=int(images.shape[1]/4)
            x_height=int(images.shape[2]/4)
            patch_shape = (nb_images,int(patch_size /4), int(patch_size /4), 3)
            arr_patch_min=np.zeros(patch_shape).astype(np.float32)
            for i in range( nb_images ):
                patchel=images[i]
                arr_patch_min[i] = img_utils.imresize(patchel, (x_width, x_height), interp='bicubic')


            images=arr_patch_min
            #images = img_utils.subimage_build_patch_global(true_img, patch_size,patch_size,5000)

            nb_images = images.shape[0]
            img_width, img_height = images.shape[1], images.shape[2]
            print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_height, img_width))
        else:
            # Use full image for super resolution
            img_height ,img_width = self.__match_autoencoder_size(img_height, img_width, init_height,
                                                                  init_width, scale_factor)
            #img_width=32
            #img_height=32

            out_width=img_width
            out_height=img_height

            #true_img = gaussian_filter(true_img, sigma=0.4)
            angle=5
            flip_img=False
            #true_img = scipy.ndimage.rotate(true_img, angle, reshape=False)
            #true_img=rgb2ycbcr(true_img)


            scaletmp=1

            img_width=int(img_width/scaletmp)
            img_height=int(img_height/scaletmp)   
            #true_img = imresize(true_img, (img_widtha, img_heighta),interp='bicubic') 


           
            #print("SIZE W H")
            #print(img_width)
            #print(img_height)
            #make alb negru
            #make alb negru


            #true_img = skimage.color.rgb2gray(true_img)
            #true_img = skimage.color.gray2rgb(true_img)
            #true_img = scipy.ndimage.rotate(true_img, angle, reshape=True)
            if flip_img:
                true_img = np.fliplr(true_img)
            #img_width=int(img_width/4)
            #img_height=int(img_height/4)
            #true_img = imresize(true_img, (int(img_width*2), int(img_height*2) ),interp='bicubic')

            #true_img = imresize(true_img, (int(img_width/2), int(img_height/2) ),interp='bicubic')

            #true_img = imresize(true_img, (int(img_width/2), int(img_height/2) ),interp='nearest')
            #images = imresize(true_img, (img_width, img_height) ,interp='nearest')
            #images = true_img
            #true_img=denoise_wavelet(true_img, multichannel=True, convert2ycbcr=True)
            #p2, p98 = np.percentile(true_img, (2, 98))
            #true_img= exposure.rescale_intensity(true_img, in_range=(p2, p98))
            #true_img = gaussian_filter(true_img, sigma=0.3)
            #true_img=exposure.adjust_log(true_img, 0.7)
            #images = imresize(true_img, (img_width, img_height))
            #true_img =img_utils.SetGama(true_img,0.2)

            #images = imresize(true_img, (img_width, img_height),interp='nearest')
            #imsave(filenameM, true_img)
            #true_img=scipy.misc.imfilter(true_img,ftype='sharpen')

            
            #images = imresize(true_img, (img_width, img_height),interp='bilinear')
            images = imresize(true_img, (img_width, img_height),interp='bicubic')
            #imsave(filenameM, images)
            #images = gaussian_filter(images, sigma=0.2)

            #true_img = imresize(true_img, (16, 16))
            #images = imresize(true_img, (img_width, img_height),interp='nearest')
            #images=np.fliplr(images)
            #true_img = gaussian_filter(true_img, sigma=0.5)
            #images=exposure.adjust_log(images, 1)
            imagesBic = imresize(images, (init_width, init_height),interp='bicubic')

            #images=rgb2ycbcr(images)
            #images=gray2rgb(rgb2gray(images))

            imsave(filenameM, images)
            #exit()

            images = np.expand_dims(images, axis=0)
            #if (img_height*4) >2100 or (img_width*4)>2100:
            #    sys.exit("Error message")
            #print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))
        # Save intermediate bilinear scaled image is needed for comparison.
        #save_intermediate=True
        intermediate_img = None
        if save_intermediate:
            if verbose: print("Saving intermediate image.")
            fn = path[0] + "_intermediate_" + path[1]
            intermediate_img = imresize(true_img, (init_width * scale_factor, init_height * scale_factor))
            #imsave(fn, intermediate_img)
            imsave(fn, images[0, :, :, :])
            #print(images[0, :, :, :].shape)

        # Transpose and Process images
        if K.image_dim_ordering() == "th":
            img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        else:
            img_conv = images.astype(np.float32) / 255.

        model = self.create_model(img_height, img_width, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=10, verbose=verbose)

        if verbose: print("De-processing images.")

         # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.
        #print(result.shape)
        # Output shape is (original_width * scale, original_height * scale, nb_channels)


        del model
        K.clear_session()
        gc.collect()


        if mode == 'patch':
            scale_factor=4
            out_shape = (int(init_width * scale_factor), int(init_height * scale_factor), 3)
            #out_shape = (101,60,3)
            #out_shape = (60,101,3)
            #out_shape = (init_height+8,init_width+8,3)
            #out_shape = (init_width+patch_size,init_height+patch_size,3)
            print(out_shape)
            print("------")
            print(out_shape)
            #(93, 52, 3)

            print(result.shape)
            print(true_img.shape)
            #result = img_utils.combine_patches(result, out_shape, scale_factor)
            result = img_utils.combine_patches(result, out_shape, scale_factor)
            #result = img_utils.subimage_combine_patches_global(true_img , result, patch_size* 2, patch_size*2, 2)
        else:
            result = result[0, :, :, :] # Access the 3 Dimensional image vector
        #print(result.shape)
        if verbose: print("Saving numpy.")
        #np.save(filenameNumpy, result)
        #result = np.rint(result)
        result = np.clip(result, 0, 255).astype('uint8')
        #result = np.clip(result, 0, 235).astype('uint8')
        #result =result.astype('uint8')

        if verbose: print("\nCompleted De-processing image.")
        #result=scipy.misc.imfilter(result,ftype='edge_enhance_more')
        #result=scipy.misc.imfilter(result,ftype='edge_enhance')
        #result=denoise_wavelet(result, multichannel=True, convert2ycbcr=True)

        p2, p98 = np.percentile(result, (2,98))
        #result= exposure.rescale_intensity(result, in_range=(p2, p98))
        #result=scipy.misc.imfilter(result,ftype='sharpen')
        #result= scipy.ndimage.rotate(result, -1*angle, reshape=False)
        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

        if flip_img:
            result = np.fliplr(result)  

          
        if verbose: print("Saving image.")
        #result=ycbcr2rgb(result)
        #result = imresize(result, (out_width, out_height),interp='bicubic')
        imsave(filename, result)
        #cv2.imwrite(filename, result)
        #f_resizenearest = imresize(result, (int(init_width), int(init_height)), interp='nearest')
        #f_resizenearest = imresize(result, (int(img_width), int(img_height)))
        #imsave(filenameN, f_resizenearest)


    def __match_autoencoder_size(self, img_height, img_width, init_height, init_width, scale_factor):
        if self.type_requires_divisible_shape:
            if not self.type_true_upscaling:
                # AE model but not true upsampling
                if ((init_height * scale_factor) % 4 != 0) or ((init_width * scale_factor) % 4 != 0) or \
                        (init_height % 2 != 0) or (init_width % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_height = ((init_height * scale_factor) // 4) * 4
                    img_width = ((init_width * scale_factor) // 4) * 4

                else:
                    # No change required
                    img_height, img_width = init_height * scale_factor, init_width * scale_factor
            else:
                # AE model and true upsampling
                if ((init_height) % 4 != 0) or ((init_width) % 4 != 0) or \
                        (init_height % 2 != 0) or (init_width % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_height = ((init_height) // 4) * 4
                    img_width = ((init_width) // 4) * 4

                else:
                    # No change required
                    img_height, img_width = init_height, init_width
        else:
            # Not AE but true upsampling
            if self.type_true_upscaling:
                img_height, img_width = init_height, init_width
            else:
                # Not AE and not true upsampling
                img_height, img_width = init_height * scale_factor, init_width * scale_factor

        return img_height, img_width




def resizeX2diff(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [2 * int(s) for s in prev_shape[1:3]]
    #return tf.image.resize_nearest_neighbor(my_input, size) 
    #return tf.image.resize_bicubic(my_input, size)
    return tf.image.resize_bilinear(my_input, size)


def resizeX2diff_outputshape(my_input_shape):
    shape = list(my_input_shape)
    print (shape)    
    size = [2 * int(s) for s in my_input_shape[1:3]]
    shape[1]=size[0] 
    shape[2]=size[1] 
    print (shape) 
    return tuple(shape)

def resizeXX4(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [4 * int(s) for s in prev_shape[1:3]]
    #return tf.image.resize_nearest_neighbor(my_input, size) 
    #return tf.image.resize_bicubic(my_input, size)
    return tf.image.resize_bilinear(my_input, size)


def resizeXX4_outputshape(my_input_shape):
    shape = list(my_input_shape)
    print (shape)    
    size = [4 * int(s) for s in my_input_shape[1:3]]
    shape[1]=size[0] 
    shape[2]=size[1] 
    print (shape) 
    return tuple(shape)

def resize2bil(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [2 * int(s) for s in prev_shape[1:3]]
    #return tf.image.resize_nearest_neighbor(my_input, size)
    #return tf.image.resize_bicubic(my_input, size) 
    return tf.image.resize_bilinear(my_input, size)

    #return tf.image.resize_bilinear(my_input, size)    
def resize2bic(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [2 * int(s) for s in prev_shape[1:3]]
    #return tf.image.resize_nearest_neighbor(my_input, size)
    return tf.image.resize_bicubic(my_input, size) 
    #return tf.image.resize_bilinear(my_input, size)    

def resizeResScale(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.999,  my_input)    
def resizeResScaleVar(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(1.007,  my_input)  


def resizeRsDi(my_input): # resizes input tensor wrt. ref_tensor 1.01  0.991 0.995 1.003
    return tf.scalar_mul(1.000, my_input)  


def resizeRsD(my_input): # resizes input tensor wrt. ref_tensor 0.0997 
    return tf.scalar_mul(0.1000 ,  my_input)     
def resizeRes01RsD(my_input): # resizes input tensor wrt. ref_tensor  0.0996
    return tf.scalar_mul(0.1,  my_input) 

def resizeX1(my_input): # resizes input tensor wrt. ref_tensor 1.01  0.991 0.995 1.003
    return tf.scalar_mul(1.00, my_input)     

def resizeX4bic(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [4 * int(s) for s in prev_shape[1:3]]
    #return tf.image.resize_nearest_neighbor(my_input, size)
    return tf.image.resize_bicubic(my_input, size) 
    #return tf.image.resize_bilinear(my_input, size)
def resizeBlockLight09(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.9,  my_input)
def resizeBlock01(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.1,  my_input)

def resizeBlockLight(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.1,  my_input)    

def resizeBlockLight01(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.1,  my_input)

def resizeBlockLight001(my_input): # resizes input tensor wrt. ref_tensor 
    #return tf.scalar_mul(0.0997,  my_input)  
    return tf.scalar_mul(0.1,  my_input)  

class Difvdsr4(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(Difvdsr4, self).__init__("Image ScaleGen", scale_factor)



        
        #self.weight_path = "weights_ScaleDiff/scaleNNDiff %dX.h5" % (self.scale_factor)
        self.weight_path ="weights_Difvdsr2scale/weights-{epoch:02d}-{val_acc:.2f}.h5"
        
        

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=24 ,width=24, channels=3, load_weights=False, batch_size=1):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        num_kernels = [ 128,128]
        #um_kernels = [ 192,192]
        #num_kernels = [ 96,96]
        numk=256
        #num_kernels = [512, 256,128]


        init = super(Difvdsr4, self).create_model(height, width, channels, load_weights, batch_size)
        

        x = Convolution2D(numk,(1,1) ,activation='relu',  padding='same',trainable=True,  name='level1')(init)
        #xInp=x
        #for j in range(48):
        #    x = self._residual_block(x, numk,train=True)
        #x=Add()([x,xInp])   
        for j in range(6):
            #x = self._residual_block_light(x, numk,train=True,dcale=True)  
            x = self._residual_block_light0(x, numk,train=True)

        x=Lambda(resize2bil ,output_shape=resizeX2_outputshape)(x) 
        xInp=x    
        for j in range(20):
            #x = self._residual_block_light(x, numk,train=True,dcale=True)  
            x = self._residual_block_light(x, numk,train=True)
        x=Add()([x,xInp])    

        x=Lambda(resize2bil ,output_shape=resizeX2_outputshape)(x)     
        for j in range(6):
            #x = self._residual_block_light(x, numk,train=True,dcale=True)  
            x = self._residual_block_light(x, numk,train=True) 


        out = Conv2D( 3, (3, 3),padding='same', activation='relu')(x) 
        model = Model(init, out)


        lr=1e-4
        #lr=0.00005
        #lr=0.000025
        #lr=0.00001
        #lr=1e-5

        #lr=0.0005
        #lr=0.003

        adam = optimizers.Adam(lr,0.9)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        #load_weights=True
        wpath="weights_Difvdsr2scale/weights05-17-0.94.h5"
        #.05
        #wpath="weights_Difvdsr2scale/weights05-03-0.94.h5"
        #.025   weights025-05-0.94.h5 psnr 32.2671    weights025-22-0.94.h5 / 32.28
        wpath="weights_Difvdsr2scale/weights025-22-0.94.h5"
 
        #.001  /0.1/weights025-08-0.94.h5 /32.303   /0.1/weights025-18-0.94.h5/32.31458
        wpath="weights_Difvdsr2scale/0.1/weights025-18-0.94.h5"
        #wpath="weights_Difvdsr2scale/0.1/weights025-08-0.94.h5"

        #wpath="weights_Scale234/basescale.h5"
        if load_weights: model.load_weights(wpath)
        #if load_weights: model.load_weights(self.weight_path)
        #print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=2 , nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(Difvdsr4, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True,train=True):

        ini=xR 
        xRDiffini =xR
        
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR) 
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR)

        


        xRdiffconvBlock=Subtract()([xR, xRDiffini])     
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xRdiffconvBlock)        
        #xRdiffconv = LeakyReLU(0.2)(xRdiffconv)
        xRdiffconv = Activation('relu')(xRdiffconv)
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xRdiffconv)
        #xRdiffconv=Add()([xRdiffconvBlock,xRdiffconv,xR])
        xRdiffconv=Add()([xRdiffconv,xR])

        xRdiffconv=Lambda(resizeBlock01 ,output_shape=resizeRes_outputshape)(xRdiffconv)
        if add:
            #xR=Multiply()([0.1,xR])
            
            xR=Add()([xRdiffconv,ini])
        return xR

    def _residual_block_light(self, xR, nr_filter ,add=True,train=True,dcale=True):

        ini=xR 
        
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR) 
        xR = Activation('relu')(xR)
        #xR = LeakyReLU(0.005)(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR) 

        if dcale:
            xR=Lambda(resizeBlockLight ,output_shape=resizeRes_outputshape)(xR)
        if add:
            #xR=Multiply()([0.1,xR])
            
            xR=Add()([xR,ini])
        return xR    

    def _residual_block_light0(self, xR, nr_filter ,add=True,train=True,dcale=True):

        ini=xR 
        
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR) 
        #xR = Activation('relu')(xR)
        xR = LeakyReLU(0.001)(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR) 

        if dcale:
            xR=Lambda(resizeBlockLight01 ,output_shape=resizeRes_outputshape)(xR)
        if add:
            #xR=Multiply()([0.1,xR])
            
            xR=Add()([xR,ini])
        return xR  



class DifvdsrDouble(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DifvdsrDouble, self).__init__("Image ScaleGen", scale_factor)



        
        #self.weight_path = "weights_ScaleDiff/scaleNNDiff %dX.h5" % (self.scale_factor)
        self.weight_path ="weights_Double/weights025-{epoch:02d}-{val_acc:.2f}.h5"
        
        

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    #def create_model(self, height=24 ,width=24, channels=3, load_weights=False, batch_size=128):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        num_kernels = [ 128,128]
        #um_kernels = [ 192,192]
        #num_kernels = [ 96,96]
        numk=128
        #num_kernels = [512, 256,128]


        init = super(DifvdsrDouble, self).create_model(height, width, channels, load_weights, batch_size)
        

        x = Convolution2D(numk,(1,1) ,activation='relu',  padding='same',trainable=True,  name='level1')(init)
        #xInp=x
        #for j in range(48):
        #    x = self._residual_block(x, numk,train=True)
        #x=Add()([x,xInp])  
        for j in range(16):
            #x = self._residual_block_light(x, numk,train=True,dcale=True)  
            x = self._residual_block_light53(x, numk,train=True)

        #x=Lambda(resize2bil ,output_shape=resizeX2_outputshape)(x) 
        xInp=x    
        for j in range(6):
            #x = self._residual_block_light(x, numk,train=True,dcale=True)  
            x = self._residual_block_light(x, numk,train=True)
        #x=Add()([x,xInp])    

        x=Lambda(resizeX4bil ,output_shape=resizeX4_outputshape)(x)     
        for j in range(2):
            #x = self._residual_block_light(x, numk,train=True,dcale=True)  
            x = self._residual_block_light53(x, numk,train=True) 


        out = Conv2D( 3, (3, 3),padding='same', activation='relu')(x) 
        model = Model(init, out)
        

        lr=1e-4
        #lr=0.00005
        #lr=0.000025
        #lr=0.00001
        #lr=1e-5

        #lr=0.0005
        #lr=0.003

        adam = optimizers.Adam(lr,0.9)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        #load_weights=True
        #lr=0.00005

        wpath="weights_Double/weights025-17-0.93.h5"
        if load_weights: model.load_weights(wpath)
        #if load_weights: model.load_weights(self.weight_path)
        #print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=10 , nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(DifvdsrDouble, self).fit(batch_size, nb_epochs, save_history, history_fn)




    def _residual_block_light(self, xR, nr_filter ,add=True,train=True,dcale=True):

        ini=xR 
        
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR) 
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR) 

        if dcale:
            xR=Lambda(resizeBlockLight01 ,output_shape=resizeRes_outputshape)(xR)
        if add:
            #xR=Multiply()([0.1,xR])
            
            xR=Add()([xR,ini])
        return xR    


    def _residual_block_light53(self, xR, nr_filter ,add=True,train=True,dcale=True):

        ini=xR 
        ini=Lambda(resizeBlockLight09 ,output_shape=resizeRes_outputshape)(ini)
        
        xR1 = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR) 
        xR1 = Activation('relu')(xR1)
        xR1 = Conv2D(nr_filter, (5, 5), padding='same',trainable=train)(xR1) 

        xR2 = Conv2D(nr_filter, (5, 5), padding='same',trainable=train)(xR) 
        xR2 = Activation('relu')(xR2)
        xR2 = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR2) 

        xR=Add()([xR1,xR2])

        if dcale:
            xR=Lambda(resizeBlockLight01 ,output_shape=resizeRes_outputshape)(xR)
            
        if add:
            #xR=Multiply()([0.1,xR])
            
            xR=Add()([xR,ini])
        return xR 



class Difvdsr(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(Difvdsr, self).__init__("Image ScaleGen", scale_factor)



        
        #self.weight_path = "weights_ScaleDiff/scaleNNDiff %dX.h5" % (self.scale_factor)
        self.weight_path ="weights_Difvdsr/weights-{epoch:02d}-{val_acc:.2f}.h5"
        
        

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=64, width=64, channels=3, load_weights=False, batch_size=128):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        num_kernels = [ 128,128]
        num_kernels = [ 192,192]
        #num_kernels = [ 96,96]
        numk=192
        #num_kernels = [512, 256,128]


        init = super(Difvdsr, self).create_model(height, width, channels, load_weights, batch_size)
        

        x = Convolution2D(192, (3, 3) ,activation='relu',  padding='same',trainable=False,  name='level1')(init)
        for j in range(32):
            x = self._residual_block(x, numk,train=True) 

        out = Conv2D( 3, (3, 3),padding='same', activation='relu')(x) 
        model = Model(init, out)
        

        lr=1e-4
        #lr=0.00005
        #lr=0.000025
        #lr=0.00001
        #lr=1e-5
        #lr=0.003

        adam = optimizers.Adam(lr,0.9)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        load_weights=True
        wpath="weights_Difvdsr/weights-23-0.96.h5"
        #wpath="weights_Scale234/basescale.h5"
        if load_weights: model.load_weights(wpath)
        #if load_weights: model.load_weights(self.weight_path)
        #print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=8, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(Difvdsr, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True,train=True):

        ini=xR 
        xRDiffini =xR
        
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR) 
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xR)


        xRdiffconvBlock=Subtract()([xR, xRDiffini])     
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xRdiffconvBlock)        
        xRdiffconv = LeakyReLU(0.2)(xRdiffconv)
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same',trainable=train)(xRdiffconv)
        xRdiffconv=Add()([xRdiffconvBlock,xRdiffconv,xR])

        xRdiffconv=Lambda(resizeRes01 ,output_shape=resizeRes_outputshape)(xRdiffconv)
        if add:
            #xR=Multiply()([0.1,xR])
            
            xR=Add()([xRdiffconv,ini])
        return xR


class Dif234(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(Dif234, self).__init__("Image ScaleGen", scale_factor)



        
        #self.weight_path = "weights_ScaleDiff/scaleNNDiff %dX.h5" % (self.scale_factor)
        self.weight_path ="weights_Scale234/weights-{epoch:02d}-{val_acc:.2f}.h5"
        
        

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=48, width=48, channels=3, load_weights=False, batch_size=128):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        num_kernels = [ 128,128]
        #num_kernels = [ 96,96]
        numk=192
        #num_kernels = [512, 256,128]


        init = super(Dif234, self).create_model(height, width, channels, load_weights, batch_size)
        

        x = Convolution2D(128, (3, 3) ,activation='relu',  padding='same',  name='level1')(init)
        
        
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)

        #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
            idx = i if i < len(num_kernels) else -1
            #if i>0:
            #    x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(5):
                x = self._residual_block(x, num_kernels[idx], scale=False,Leaky=0.2)

            for j in range(27):
                x = self._residual_block(x, num_kernels[idx] ,Leaky=0.2) 

            #x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
            x=Lambda(resizeX2bil ,output_shape=resizeX2_outputshape)(x)
            #x=Lambda(resizeXX4 ,output_shape=resizeXX4_outputshape)(x)


        #x = LeakyReLU(0.2)(x)
        #x =  Conv2D(256, (1,1), padding='same' ,activation='relu' )(x)
        x =  Conv2D(256, (1,1), padding='same'  )(x)
        #xIni =x
        #x = BatchNormalization()(x)
        #x =  Conv2D(256, (3,3), padding='same' ,activation='relu' )(x)
        #for j in range(2):
        #    x = self._residual_block(x, 256, scale=True)
        for j in range(4):
            x = self._residual_block(x, 256)    
        #x=Add()([x,xIni])
        
        #x = self._residual_block(x, 256) 



        out = Conv2D( 3, (3, 3),padding='same', activation='relu')(x)   
        #out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)
        

        lr=1e-4
        #lr=0.00005
        #lr=0.000025
        #lr=0.00001
        #lr=1e-5
        #lr=0.003

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        #model.compile(optimizer=adam, loss='mse')
        #load_weights=True
        wpath="weights_Scale234/weights-13-0.96.h5"
        if load_weights: model.load_weights(wpath)
        #f load_weights: model.load_weights(self.weight_path)
        #print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=8, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(Dif234, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True,scale=False,Leaky=0.2):

        ini=xR 
        xRDiffini =xR
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)


        xRdiffconvBlock=Subtract()([xR, xRDiffini])     
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same')(xRdiffconvBlock)        
        xRdiffconv = LeakyReLU(Leaky)(xRdiffconv)
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same')(xRdiffconv)
        
        if scale:
            xRdiffconv=Lambda(resizeRsDi ,output_shape=resizeRes_outputshape)(xRdiffconv)


        xRdiffconv=Add()([xRdiffconvBlock,xRdiffconv,xR])
        xRdiffconv=Lambda(resizeRes01 ,output_shape=resizeRes_outputshape)(xRdiffconv)
        #if scale:
        #    xRdiffconv=Lambda(resizeRsD ,output_shape=resizeRes_outputshape)(xRdiffconv)
        #else: 
        #    xRdiffconv=Lambda(resizeRes01RsD ,output_shape=resizeRes_outputshape)(xRdiffconv)

        if add:
            xR=Add()([xRdiffconv,ini])

        return xR




class SDiff(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(SDiff, self).__init__("Image ScaleGen", scale_factor)



        #self.weight_path = "weights_nobp_32_X4/scale WeightsRezidual3X3 %dX.h5" % (self.scale_factor)
        self.weight_path = "weights_ScaleDiff/scaleNNDiff %dX.h5" % (self.scale_factor)

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=48, width=48, channels=3, load_weights=False, batch_size=128):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        num_kernels = [ 128,128]
        #num_kernels = [ 96,96]
        numk=128
        #num_kernels = [512, 256,128]


        init = super(SDiff, self).create_model(height, width, channels, load_weights, batch_size)
        

        x = Convolution2D(128, (3, 3) ,activation='relu',  padding='same',  name='level1')(init)
        
        
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)

        #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
            idx = i if i < len(num_kernels) else -1
            #if i>0:
            #    x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)
            for j in range(10):
                x = self._residual_block(x, num_kernels[idx], scale=True ) 

            for j in range(14):
                x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX2d ,output_shape=resizeX2d_outputshape)(x)
            #x=Lambda(resizeX2diff ,output_shape=resizeX2d_outputshape)(x)
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)


        #x = LeakyReLU(0.2)(x)
        x =  Conv2D(256, (1,1), padding='same' ,activation='relu' )(x)
        #xIni =x
        x = BatchNormalization()(x)
        x =  Conv2D(256, (3,3), padding='same' ,activation='relu' )(x)
        #x=Add()([x,xIni])
        
        #x = self._residual_block(x, 256) 



        out = Conv2D( 3, (3, 3),padding='same', activation='relu')(x)   
        #out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)
        

        lr=1e-4
        #lr=1e-5
        #lr=0.003

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        #model.compile(optimizer=adam, loss='mse')
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        #print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=12, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(SDiff, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,scale=False ,add=True):

        

        #xR = BatchNormalization()(xR)
        ini=xR 

        #xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        xRDiffini =xR
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)

        #xRT=Add()([xR,ini])
        #xR = BatchNormalization()(xR)
       

        #xRdiff = BatchNormalization()(xRdiff)
        #xRdiffconv = LeakyReLU(0.2)(xRdiffconv)
        #xR = Activation('relu')(xR)  
        xRdiffconvBlock=Subtract()([xR, xRDiffini])
        #if scale:
        #    xRdiffconvBlock=Lambda(resizeRsD ,output_shape=resizeRes_outputshape)(xRdiffconvBlock)
        #xRdiffconvBlock = BatchNormalization()(xRdiffconvBlock)      
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same')(xRdiffconvBlock) 
        if scale:  
            #0.18     
            xRdiffconv = LeakyReLU(0.2)(xRdiffconv)
        else:
             #0.21
            xRdiffconv = LeakyReLU(0.2)(xRdiffconv)
            
        #xRdiffconv = Activation('relu')(xRdiffconv) 
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same')(xRdiffconv)
        #xRdiffconv=Lambda(resizeRsDi ,output_shape=resizeRes_outputshape)(xRdiffconv)
        if scale:
            xRdiffconv=Lambda(resizeRsDi ,output_shape=resizeRes_outputshape)(xRdiffconv)
            #xRdiffconvBlock=Lambda(resizeRsDi ,output_shape=resizeRes_outputshape)(xRdiffconvBlock)
            #xR=Lambda(resizeRsDi ,output_shape=resizeRes_outputshape)(xR)


        xRdiffconv=Add()([xRdiffconvBlock,xRdiffconv,xR])
        #xRdiffconv=Add()([xRdiffconvBlock,xRdiffconv])

        if scale:
            xRdiffconv=Lambda(resizeRsD ,output_shape=resizeRes_outputshape)(xRdiffconv)
        else:
            xRdiffconv=Lambda(resizeRes01RsD ,output_shape=resizeRes_outputshape)(xRdiffconv)    

        #xR = BatchNormalization()(xR)
        
        if add:
            #xR=Multiply()([0.1,xR])
            
            xR=Add()([xRdiffconv,ini])


        return xR  



class ScaleDiff(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleDiff, self).__init__("Image ScaleGen", scale_factor)



        #self.weight_path = "weights_nobp_32_X4/scale WeightsRezidual3X3 %dX.h5" % (self.scale_factor)
        self.weight_path = "weights_ScaleDiff/scaleScaleDiff %dX.h5" % (self.scale_factor)

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=48, width=48, channels=3, load_weights=False, batch_size=128):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        num_kernels = [ 128,128]
        #num_kernels = [ 96,96]
        numk=128
        #num_kernels = [512, 256,128]


        init = super(ScaleDiff, self).create_model(height, width, channels, load_weights, batch_size)
        

        x = Convolution2D(128, (3, 3) ,activation='relu',  padding='same',  name='level1')(init)
        
        xIni =x
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)

        #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
            idx = i if i < len(num_kernels) else -1
            #if i>0:
            #    x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)
            
            for j in range(10):
                x = self._residual_block_var(x, num_kernels[idx])

            for j in range(14):
                x = self._residual_block(x, num_kernels[idx])                 

            #for j in range(24):
            #    x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
            #x=Lambda(resizeXX4 ,output_shape=resizeXX4_outputshape)(x)


        #x = LeakyReLU(0.2)(x)
        x =  Conv2D(256, (1,1), padding='same' ,activation='relu' )(x)
        x = BatchNormalization()(x)
        x =  Conv2D(256, (3,3), padding='same' ,activation='relu' )(x)
        
        #x = self._residual_block(x, 256) 



        out = Conv2D( 3, (3, 3),padding='same', activation='relu')(x)   
        #out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)
        lr=1e-4
        #lr=0.045

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse')
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        #print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=12, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleDiff, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        #xR = BatchNormalization()(xR)
        ini=xR 

        xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        xRDiffini =xR
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)

        #xRT=Add()([xR,ini])
        #xR = BatchNormalization()(xR)
       

        #xRdiff = BatchNormalization()(xRdiff)
        #xRdiffconv = LeakyReLU(0.2)(xRdiffconv)
        #xR = Activation('relu')(xR)  
        xRdiffconvBlock=Subtract()([xR, xRDiffini])      
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same')(xRdiffconvBlock)        
        xRdiffconv = LeakyReLU(0.1)(xRdiffconv) 
        #0.102
        #xRdiffconv = Activation('relu')(xRdiffconv) 
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same')(xRdiffconv)
        #xRdiffconv=Lambda(resizeResScale ,output_shape=resizeRes_outputshape)(xRdiffconv)
        xRdiffconv=Add()([xRdiffconvBlock,xRdiffconv])
        #xR=Lambda(resizeRes05 ,output_shape=resizeRes_outputshape)(xR)

        #xR = BatchNormalization()(xR)
        
        if add:
            #xR=Multiply()([0.1,xR])
            xR=Add()([xRdiffconv,xR,ini])


        return xR   

    def _residual_block_var(self, xR, nr_filter ,add=True):

        

        #xR = BatchNormalization()(xR)
        ini=xR 

        xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        xRDiffini =xR
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)

        #xRT=Add()([xR,ini])
        #xR = BatchNormalization()(xR)
       

        #xRdiff = BatchNormalization()(xRdiff)
        #xRdiffconv = LeakyReLU(0.2)(xRdiffconv)
        #xR = Activation('relu')(xR)  
        xRdiffconvBlock=Subtract()([xR, xRDiffini])      
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same')(xRdiffconvBlock)        
        xRdiffconv = LeakyReLU(0.1)(xRdiffconv)
        #0.098
        #xRdiffconv = Activation('relu')(xRdiffconv) 
        xRdiffconv = Conv2D(nr_filter, (3, 3), padding='same')(xRdiffconv)
        #xRdiffconv=Lambda(resizeResScaleVar ,output_shape=resizeRes_outputshape)(xRdiffconv)
        xRdiffconv=Add()([xRdiffconvBlock,xRdiffconv])
        #xR=Lambda(resizeRes05 ,output_shape=resizeRes_outputshape)(xR)

        #xR = BatchNormalization()(xR)
        
        if add:
            #xR=Multiply()([0.1,xR])
            xR=Add()([xRdiffconv,xR,ini])


        return xR 

class ScaleGenX4(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGenX4, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights_big_32_X4/scale WeightsRezidual %dX.h5" % (self.scale_factor)

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=48, width=48, channels=3, load_weights=False, batch_size=128):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        #num_kernels = [512, 256,128]


        init = super(ScaleGenX4, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(256, (3, 3) ,activation='relu',  padding='same',  name='level1')(init)
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)

        #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
            idx = i if i < len(num_kernels) else -1
            #if i>0:
            #    x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(14):
                x = self._residual_block(x, num_kernels[idx]) 


            #for j in range(7):
            #    x = self._residual_block(x, num_kernels[idx]) 


            #for j in range(6):
            #    x = self._residualLeaky_block(x, num_kernels[idx])  

            #for j in range(8):
            #    x = self._residual_block(x, num_kernels[idx])                                

            #x =  Conv2D(256, (3,3), padding='same' )(x) 
            x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)

            #for j in range(2): 
            #    x = self._residual_block(x, num_kernels[idx]) 
            iL=x
            x = BatchNormalization()(x)
            #x = Activation('relu')(x)
            #x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(4,4),  padding='same' )(x)
            
        #y = self._residual_block(y, num_kernels[idx]) 
        #z=Add()([x,y])
            
        #x =  Conv2D(256, (3,3), padding='same'  )(x) 
        #x = LeakyReLU(alpha=0.6)(x)
        #x = ELU(alpha=1.0)(x)
        x =  Conv2D(256, (3,3), padding='same' ,activation='relu' )(x) 
        #x=Add()([x,iL])
        #x =  Conv2D(64, (3,3), padding='same' )(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)
        lr=1e-3
        #lr=0.045

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=15, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGenX4, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 

        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        #xR = LeakyReLU(alpha=0.1)(xR)
        #xR = ELU(alpha=1.0)(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        #xR = LeakyReLU(alpha=0.1)(xR)
        #xR = ELU(alpha=0.1)(xR)        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)
        if add:
            xR=Add()([xR,ini])

        return xR  

    def _residualLeaky_block(self, xR, nr_filter ,add=True):

        

        ini=xR 

        xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        xR = LeakyReLU(alpha=0.1)(xR)
        #xR = ELU(alpha=1.0)(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        xR = LeakyReLU(alpha=0.1)(xR)
        #xR = ELU(alpha=0.1)(xR)        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)
        if add:
            xR=Add()([xR,ini])

        return xR          



class ScaleGenBIGBP(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGenBIGBP, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        #self.weight_path = "weights_nobp_32_X4/scale WeightsRezidual3X3 %dX.h5" % (self.scale_factor)
        self.weight_path = "weights_nobp_32_X2/scale05WeightsRezidual5X5 %dX.h5" % (self.scale_factor)

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=48, width=48, channels=3, load_weights=False, batch_size=128):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        num_kernels = [ 128,128]
        #num_kernels = [ 96,96]
        #num_kernels = [512, 256,128]


        init = super(ScaleGenBIGBP, self).create_model(height, width, channels, load_weights, batch_size)
        

        #x = Convolution2D(128, (3, 3) ,activation='relu',  padding='same',  name='level1')(init)
        x = Convolution2D(128, (1, 1) ,activation='relu',  padding='same',  name='level0')(init)
        x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
        x = Convolution2D(128, (3, 3) ,activation='relu',  padding='same',  name='level1')(x)
        xIni =x
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)

        #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
            idx = i if i < len(num_kernels) else -1
            #if i>0:
            #    x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(10):
                x = self._residual_block(x, num_kernels[idx]) 


            #x = BatchNormalization()(x)
            #x=Add()([x,xIni])  


            #x =  Conv2D(256, (3,3), padding='same' )(x) 
            #x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
            #x =  Conv2D(256, (1,1), padding='same' )(x) 

            #for j in range(4):
            #    x = self._residual_block(x, 256)
            #for j in range(2): 
            #    x = self._residual_block(x, num_kernels[idx]) 
            
            #x = BatchNormalization()(x)
            #x = Activation('relu')(x)
            #x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(4,4),  padding='same' )(x)
            
        #y = self._residual_block(y, num_kernels[idx]) 
        #z=Add()([x,y])
            
        #x =  Conv2D(256, (3,3), padding='same'  )(x) 
        #x =  Conv2D(256, (3,3), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(64, (3,3), padding='same' )(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)
        lr=1e-3
        #lr=0.045

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse')
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        #print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=10, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGenBIGBP, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        #xR = BatchNormalization()(xR)
        ini=xR 

        #xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        #xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)
        
        xR=Lambda(resizeRes015First ,output_shape=resizeRes_outputshape)(xR)
        if add:
            #xR=Multiply()([0.1,xR])
            xR=Add()([xR,ini])


        ini=xR 

        #xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        #xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)
        xR=Lambda(resizeRes015 ,output_shape=resizeRes_outputshape)(xR)
        if add:
            xR=Add()([xR,ini])


        ini=xR 

        #xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        #xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)
        xR=Lambda(resizeRes015 ,output_shape=resizeRes_outputshape)(xR)
        if add:
            xR=Add()([xR,ini])


        #xR = BatchNormalization()(xR)    


        return xR   
      



class ScaleGenSMALLBP(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGenSMALLBP, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        #self.weight_path = "weights_nobp_32_X4/scale WeightsRezidual3X3 %dX.h5" % (self.scale_factor)
        self.weight_path = "weights_nobp_15_X2/scale WeightsRezidual3X3 %dX.h5" % (self.scale_factor)

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=48, width=48, channels=3, load_weights=False, batch_size=128):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 128,128]
        #num_kernels = [512, 256,128]


        init = super(ScaleGenSMALLBP, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(128, (3, 3) ,activation='relu',  padding='same',  name='level1')(init)
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)
        xIni=x

        #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
            idx = i if i < len(num_kernels) else -1
            #if i>0:
            #    x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(3):
                x = self._residual_block_var(x, num_kernels[idx]) 

            for j in range(7):
                x = self._residual_block(x, num_kernels[idx])   

            
            #x=Add()([x,xIni])

            #x = BatchNormalization()(x)


            #x =  Conv2D(256, (3,3), padding='same' )(x) 
            x=Lambda(resizeX2bil ,output_shape=resizeX2bil_outputshape)(x)
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
            

            #for j in range(1):
            #    x = self._residual_blockleaky(x, num_kernels[idx])
            #for j in range(2): 
            #    x = self._residual_block(x, num_kernels[idx]) 
            
            #x = BatchNormalization()(x)
            #x = Activation('relu')(x)
            #x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(4,4),  padding='same' )(x)
            
        #y = self._residual_block(y, num_kernels[idx]) 
        #z=Add()([x,y])
            
        x =  Conv2D(256, (1,1),activation='relu', padding='same'  )(x) 
        x = BatchNormalization()(x)

        x =  Conv2D(256, (3,3),activation='relu', padding='same'  )(x)
        #x =  Conv2D(512, (3,3), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(64, (3,3), padding='same' )(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        #out = Conv2D( 3, (3,3),padding='same', activation='sigmoid')(x)
        out = Conv2D( 3, (3,3),padding='same', activation='relu')(x)

        # Compile the model



        model = Model(init, out)
        lr=1e-4
        #lr=0.0002
        #lr=0.045

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse')
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        #print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=5, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGenSMALLBP, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        #xR = BatchNormalization()(xR)
        ini=xR 

        #xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        
        b = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        #xR = BatchNormalization()(xR)
        b = Activation('relu')(b)  
        #xR = LeakyReLU(0.2)(xR)      
        b = Conv2D(nr_filter, (3, 3), padding='same')(b)

        xDiff = Subtract()([b, ini])
        #xDiff = LeakyReLU(0.2)(xDiff) 
        xDiff = BatchNormalization()(xDiff)
        xDiff = Conv2D(nr_filter, (3, 3), padding='same')(xDiff)
        xDiff = LeakyReLU(0.2)(xDiff) 
        xDiff = Conv2D(nr_filter, (3, 3), padding='same')(xDiff)
        #xR=Lambda(resizeRes02 ,output_shape=resizeRes_outputshape)(xR)
        if add:
            #xR=Multiply()([0.1,xR])
            xR=Add()([b,ini,xDiff])

        return xR  

    def _residual_block_var(self, xR, nr_filter ,add=True):

        

        #xR = BatchNormalization()(xR)
        ini=xR 

        #xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        
        b = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        #xR = BatchNormalization()(xR)
        b = Activation('relu')(b)  
        #xR = LeakyReLU(0.2)(xR)      
        b = Conv2D(nr_filter, (3, 3), padding='same')(b)

        xDiff = Subtract()([b, ini])
        #xDiff=Lambda(resizeResScaleVar ,output_shape=resizeRes_outputshape)(xDiff)
        #xDiff = LeakyReLU(0.2)(xDiff) 
        xDiff = BatchNormalization()(xDiff)
        xDiff = Conv2D(nr_filter, (3, 3), padding='same')(xDiff)
        xDiff = LeakyReLU(0.2)(xDiff) 
        xDiff = Conv2D(nr_filter, (3, 3), padding='same')(xDiff)
        #xDiff=Lambda(resizeResScaleVar ,output_shape=resizeRes_outputshape)(xDiff)



        
        #xR=Lambda(resizeRes02 ,output_shape=resizeRes_outputshape)(xR)
        if add:
            #xR=Multiply()([0.1,xR])
            xR=Add()([b,ini,xDiff])

        return xR    


       
      


class ScaleGenX4Relu(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGenX4Relu, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        #self.weight_path = "weights_big_32_X4/scale WeightsRezidual %dX.h5" % (self.scale_factor)
        self.weight_path = "weights_big_32_X4/scale WeightsRezidualLeaky %dX.h5" % (self.scale_factor)

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=48, width=48, channels=3, load_weights=False, batch_size=128):        
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        #num_kernels = [512, 256,128]


        init = super(ScaleGenX4Relu, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(256, (3, 3) ,activation='relu',  padding='same',  name='level1')(init)
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)

        #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
            idx = i if i < len(num_kernels) else -1
            #if i>0:
            #    x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(10):
                x = self._residual_block(x, num_kernels[idx]) 



            #x =  Conv2D(256, (3,3), padding='same' )(x) 
            x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)

            for j in range(2):
                x = self._residual_blockleaky(x, num_kernels[idx])
                #x = self._residual_block(x, num_kernels[idx]) 
            #for j in range(2): 
            #    x = self._residual_block(x, num_kernels[idx]) 
            
            x = BatchNormalization()(x)
            #x = Activation('relu')(x)
            #x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(4,4),  padding='same' )(x)
            
        #y = self._residual_block(y, num_kernels[idx]) 
        #z=Add()([x,y])
            
        #x =  Conv2D(256, (3,3), padding='same'  )(x) 
        x =  Conv2D(256, (3,3), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(64, (3,3), padding='same' )(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)
        lr=1e-3
        #lr=0.045

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=8, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGenX4Relu, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 

        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)
        if add:
            xR=Add()([xR,ini])

        return xR   
   
    def _residual_blockleaky(self, xR, nr_filter ,add=True):

        

        ini=xR 

        xR = BatchNormalization()(xR)
        #xR = Activation('relu')(xR)
        xR = LeakyReLU(alpha=0.1)(xR)
        
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)  
        #xR = LeakyReLU(alpha=0.1)(xR)      
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)
        if add:
            xR=Add()([xR,ini])

        return xR   


class ScaleGenX(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGenX, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights_big_32_gen/scale WeightsRezidual6 %dX.h5" % (self.scale_factor)

    #def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
    def create_model(self, height=48, width=48, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        #num_kernels = [512, 256,128]


        init = super(ScaleGenX, self).create_model(height, width, channels, load_weights, batch_size)

        
        x = Convolution2D(256, (3, 3) ,activation='relu',  padding='same',  name='level1')(init)
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)

        #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            #x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)
            idx = i if i < len(num_kernels) else -1
            #if i>0:
            #    x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(10):
                x = self._residual_block(x, num_kernels[idx]) 


            #x =  Conv2D(256, (3,3), padding='same' )(x) 
            x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)

            #x = self._residual_block(x, num_kernels[idx]) 
            
            x = BatchNormalization()(x)
            #x = Activation('relu')(x)
            #x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(4,4),  padding='same' )(x)
            
        #y = self._residual_block(y, num_kernels[idx]) 
        #z=Add()([x,y])
            
        #x =  Conv2D(256, (3,3), padding='same'  )(x) 
        x =  Conv2D(256, (3,3), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(64, (3,3), padding='same' )(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)
        lr=1e-3
        #lr=0.045

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model

    def fit(self, batch_size=25, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGenX, self).fit(batch_size, nb_epochs, save_history, history_fn)

    

    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 

        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)        
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR)
        if add:
            xR=Add()([xR,ini])

        return xR    


class ScaleGen357gen(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGen357gen, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights_big_32_gen/scale Weights357 %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        #num_kernels = [512, 256,128]


        init = super(ScaleGen357gen, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='level1')(init)
        y = Convolution2D(256, (5, 5), activation='relu', padding='same', name='level1y')(init)
        z = Convolution2D(256, (7, 7), activation='relu', padding='same', name='level1z')(init)
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)


   
        for i in range(0, len(num_kernels)-1):
            idx = i if i < len(num_kernels) else -1
            
            for j in range(2):
                x = self._residual_block(x, num_kernels[idx]) 
            x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)

        for i in range(0, len(num_kernels)-1):
            idx = i if i < len(num_kernels) else -1
            
            for j in range(2):
                y = self._residual_block(y, num_kernels[idx]) 
            y=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(y)

        for i in range(0, len(num_kernels)-1):
            idx = i if i < len(num_kernels) else -1
            
            for j in range(2):
                z = self._residual_block(z, num_kernels[idx]) 
            z=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(z)            
        # x = self._residual_block(x, num_kernels[idx])

        out= Add()([x,y,z])



            
        out =  Conv2D(256, (3,3), padding='same' ,activation='relu')(out) 
        out =  Conv2D(96, (1,1), padding='same' ,activation='relu')(out) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(out)

        # Compile the model



        model = Model(init, out)
        lr=1e-3
        #lr=0.1

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=15, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGen357gen, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 

        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        if add:
            xR=Add()([xR,ini])

        return xR   
   

class ScaleGen32gen(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGen32gen, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights_big_gen/scale Weights16 %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        num_kernels = [ 256,256]
        #num_kernels = [512, 256,128]


        init = super(ScaleGen32gen, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='level1')(init)
        #x = Convolution2D(256, (3, 3),  padding='same', name='level1')(init)


   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            idx = i if i < len(num_kernels) else -1
            #if i>0:
            #    x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(10):
                x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)

            for j in range(2):
                x = self._residual_block(x, num_kernels[idx]) 

            #x = self._residual_block(x, num_kernels[idx]) 
            
            x = BatchNormalization()(x)
            #x = Activation('relu')(x)
            #x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(1,1),  padding='same' )(x)
            
       # x = self._residual_block(x, num_kernels[idx]) 

            
        x =  Conv2D(256, (3,3), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(96, (1,1), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)
        lr=1e-3
        #lr=0.1

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=3, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGen32gen, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 

        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        if add:
            xR=Add()([xR,ini])

        return xR   
   



class ScaleGen32gen2(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGen32gen2, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights_big_32_gen/scale Weights2563X3 %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        #num_kernels = [512, 256,128]


        init = super(ScaleGen32gen2, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='level1')(init)
        #x = Convolution2D(128, (self.f2_2, self.f2_2), activation='relu', padding='same', name='lavel1_2')(x)
        #x = MaxPooling2D((2,2))(x)

   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            idx = i if i < len(num_kernels) else -1
            if i>0:
                x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(2):
                x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX4 ,output_shape=resizeX4_outputshape)(x)

            #x = self._residual_block(x, num_kernels[idx]) 
            
            #x = BatchNormalization()(x)
            #x = Activation('relu')(x)
            #x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(1,1),  padding='same' )(x)
            
        x = self._residual_block(x, num_kernels[idx]) 

            
        x =  Conv2D(128, (3,3), padding='same' ,activation='relu')(x) 
        x =  Conv2D(96, (1,1), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)
        lr=1e-3

        adam = optimizers.Adam(lr,0.9)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=10, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGen32gen2, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 
        #xUP = UpSampling2D((2,2))(x) 
        #xZE =ZeroPadding2D((step_pad,step_pad))(x)            
        #x=Add()([xUP,xT,xZE])

        #x = LeakyReLU()(x)
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        if add:
            xR=Add()([xR,ini])

        return xR   
   



class ScaleGenPLAQUE(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGen, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights/scale WeightsPlateCar %dX.h5" % (self.scale_factor)

    def create_model(self, height=50, width=50, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,128,96]


        init = super(ScaleGen, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(64, (1, 1), activation='relu', padding='same', name='level1')(init)
        x = Convolution2D(128, (self.f2_2, self.f2_2), activation='relu', padding='same', name='lavel1_2')(x)
        x = MaxPooling2D((2,2))(x)

   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            idx = i if i < len(num_kernels) else -1
            x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(2):
                x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
            #x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(1,1),  padding='same' )(x)
            
            

            
        x =  Conv2D(96, (3,3), padding='same' ,activation='relu')(x) 
        x =  Conv2D(96, (1,1), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=15, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGen, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 
        #xUP = UpSampling2D((2,2))(x)
        #xZE =ZeroPadding2D((step_pad,step_pad))(x)            
        #x=Add()([xUP,xT,xZE])

        #x = LeakyReLU()(x)
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        if add:
            xR=Add()([xR,ini])

        return xR   


class ScaleGen(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGen, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights_big/scale WeightsGamma2 %dX.h5" % (self.scale_factor)

    def create_model(self, height=16, width=16, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        #num_kernels = [ 256,192,128,96]
        num_kernels = [512, 256,128,96]


        init = super(ScaleGen, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(512, (1, 1), activation='relu', padding='same', name='level1')(init)
        #x = Convolution2D(128, (self.f2_2, self.f2_2), activation='relu', padding='same', name='lavel1_2')(x)
        #x = MaxPooling2D((2,2))(x)

   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            idx = i if i < len(num_kernels) else -1
            if i>0:
                x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(2):
                x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
            #x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(1,1),  padding='same' )(x)
            
            

            
        x =  Conv2D(96, (3,3), padding='same' ,activation='relu')(x) 
        x =  Conv2D(96, (1,1), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=28, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGen, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 
        #xUP = UpSampling2D((2,2))(x) 
        #xZE =ZeroPadding2D((step_pad,step_pad))(x)            
        #x=Add()([xUP,xT,xZE])

        #x = LeakyReLU()(x)
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        if add:
            xR=Add()([xR,ini])

        return xR   
  


class ScaleGen16(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGen16, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        #self.weight_path = "weights/scale WeightsBIG_NAD_16 %dX.h5" % (self.scale_factor)
        #self.weight_path = "weights/scale WeightsBIG_RELU_16 %dX.h5" % (self.scale_factor) 

        #self.weight_path = "weights/scale Weights512_RELU_16 %dX.h5" % (self.scale_factor) 

        #self.weight_path = "weights/scale Weights512_RELU_48 %dX.h5" % (self.scale_factor)   
        self.weight_path = "weights/scale WeightsGamma2 %dX.h5" % (self.scale_factor) 
        

    def create_model(self, height=16, width=16, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128,96]
        num_kernels = [ 512,256,128,96]


        init = super(ScaleGen16, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(512, (1, 1), activation='relu', padding='same', name='level1')(init)
        #x = Convolution2D(128, (self.f2_2, self.f2_2), activation='relu', padding='same', name='lavel1_2')(x)
        #x = MaxPooling2D((2,2))(x)

   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            idx = i if i < len(num_kernels) else -1
            if i>0:
                x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(2):
                x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
            #x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            #x = LeakyReLU()(x)
            x = Activation('relu')(x)
            #x = ReLU()(x)
            x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(1,1),  padding='same' )(x)
            
            


        x =  Conv2D(96, (3, 3), padding='same' ,activation='relu')(x) 
        x =  Conv2D(96, (1,1), padding='same' ,activation='relu')(x)
            
        #x =  Conv2D(32, (3,3), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=8, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGen16, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 
        #xUP = UpSampling2D((2,2))(x)
        #xZE =ZeroPadding2D((step_pad,step_pad))(x)            
        #x=Add()([xUP,xT,xZE])

        #x = LeakyReLU()(x)
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        if add:
            xR=Add()([xR,ini])

        return xR   

class ScaleGen16L(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGen16L, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights/scale WeightsBIG_NAD16 %dX.h5" % (self.scale_factor)

    def create_model(self, height=16, width=16, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,128,96, 32]


        init = super(ScaleGen16L, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(96, (1, 1), activation='relu', padding='same', name='level1')(init)
        x = Convolution2D(128, (self.f2_2, self.f2_2), activation='relu', padding='same', name='lavel1_2')(x)
        #x = MaxPooling2D((2,2))(x)

   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            idx = i if i < len(num_kernels) else -1
            x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(2):
                x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
            #x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(1,1),  padding='same' )(x)
            
            

            
        x = LeakyReLU()( Conv2D(32, (3,3), padding='same')(x) )

        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        x =  Conv2D(8, (5, 5), padding='same' ,activation='relu',)(x) 
        x =  Conv2D(8, (3, 3), padding='same', activation='relu',)(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=5, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGen16L, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 
        #xUP = UpSampling2D((2,2))(x)
        #xZE =ZeroPadding2D((step_pad,step_pad))(x)            
        #x=Add()([xUP,xT,xZE])

        #x = LeakyReLU()(x)
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        if add:
            xR=Add()([xR,ini])

        return xR 


class ScaleGen32(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGen32, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights_big_32/scale Weights2563X3 %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        num_kernels = [ 256,192,128]
        #num_kernels = [512, 256,128]


        init = super(ScaleGen32, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='level1')(init)
        #x = Convolution2D(128, (self.f2_2, self.f2_2), activation='relu', padding='same', name='lavel1_2')(x)
        #x = MaxPooling2D((2,2))(x)

   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            idx = i if i < len(num_kernels) else -1
            if i>0:
                x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(2):
                x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
            #x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(1,1),  padding='same' )(x)
            
            

            
        x =  Conv2D(128, (3,3), padding='same' ,activation='relu')(x) 
        x =  Conv2D(96, (1,1), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-4)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=48, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGen32, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 
        #xUP = UpSampling2D((2,2))(x) 
        #xZE =ZeroPadding2D((step_pad,step_pad))(x)            
        #x=Add()([xUP,xT,xZE])

        #x = LeakyReLU()(x)
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        if add:
            xR=Add()([xR,ini])

        return xR   

class ScaleGen32_512(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ScaleGen32, self).__init__("Image ScaleGen", scale_factor)

        #self.f1 = 7
        self.f1 = 1
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 32
        self.n2 = 64
        self.n3 = 96
        self.n4 = 128
        self.n5 = 256

        self.weight_path = "weights_big_32/scale WeightsGAUSS %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        deconv_layers=4
        #num_kernels = [ 256,192,128,96]
        num_kernels = [512, 256,128]


        init = super(ScaleGen32, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(512, (1, 1), activation='relu', padding='same', name='level1')(init)
        #x = Convolution2D(128, (self.f2_2, self.f2_2), activation='relu', padding='same', name='lavel1_2')(x)
        #x = MaxPooling2D((2,2))(x)

   
        for i in range(0, len(num_kernels)-1):
            # Apply 5x5 and 3x3 convolutions

            # If we didn't specify the number of kernels to use for this many
            # layers, just repeat the last one in the list.
            idx = i if i < len(num_kernels) else -1
            if i>0:
                x = Convolution2D(num_kernels[idx], (1,1), activation='relu', padding='same')(x)

            for j in range(2):
                x = self._residual_block(x, num_kernels[idx]) 

            x=Lambda(resizeX2 ,output_shape=resizeX2_outputshape)(x)
            #x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2DTranspose(num_kernels[idx], (3,3),strides=(1,1),  padding='same' )(x)
            
            

            
        x =  Conv2D(96, (3,3), padding='same' ,activation='relu')(x) 
        x =  Conv2D(96, (1,1), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(16, (3, 3), padding='same' ,activation='relu')(x) 
        # Last deconvolution layer: Create 3-channel image.
        #x = MaxPooling2D((1,1))(x)
        #x =  Conv2D(8, (5, 5), padding='same' ,activation='relu')(x) 
        #x =  Conv2D(8, (3, 3), padding='same', activation='relu')(x) 
        out = Conv2D( 3, (3, 3),padding='same', activation='sigmoid')(x)

        # Compile the model



        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        #nadam = optimizers.Nadam(lr=1e-2)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #load_weights=True
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model


    def fit(self, batch_size=28, nb_epochs=100, save_history=False, history_fn="ScaleGen History.txt"):
        return super(ScaleGen32, self).fit(batch_size, nb_epochs, save_history, history_fn)


    def _residual_block(self, xR, nr_filter ,add=True):

        

        ini=xR 
        #xUP = UpSampling2D((2,2))(x) 
        #xZE =ZeroPadding2D((step_pad,step_pad))(x)            
        #x=Add()([xUP,xT,xZE])

        #x = LeakyReLU()(x)
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        xR = BatchNormalization()(xR)
        xR = Activation('relu')(xR)
        xR = Conv2D(nr_filter, (3, 3), padding='same')(xR) 
        if add:
            xR=Add()([xR,ini])

        return xR   
   




class ImageSuperResolutionModel(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ImageSuperResolutionModel, self).__init__("Image SR", scale_factor)

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/SR Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = super(ImageSuperResolutionModel, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(self.n1, (self.f1, self.f1), activation='relu', padding='same', name='level1')(init)
        x = Convolution2D(self.n2, (self.f2, self.f2), activation='relu', padding='same', name='level2')(x)

        out = Convolution2D(channels, (self.f3, self.f3), padding='same', name='output')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="SRCNN History.txt"):
        return super(ImageSuperResolutionModel, self).fit(batch_size, nb_epochs, save_history, history_fn)


class ExpantionSuperResolution(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ExpantionSuperResolution, self).__init__("Expanded Image SR", scale_factor)

        self.f1 = 9
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/Expantion SR Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = super(ExpantionSuperResolution, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(self.n1, (self.f1, self.f1), activation='relu', padding='same', name='level1')(init)

        x1 = Convolution2D(self.n2, (self.f2_1, self.f2_1), activation='relu', padding='same', name='lavel1_1')(x)
        x2 = Convolution2D(self.n2, (self.f2_2, self.f2_2), activation='relu', padding='same', name='lavel1_2')(x)
        x3 = Convolution2D(self.n2, (self.f2_3, self.f2_3), activation='relu', padding='same', name='lavel1_3')(x)

        x = Average()([x1, x2, x3])

        out = Convolution2D(channels, (self.f3, self.f3), activation='relu', padding='same', name='output')(x)

        model = Model(init, out)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="ESRCNN History.txt"):
        return super(ExpantionSuperResolution, self).fit(batch_size, nb_epochs, save_history, history_fn)


class DenoisingAutoEncoderSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DenoisingAutoEncoderSR, self).__init__("Denoise AutoEncoder SR", scale_factor)

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/Denoising AutoEncoder %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import Deconvolution2D

        # Perform check that model input shape is divisible by 4
        init = super(DenoisingAutoEncoderSR, self).create_model(height, width, channels, load_weights, batch_size)

        if K.image_dim_ordering() == "th":
            output_shape = (None, channels, width, height)
        else:
            output_shape = (None, width, height, channels)

        #print("shape ini")
        #print(init.shape)

        level1_1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        level2_1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(level1_1)

        level2_2 = Convolution2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2_1)
        level2 = Add()([level2_1, level2_2])

        level1_2 = Convolution2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2)
        level1 = Add()([level1_1, level1_2])

        decoded = Convolution2D(channels, (5, 5), activation='linear', padding='same')(level1)
        print("shape out")
        #print(decoded.output_shape())
        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        print("shape model out")
        print(model.summary())
        self.model = model
        return model

    def fit(self, batch_size=28, nb_epochs=100, save_history=True, history_fn="DSRCNN History.txt"):
        return super(DenoisingAutoEncoderSR, self).fit(batch_size, nb_epochs, save_history, history_fn)



class MEDenoisingSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(MEDenoisingSR, self).__init__("MEDenoisingSR", scale_factor)

        #self.n1 = 64
        self.n1 = 128
        self.n2 = 32

        self.weight_path = "weights/MEDenoisingSR %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import Deconvolution2D

        # Perform check that model input shape is divisible by 4
        init = super(MEDenoisingSR, self).create_model(height, width, channels, load_weights, batch_size)

        if K.image_dim_ordering() == "th":
            output_shape = (None, channels, width, height)
        else:
            output_shape = (None, width, height, channels)

        #print("shape ini")
        #print(init.shape)

        level1_1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        level1_1 = LeakyReLU(alpha=0.2)(level1_1)
        level2_1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(level1_1)
        level2_1 = LeakyReLU(alpha=0.2)(level2_1)

        level2_2 = Convolution2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2_1)
        level2 = Add()([level2_1, level2_2])
        level2 = LeakyReLU(alpha=0.2)(level2)

        level1_2 = Convolution2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2)
        level1 = Add()([level1_1, level1_2])
        level1 = LeakyReLU(alpha=0.2)(level1)

        decoded = Convolution2D(channels, (5, 5), activation='linear', padding='same')(level1)
        #print("shape out")
        #print(decoded.output_shape())
        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        #print("shape model out")
        #print(model.summary())
        self.model = model
        return model

    def fit(self, batch_size=28, nb_epochs=100, save_history=True, history_fn="MEDenoisingSRHistory.txt"):
        return super(MEDenoisingSR, self).fit(batch_size, nb_epochs, save_history, history_fn)

class DenoisingMe(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DenoisingMe, self).__init__("DenoisingMe ", scale_factor)

        #self.n1 = 64
        self.n1 = 128
        self.n2 = 32

        self.weight_path = "weights/DenoisingMe %dXLAST.h5" % (self.scale_factor)
        #self.weight_path = "weights/DenoisingMe128 %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import Deconvolution2D

        # Perform check that model input shape is divisible by 4
        init = super(DenoisingMe, self).create_model(height, width, channels, load_weights, batch_size)

        if K.image_dim_ordering() == "th":
            output_shape = (None, channels, width, height)
        else:
            output_shape = (None, width, height, channels)

        #print("shape ini")
        #print(init.shape)
        #level0_0 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        #level0_1 = Convolution2D(self.n1, (2, 2), activation='relu', padding='same')(level0_0)


        level1_1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        level2_1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(level1_1)
        





        level2_2 = Convolution2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2_1)
        level2 = Add()([level2_1, level2_2])
        for i in range(1):
            level2 = self._residual_Separable_block(level2 , "level2",i)
        



        level1_2 = Convolution2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2)
        level1 = Add()([level1_1, level1_2])
        for i in range(1):
            level1 = self._residual_Separable_block(level1 , "level1" ,i+4)        

        

        decoded = Convolution2D(channels, (5, 5), activation='linear', padding='same')(level1)
        print("shape out")
        #print(decoded.output_shape())





        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #if load_weights: model.load_weights(self.weight_path)

        print("shape model out")
        print(model.summary())
        self.model = model
        #sys.exit("GATA MODEL")
        return model




    def _residual_Separable_block(self, ip, prefix ,idnr):
        id=str(prefix)+'_'+str(idnr)      
        residual = ip

        
        x = Activation('relu')(ip)
        x = SeparableConv2D(self.n1, (3, 1),  padding='same', use_bias=False,
                          name='sr_res_conv_' + str(id) + '_1')(x)
        x = BatchNormalization( name="sr_res_sep_batchnorm0_" + str(id) + "_1")(x)
        x = Activation('relu')(x)
        #x = LeakyReLU(alpha=0.2)(x)


        x = SeparableConv2D(self.n1, (1, 3),  padding='same', use_bias=False,
                          name='sr_res_conv3_' + str(id) + '_1')(x)
        x = BatchNormalization( name="sr_res_sep_batchnorm3_" + str(id) + "_1")(x)
        x = Activation('relu', name="sr_res_activation3_" + str(id) + "_1")(x)
        #x = LeakyReLU(alpha=0.2)(x)


        x = SeparableConv2D(self.n1, (3, 3),  padding='same',use_bias=False,
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization( name="sr_res_sep_batchnorm1_" + str(id) + "_2")(x)

        m = Add(name="sr_res_merge_" + str(id))([x, residual])

        return m  


    def _residual_Separable_blockBK(self, ip, prefix ,idnr):
        id=str(prefix)+'_'+str(idnr)      
        residual = ip

        x = Activation('relu')(ip)
        x = SeparableConv2D(self.n1, (3, 3),  padding='same', use_bias=False,
                          name='sr_res_conv_' + str(id) + '_1')(x)
        x = BatchNormalization( name="sr_res_sep_batchnorm0_" + str(id) + "_1")(x)
        x = Activation('relu')(x)


        #x = SeparableConv2D(64, (3, 3),  padding='same', use_bias=False,
        #                  name='sr_res_conv3_' + str(id) + '_1')(x)
        #x = BatchNormalization( name="sr_res_sep_batchnorm3_" + str(id) + "_1")(x)
        #x = Activation('relu', name="sr_res_activation3_" + str(id) + "_1")(x)


        x = SeparableConv2D(self.n1, (3, 3),  padding='same',use_bias=False,
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization( name="sr_res_sep_batchnorm1_" + str(id) + "_2")(x)

        m = Add(name="sr_res_merge_" + str(id))([x, residual])
        #m = Activation('relu')(m)

        return m  

    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x          


    def fit(self, batch_size=40, nb_epochs=100, save_history=True, history_fn="DSRCNN History.txt"):
        return super(DenoisingMe, self).fit(batch_size, nb_epochs, save_history, history_fn)       




class DeepDenoiseSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DeepDenoiseSR, self).__init__("Deep Denoise SR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

        self.weight_path = "weights/Deep Denoise Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        # Perform check that model input shape is divisible by 4
        init = super(DeepDenoiseSR, self).create_model(height, width, channels, load_weights, batch_size)

        c1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        c1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Convolution2D(self.n3, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2_2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)

        c1_2 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(m1)
        c1_2 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(c1_2)

        m2 = Add()([c1, c1_2])

        decoded = Convolution2D(channels, 5, 5, activation='linear', padding='same')(m2)

        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model

    def fit(self, batch_size=28, nb_epochs=100, save_history=True, history_fn="Deep DSRCNN History.txt"):
        super(DeepDenoiseSR, self).fit(batch_size, nb_epochs, save_history, history_fn)

class DeepDenoiseSREPS(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DeepDenoiseSREPS, self).__init__("Deep Denoise SREPS", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

        self.weight_path = "weights/Deep Denoise WeightsEPS %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        # Perform check that model input shape is divisible by 4
        init = super(DeepDenoiseSREPS, self).create_model(height, width, channels, load_weights, batch_size)

        c1 = Conv2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        c1 = Conv2D(self.n1, (3, 3), activation='relu', padding='same')(c1)

        x_c1 = MaxPooling2D((2, 2))(c1)
        x= Dropout(0.05)(x_c1)

        c2 = Conv2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2 = Conv2D(self.n2, (3, 3), activation='relu', padding='same')(c2)

        x_c2 = MaxPooling2D((2, 2))(c2)
        x= Dropout(0.05)(x_c2)

        c3 = Conv2D(self.n3, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Conv2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2_2 = Conv2D(self.n2, (3, 3), activation='relu', padding='same')(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)

        c1_2 = Conv2D(self.n1, (3, 3), activation='relu', padding='same')(m1)
        c1_2 = Conv2D(self.n1, (3, 3), activation='relu', padding='same')(c1_2)

        m2 = Add()([c1, c1_2])

        decoded = Conv2D(channels, (5, 5), activation='linear', padding='same')(m2)

        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        return model

    def fit(self, batch_size=28, nb_epochs=100, save_history=True, history_fn="Deep DSRCNNEPS History.txt"):
        super(DeepDenoiseSREPS, self).fit(batch_size, nb_epochs, save_history, history_fn)        

class ResNetSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ResNetSR, self).__init__("ResNetSR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True
        self.uses_learning_phase = False

        self.n = 64
        self.mode = 2

        self.weight_path = "weights/ResNetSR %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        init =  super(ResNetSR, self).create_model(height, width, channels, load_weights, batch_size)

        x0 = Convolution2D(64, (3, 3), activation='relu', padding='same', name='sr_res_conv1')(init)

        x1 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), name='sr_res_conv2')(x0)

        x2 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), name='sr_res_conv3')(x1)

        x = self._residual_block(x2, 1)

        nb_residual = 2
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        #x = self._upscale_block(x, 1)
        #x = Add()([x, x1])

        #x = self._upscale_block(x, 2)
        #x = Add()([x, x0])

        x = Convolution2D(64, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)

        model = Model(init, x)

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        #if load_weights: model.load_weights(self.weight_path)
        print(model.summary())
        self.model = model
        sys.exit("GATA MODEL")
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        init = ip

        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        x = Convolution2D(256, (3, 3), activation="relu", padding='same', name='sr_res_upconv1_%d' % id)(init)
        x = SubPixelUpscaling(r=2, channels=self.n, name='sr_res_upscale1_%d' % id)(x)
        x = Convolution2D(self.n, (3, 3), activation="relu", padding='same', name='sr_res_filter1_%d' % id)(x)

        return x

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="ResNetSR History.txt"):
        super(ResNetSR, self).fit(batch_size, nb_epochs, save_history, history_fn)







def resizeX4(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [4 * int(s) for s in prev_shape[1:3]]
    return tf.image.resize_nearest_neighbor(my_input, size)
    #return tf.image.resize_bicubic(my_input, size) 
    #eturn tf.image.resize_bilinear(my_input, size) 


def resizeX4_outputshape(my_input_shape):
    shape = list(my_input_shape)
    print (shape)    
    size = [4 * int(s) for s in my_input_shape[1:3]]
    shape[1]=size[0] 
    shape[2]=size[1] 
    print (shape) 
    return tuple(shape) 

def resizeX4bil(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [4 * int(s) for s in prev_shape[1:3]]
    #return tf.image.resize_nearest_neighbor(my_input, size)
    #return tf.image.resize_bicubic(my_input, size) 
    return tf.image.resize_bilinear(my_input, size)


def resizeX4bil_outputshape(my_input_shape):
    shape = list(my_input_shape)
    print (shape)    
    size = [4 * int(s) for s in my_input_shape[1:3]]
    shape[1]=size[0] 
    shape[2]=size[1] 
    print (shape) 
    return tuple(shape)     

def resizeX2(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [2 * int(s) for s in prev_shape[1:3]]
    #return tf.image.resize_nearest_neighbor(my_input, size)
    return tf.image.resize_bilinear(my_input, size)
    #return tf.image.resize_bicubic(my_input, size)


def resizeX2_outputshape(my_input_shape):
    shape = list(my_input_shape)
    print (shape)    
    size = [2 * int(s) for s in my_input_shape[1:3]]
    shape[1]=size[0] 
    shape[2]=size[1] 
    print (shape) 
    return tuple(shape)   



def resizeRes01(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.1,  my_input)


def resizeRes015First(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.15,  my_input)
def resizeRes015(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.15,  my_input)


def resizeRes11(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(1.1,  my_input)

def resizeRes02(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.2,  my_input)

def resizeRes05(my_input): # resizes input tensor wrt. ref_tensor 
    return tf.scalar_mul(0.5,  my_input)    

def resizeRes_outputshape(my_input_shape):
    return my_input_shape







   



def resizeX2bil(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [2 * int(s) for s in prev_shape[1:3]]
    #return tf.image.resize_nearest_neighbor(my_input, size)
    #return tf.image.resize_bicubic(my_input, size) 
    return tf.image.resize_bilinear(my_input, size)


def resizeX2bil_outputshape(my_input_shape):
    shape = list(my_input_shape)
    print (shape)    
    size = [2 * int(s) for s in my_input_shape[1:3]]
    shape[1]=size[0] 
    shape[2]=size[1] 
    print (shape) 
    return tuple(shape) 

def resizeX2d(my_input): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    #prev_shape = my_input.output.get_shape()
    prev_shape = my_input._keras_shape
    size = [2 * int(s) for s in prev_shape[1:3]]
    return tf.image.resize_nearest_neighbor(my_input, size)
    #return tf.image.resize_bicubic(my_input, size) 
    #return tf.image.resize_bilinear(my_input, size)


def resizeX2d_outputshape(my_input_shape):
    shape = list(my_input_shape)
    print (shape)    
    size = [2 * int(s) for s in my_input_shape[1:3]]
    shape[1]=size[0] 
    shape[2]=size[1] 
    print (shape) 
    return tuple(shape) 


















def _evaluate(sr_model : BaseSuperResolutionModel, validation_dir, scale_pred=False):
    """
    Evaluates the model on the Validation images
    """
    print("Validating %s model" % sr_model.model_name)
    if sr_model.model == None: sr_model.create_model(load_weights=True)
    if sr_model.evaluation_func is None:
        if sr_model.uses_learning_phase:
            sr_model.evaluation_func = K.function([sr_model.model.layers[0].input, K.learning_phase()],
                                                  [sr_model.model.layers[-1].output])
        else:
            sr_model.evaluation_func = K.function([sr_model.model.layers[0].input],
                                              [sr_model.model.layers[-1].output])
    predict_path = "val_predict/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    validation_path_set5 = validation_dir + "set5/"
    validation_path_set14 = validation_dir + "set14/"
    validation_dirs = [validation_path_set5, validation_path_set14]
    for val_dir in validation_dirs:
        image_fns = [name for name in os.listdir(val_dir)]
        nb_images = len(image_fns)
        print("Validating %d images from path %s" % (nb_images, val_dir))

        total_psnr = 0.0

        for impath in os.listdir(val_dir):
            t1 = time.time()

            # Input image
            y = img_utils.imread(val_dir + impath, mode='RGB')
            width, height, _ = y.shape

            if sr_model.type_requires_divisible_shape:
                # Denoise models require precise width and height, divisible by 4

                if ((width // sr_model.scale_factor) % 4 != 0) or ((height // sr_model.scale_factor) % 4 != 0) \
                        or (width % 2 != 0) or (height % 2 != 0):
                    width = ((width // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor
                    height = ((height // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor

                    print("Model %s require the image size to be divisible by 4. New image size = (%d, %d)" % \
                          (sr_model.model_name, width, height))

                    y = img_utils.imresize(y, (width, height), interp='bicubic')

            y = y.astype('float32')
            x_width = width if not sr_model.type_true_upscaling else width // sr_model.scale_factor
            x_height = height if not sr_model.type_true_upscaling else height // sr_model.scale_factor

            x_temp = y.copy()

            if sr_model.type_scale_type == "tanh":
                x_temp = (x_temp - 127.5) / 127.5
                y = (y - 127.5) / 127.5
            else:
                x_temp /= 255.
                y /= 255.

            y = np.expand_dims(y, axis=0)

            img = img_utils.imresize(x_temp, (x_width, x_height),
                                     interp='bicubic')

            if not sr_model.type_true_upscaling:
                img = img_utils.imresize(img, (x_width, x_height), interp='bicubic')
                print("TRUE SCALE")


            x = np.expand_dims(img, axis=0)

            if K.image_dim_ordering() == "th":
                x = x.transpose((0, 3, 1, 2))
                y = y.transpose((0, 3, 1, 2))

            if sr_model.uses_learning_phase:
                y_pred = sr_model.evaluation_func([x, 0])[0][0]
            else:
                y_pred = sr_model.evaluation_func([x])[0][0]

            if scale_pred:
                if sr_model.type_scale_type == "tanh":
                    y_pred = (y_pred + 1) * 127.5
                else:
                    y_pred *= 255.

            if sr_model.type_scale_type == 'tanh':
                y = (y + 1) / 2

            psnr_val = psnr(y[0], np.clip(y_pred, 0, 255) / 255)
            total_psnr += psnr_val

            t2 = time.time()
            print("Validated image : %s, Time required : %0.2f, PSNR value : %0.4f" % (impath, t2 - t1, psnr_val))

            generated_path = predict_path + "%s_%s_generated.png" % (sr_model.model_name, os.path.splitext(impath)[0])

            if K.image_dim_ordering() == "th":
                y_pred = y_pred.transpose((1, 2, 0))

            y_pred = np.clip(y_pred, 0, 255).astype('uint8')
            img_utils.imsave(generated_path, y_pred)

        print("Average PRNS value of validation images = %00.4f \n" % (total_psnr / nb_images))


def _evaluate_denoise(sr_model : BaseSuperResolutionModel, validation_dir, scale_pred=False):
    print("Validating %s model" % sr_model.model_name)
    predict_path = "val_predict/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    validation_path_set5 = validation_dir + "set5/"
    validation_path_set14 = validation_dir + "set14/"

    validation_dirs = [validation_path_set5, validation_path_set14]
    for val_dir in validation_dirs:
        image_fns = [name for name in os.listdir(val_dir)]
        nb_images = len(image_fns)
        print("Validating %d images from path %s" % (nb_images, val_dir))

        total_psnr = 0.0

        for impath in os.listdir(val_dir):
            t1 = time.time()

            # Input image
            y = img_utils.imread(val_dir + impath, mode='RGB')
            width, height, _ = y.shape

            if ((width // sr_model.scale_factor) % 4 != 0) or ((height // sr_model.scale_factor) % 4 != 0) \
                    or (width % 2 != 0) or (height % 2 != 0):
                width = ((width // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor
                height = ((height // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor

                print("Model %s require the image size to be divisible by 4. New image size = (%d, %d)" % \
                      (sr_model.model_name, width, height))

                y = img_utils.imresize(y, (width, height), interp='bicubic')

            y = y.astype('float32')
            y = np.expand_dims(y, axis=0)

            x_temp = y.copy()

            if sr_model.type_scale_type == "tanh":
                x_temp = (x_temp - 127.5) / 127.5
                y = (y - 127.5) / 127.5
            else:
                x_temp /= 255.
                y /= 255.

            img = img_utils.imresize(x_temp[0], (width // sr_model.scale_factor, height // sr_model.scale_factor),
                                     interp='bicubic', mode='RGB')

            if not sr_model.type_true_upscaling:
                img = img_utils.imresize(img, (width, height), interp='bicubic')

            x = np.expand_dims(img, axis=0)

            if K.image_dim_ordering() == "th":
                x = x.transpose((0, 3, 1, 2))
                y = y.transpose((0, 3, 1, 2))

            sr_model.model = sr_model.create_model(height, width, load_weights=True)

            if sr_model.evaluation_func is None:
                if sr_model.uses_learning_phase:
                    sr_model.evaluation_func = K.function([sr_model.model.layers[0].input, K.learning_phase()],
                                                          [sr_model.model.layers[-1].output])
                else:
                    sr_model.evaluation_func = K.function([sr_model.model.layers[0].input],
                                                          [sr_model.model.layers[-1].output])

            if sr_model.uses_learning_phase:
                y_pred = sr_model.evaluation_func([x, 0])[0][0]
            else:
                y_pred = sr_model.evaluation_func([x])[0][0]

            if scale_pred:
                if sr_model.type_scale_type == "tanh":
                    y_pred = (y_pred + 1) * 127.5
                else:
                    y_pred *= 255.

            if sr_model.type_scale_type == 'tanh':
                y = (y + 1) / 2

            psnr_val = psnr(y[0], np.clip(y_pred, 0, 255) / 255)
            total_psnr += psnr_val

            t2 = time.time()
            print("Validated image : %s, Time required : %0.2f, PSNR value : %0.4f" % (impath, t2 - t1, psnr_val))

            generated_path = predict_path + "%s_%s_generated.png" % (sr_model.model_name, os.path.splitext(impath)[0])

            if K.image_dim_ordering() == "th":
                y_pred = y_pred.transpose((1, 2, 0))

            y_pred = np.clip(y_pred, 0, 255).astype('uint8')
            img_utils.imsave(generated_path, y_pred)

        print("Average PRNS value of validation images = %00.4f \n" % (total_psnr / nb_images))

