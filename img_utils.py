from __future__ import print_function, division, absolute_import

import numpy as np
import scipy
from scipy.misc import imsave, imread, imresize
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.ndimage.filters import gaussian_filter
from skimage.util.shape import view_as_windows
from keras import backend as K

import os
import time
#import cv2

'''
_image_scale_multiplier is a special variable which is used to alter image size.

The default image size is 32x32. If a true upscaling model is used, then the input image size is 16x16,
which not offer adequate training samples.
'''
_image_scale_multiplier = 1

img_size = 256 * _image_scale_multiplier
stride = 16 * _image_scale_multiplier

assert (img_size ** 2) % (stride ** 2) == 0, "Number of images generated from strided subsample of the image needs to be \n" \
                                             "a positive integer. Change stride such that : \n" \
                                             "(img_size ** 2) / (stride ** 2) is a positive integer."

input_path = r"input_images/"
validation_path = r"val_images/"

validation_set5_path = validation_path + "set5/"
validation_set14_path = validation_path + "set14/"

base_dataset_dir = os.path.expanduser("~") + "/Image Super Resolution Dataset/"

output_path = base_dataset_dir + "train_images/train/"
validation_output_path = base_dataset_dir + r"train_images/validation/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

def transform_images(directory, output_directory, scaling_factor=2, max_nb_images=-1, true_upscale=False):
    index = 1

    if not os.path.exists(output_directory + "X/"):
        os.makedirs(output_directory + "X/")

    if not os.path.exists(output_directory + "y/"):
        os.makedirs(output_directory + "y/")

    # For each image in input_images directory
    nb_images = len([name for name in os.listdir(directory)])

    if max_nb_images != -1:
        print("Transforming %d images." % max_nb_images)
    else:
        assert max_nb_images <= nb_images, "Max number of images must be less than number of images in path"
        print("Transforming %d images." % (nb_images))

    if nb_images == 0:
        print("Extract the training images or images from imageset_91.zip (found in the releases of the project) "
              "into a directory with the name 'input_images'")
        print("Extract the validation images or images from set5_validation.zip (found in the releases of the project) "
              "into a directory with the name 'val_images'")
        exit()

    for file in os.listdir(directory):
        img = imread(directory + file, mode='RGB')

        # Resize to 256 x 256
        img = imresize(img, (img_size, img_size))
        img=scipy.misc.imfilter(img,ftype='sharpen')
        # Create patches
        hr_patch_size = (16 * scaling_factor * _image_scale_multiplier)
        nb_hr_images = (img_size ** 2) // (stride ** 2)

        hr_samples = np.empty((nb_hr_images, hr_patch_size, hr_patch_size, 3))

        image_subsample_iterator = subimage_generator(img, stride, hr_patch_size, nb_hr_images)

        stride_range = np.sqrt(nb_hr_images).astype(int)

        i = 0
        for j in range(stride_range):
            for k in range(stride_range):
                hr_samples[i, :, :, :] = next(image_subsample_iterator)
                i += 1

        lr_patch_size = 16 * _image_scale_multiplier

        t1 = time.time()
        # Create nb_hr_images 'X' and 'Y' sub-images of size hr_patch_size for each patch
        for i in range(nb_hr_images):
            ip = hr_samples[i]
            # Save ground truth image X
            imsave(output_directory + "/y/" + "%d_%d.png" % (index, i + 1), ip)

            # Apply Gaussian Blur to Y
            op = gaussian_filter(ip, sigma=0.5)
            print("CVVVVVVVVVVVVVVVV")
            #ip = np.array(ip, dtype=np.uint8)
            #op = cv2.bilateralFilter(ip,15,125,125)

            # Subsample by scaling factor to Y
            op = imresize(op, (lr_patch_size, lr_patch_size), interp='bicubic')

            if not true_upscale:
                # Upscale by scaling factor to Y
                op = imresize(op, (hr_patch_size, hr_patch_size), interp='bicubic')

            # Save Y
            imsave(output_directory + "/X/" + "%d_%d.png" % (index, i+1), op)

        print("Finished image %d in time %0.2f seconds. (%s)" % (index, time.time() - t1, file))
        index += 1

        if max_nb_images > 0 and index >= max_nb_images:
            print("Transformed maximum number of images. ")
            break

    print("Images transformed. Saved at directory : %s" % (output_directory))


def image_count():
    return len([name for name in os.listdir(output_path + "X/")])


def val_image_count():
    return len([name for name in os.listdir(validation_output_path + "X/")])


def subimage_generator(img, stride, patch_size, nb_hr_images):
    for _ in range(nb_hr_images):
        for x in range(0, img_size - patch_size, stride):
            for y in range(0, img_size - patch_size, stride):
                subimage = img[x : x + patch_size, y : y + patch_size, :]

                yield subimage



def subimage_patch(img, stride, patch_size, nb_hr_images):
    heightini, widthini = img.shape[:2]
    #print(str(heightini)+'--'+str(widthini))
    #j=0
    for y in range(0, widthini , stride):
        #for y in range(0, heightini - patch_size, stride):
        for x in range(0, heightini , stride):          
            if (x + patch_size)<widthini and (y + patch_size) <heightini:
                subimage = img[y : y + patch_size, x : x + patch_size, :]
                #height, width = subimage.shape[:2]
                #print(str(height)+'<<-->>'+str(width))
                #print(str(x)+'--'+str(y)+'--'+str(x + patch_size)+'--'+str(y + patch_size))
                #j += 1
                yield subimage

def make_patches(x, scale, patch_size, upscale=True, verbose=1):
    '''x shape: (num_channels, rows, cols)'''
    height, width = x.shape[:2]
    img_height =width * scale
    img_width  = height * scale
    #x = imresize(x, (int(img_width/1.1), int(img_height/1.1) )) 
    #imsave("intermediate.jpg", x)
    #x = imresize(x, (img_width,img_height))
    #print("PATCH   SIZE SIZE")
    #imsave("intermediateafter.jpg", x)
    #if upscale: x = imresize(x, (height * scale, width * scale), interp='bicubic')
    #if upscale: x = imresize(x, (height * scale, width * scale))
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches

def make_patchesOrig(x, scale, patch_size, upscale=False, verbose=1):
    '''x shape: (num_channels, rows, cols)'''
    height, width = x.shape[:2]
    if upscale: x = imresize(x, (height * scale, width * scale))
    patches = extract_patches_2dv2(x, (patch_size, patch_size))
    return patches


def make_patchesStep(x, scale, patch_size, upscale=False,extraction_step=24, verbose=1):
    '''x shape: (num_channels, rows, cols)'''
    height, width = x.shape[:2]
    if upscale: x = imresize(x, (height * scale, width * scale))
    patches = extract_patches_Step(x, (patch_size, patch_size),extraction_step)
    return patches    

def combine_patches(in_patches, out_shape, scale):
    '''Reconstruct an image from these `patches`'''
    print("wpatch")
    recon = reconstruct_from_patches_2d(in_patches, out_shape)
    return recon
from itertools import product
def reconstruct_from_patches_2dloc(patches, image_size):
    """Reconstruct the image from all of its patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    patches : array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.
    image_size : tuple of ints (image_height, image_width) or
        (image_height, image_width, n_channels)
        the size of the image that will be reconstructed
    Returns
    -------
    image : array, shape = image_size
        the reconstructed image
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(min(i + 1, p_h, i_h - i) *
                               min(j + 1, p_w, i_w - j))
    return img
    # generate an overlap count map directly, this is fast
    Y, X = np.ogrid[0:i_h, 0:i_w]
    x_h = min(p_w, i_w - p_w)
    y_h = min(p_h, i_h - p_h)
    overlap_cnt = ((np.minimum(np.minimum(X+1,x_h), np.minimum(i_w-X,x_h)))
                  *(np.minimum(np.minimum(Y+1,y_h), np.minimum(i_h-Y,y_h))),1)
    return img/overlap_cnt

def subimage_build_patch_global(img, stride, patch_size, nb_hr_images):
    heightini, widthini = img.shape[:2]
    print("///////////------")
    print(img.shape)
    #print(str(heightini)+'--'+str(widthini))
    i=0
    for y in range(0, widthini , stride):
        #for y in range(0, heightini - patch_size, stride):
        for x in range(0, heightini , stride):          
            if (x + patch_size)<widthini and (y + patch_size) <heightini:

                i += 1
    subimages= np.empty((i, patch_size, patch_size, 3))
    j=0
    for y in range(0, widthini , stride):
        #for y in range(0, heightini - patch_size, stride):
        for x in range(0, heightini , stride):          
            if (x + patch_size)<widthini and (y + patch_size) <heightini:
                subimages[j, :, :, :] = img[y : y + patch_size, x : x + patch_size, :]
                #height, width = subimage.shape[:2]
                #print(str(height)+'<<-->>'+str(width))
                #print(str(x)+'--'+str(y)+'--'+str(x + patch_size)+'--'+str(y + patch_size))
                j += 1
                #yield subimage
    print(i)             
    return subimages              
           

def subimage_combine_patches_global(imgtrue , patches, stride, patch_size, scale):
    heighttrue, widthtrue = imgtrue.shape[:2]
    img = imresize(imgtrue, (heighttrue*scale, widthtrue*scale), interp='bicubic')
    heightini, widthini = img.shape[:2]
    print("///////////------")
    print(img.shape)
    print(patches.shape)
    j=0
    for y in range(0, widthini , stride):
        #for y in range(0, heightini - patch_size, stride):
        for x in range(0, heightini , stride):          
            if (x + patch_size)<widthini and (y + patch_size) <heightini:
                #subimages[j, :, :, :] = img[y : y + patch_size, x : x + patch_size, :]
                img[y : y + patch_size, x : x + patch_size, :]=patches[j, :, :, :]
                #height, width = subimage.shape[:2]
                #print(str(height)+'<<-->>'+str(width))
                #print(str(x)+'--'+str(y)+'--'+str(x + patch_size)+'--'+str(y + patch_size))
                j += 1
    print(j)              
    return img 


def image_generator(directory, scale_factor=2, target_shape=None, channels=3, small_train_images=False, shuffle=True,
                    batch_size=32, seed=None):
    if not target_shape:
        if small_train_images:
            if K.image_dim_ordering() == "th":
                image_shape = (channels, 16 * _image_scale_multiplier, 16 * _image_scale_multiplier)
                y_image_shape = (channels, 16 * scale_factor * _image_scale_multiplier,
                                 16 * scale_factor * _image_scale_multiplier)
            else:
                image_shape = (16 * _image_scale_multiplier, 16 * _image_scale_multiplier, channels)
                y_image_shape = (16 * scale_factor * _image_scale_multiplier,
                                 16 * scale_factor * _image_scale_multiplier, channels)
        else:
            if K.image_dim_ordering() == "th":
                image_shape = (channels, 16 * scale_factor * _image_scale_multiplier, 16 * scale_factor * _image_scale_multiplier)
                y_image_shape = image_shape
            else:
                image_shape = (16 * scale_factor * _image_scale_multiplier, 16 * scale_factor * _image_scale_multiplier,
                               channels)
                y_image_shape = image_shape
    else:
        if small_train_images:
            if K.image_dim_ordering() == "th":
                y_image_shape = (3,) + target_shape

                target_shape = (target_shape[0] * _image_scale_multiplier // scale_factor,
                                target_shape[1] * _image_scale_multiplier // scale_factor)
                image_shape = (3,) + target_shape
            else:
                y_image_shape = target_shape + (channels,)

                target_shape = (target_shape[0] * _image_scale_multiplier // scale_factor,
                                target_shape[1] * _image_scale_multiplier // scale_factor)
                image_shape = target_shape + (channels,)
        else:
            if K.image_dim_ordering() == "th":
                image_shape = (channels,) + target_shape
                y_image_shape = image_shape
            else:
                image_shape = target_shape + (channels,)
                y_image_shape = image_shape

    file_names = [f for f in sorted(os.listdir(directory + "X/"))]
    X_filenames = [os.path.join(directory, "X", f) for f in file_names]
    y_filenames = [os.path.join(directory, "y", f) for f in file_names]

    nb_images = len(file_names)
    print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)
        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            x_fn = X_filenames[j]
            img = imread(x_fn, mode='RGB')
            #print(img.shape)
            if small_train_images:
                img = imresize(img, (16 * _image_scale_multiplier, 16 * _image_scale_multiplier))
            img = img.astype('float32') / 255.
            #print(img.shape)
            #print(_image_scale_multiplier)
            if K.image_dim_ordering() == "th":
                batch_x[i] = img.transpose((2, 0, 1))
            else:
                batch_x[i] = img

            y_fn = y_filenames[j]
            img = imread(y_fn, mode="RGB")
            img = img.astype('float32') / 255.

            if K.image_dim_ordering() == "th":
                batch_y[i] = img.transpose((2, 0, 1))
            else:
                batch_y[i] = img
                #print(img.shape)
            #print("batch_x.shape")
            #print(batch_x.shape)
        yield (batch_x, batch_y)

def _index_generator(N, batch_size=32, shuffle=True, seed=None):
    batch_index = 0
    total_batches_seen = 0

    while 1:
        if seed is not None:
            np.random.seed(seed + total_batches_seen)

        if batch_index == 0:
            index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)

        current_index = (batch_index * batch_size) % N

        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
        total_batches_seen += 1

        yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)


def smooth_gan_labels(y):
    assert len(y.shape) == 2, "Needs to be a binary class"
    y = np.asarray(y, dtype='int')
    Y = np.zeros(y.shape, dtype='float32')

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i, j] == 0:
                Y[i, j] = np.random.uniform(0.0, 0.3)
            else:
                Y[i, j] = np.random.uniform(0.7, 1.2)

    return Y

def SetGama(imgParam,gamma=0.1):
    im=np.array(imgParam)
    height, width = im.shape[:2]
    gammaCorrection = 1 / gamma
    for y in range(0,width):
        for x in range (0,height):
            colour = im[x, y] #[red, green, blue]
            newRed   = 255 * (colour[0].astype(np.float32)   / 255) ** gammaCorrection
            newGreen = 255 * (colour[1].astype(np.float32)  / 255) ** gammaCorrection
            newBlue  = 255 * (colour[2].astype(np.float32)   / 255) ** gammaCorrection
            im[x, y] = [newRed, newGreen, newBlue]
    return im    


def SetContrast(im,contrast=128):
    height, width = im.shape[:2]
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    for y in range(0,width):
        for x in range (0,height):
            colour = im[x, y] #[red, green, blue]
            newRed   = Truncate(factor * (colour[0].astype(np.float32)   - 128) + 128)
            newGreen = Truncate(factor * (colour[1].astype(np.float32) - 128) + 128)
            newBlue  = Truncate(factor * (colour[2].astype(np.float32)  - 128) + 128)
            im[x, y] = [newRed, newGreen, newBlue]

    return im 

def reconstruct_from_patches_2dlocal(patches,patchcnn, image_size,step=16):

    countstep_i=0
    countstep_j=0
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    imgmap = np.zeros(image_size) #+1
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1

    cnt=0

    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        if i % step==0 and j %step==0:
            #img[i:i + p_h, j:j + p_w] =patchcnn[cnt]
            cnt+=1
  
    cnt=0
    pad=4
    #print("Number of patches  = %d, Patch Shape W H= (%d, %d)" % (patches.shape[0], n_h, n_w))
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
    	#img[i:i + p_h, j:j + p_w] += p
    	if i % step==0 and j %step==0:
            #print("i j  = (%d , %d)" % (i,j))
            #img[i:i + p_h, j:j + p_w] =patchcnn[cnt]
            if i >0 and j>0 and i <n_h-1 and j<n_w-1:
                pa=patchcnn[cnt]    
                #img[i+pad:i + p_h-pad, j+pad:j + p_w-pad] =pa[pad:p_h-pad , pad:p_w-pad]
                img[i+pad:i + p_h-pad, j+pad:j + p_w-pad] +=pa[pad:p_h-pad , pad:p_w-pad]
                imgmap[i+pad:i + p_h-pad, j+pad:j + p_w-pad] +=1
                #print("I  J h w= (%d , %d) (%d , %d)" % (i,j,n_h,n_w))
        	   #img[i:i + p_h, j:j + p_w] = p  

        	   
            else:
                #print("ALL I  J h w= (%d , %d) (%d , %d)" % (i,j,n_h,n_w))
                img[i:i + p_h, j:j + p_w] +=patchcnn[cnt]
                imgmap[i:i + p_h, j:j + p_w] +=1
            cnt+=1
        	#print("i and j  = (%d, %d)" % (i, j))
        	#countstep_i+=1
        	#countstep_j+=1
    	#else:
        #	img[i:i + p_h, j:j + p_w] +=img[i:i + p_h, j:j + p_w]
    #print (cnt)
    print("nh  nw  = (%d , %d)" % (n_h,n_w))
    #return img 
   

    cnt_i=0
    cnt_j=0   

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            #if i % 10==0 and j %10==0:
            #if i % step==0 and j %step==0:
            	#img[i, j] /= 1
            #print(img[i, j])
            img[i, j] /= imgmap[i, j]
            	#print("i + 1, p_h, i_h - i  = (%d, %d, %d), j + 1, p_w, i_w - j= (%d, %d, %d)" % (i + 1, p_h, i_h - i,j + 1,p_w ,i_w - j ))  
            #img[i, j] /= float(min(i + 1, p_h, i_h - i) * min(j + 1, p_w, i_w - j))
            #print("factor  = (%d)" % (float(min(i + 1, p_h, i_h - i) * min(j + 1, p_w, i_w - j))))  
            cnt_j+=1
        cnt_i+=1
        cnt_j =0
    return img

def extract_patches_2dlocal(image,patches, patch_size, step=None):
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")



    #i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image.shape)
    img=img-1
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    nb_patch_new=0
    
    #print("Number of patches  = %d, Patch Shape W H= (%d, %d)" % (patches.shape[0], n_h, n_w))
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
    	#img[i:i + p_h, j:j + p_w] += p
    	if i % step==0 and j %step==0:
        	img[i:i + p_h, j:j + p_w] = p
        	#print("i and j  = (%d, %d)" % (i, j))
        	nb_patch_new+=1
        	
    new_patch= np.zeros( (nb_patch_new,p_h,p_w,3))
    nb_patch_cnt=0 

    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
    	#img[i:i + p_h, j:j + p_w] += p
    	if i % step==0 and j %step==0:
        	new_patch[nb_patch_cnt] = p        	
        	nb_patch_cnt+=1



    print (nb_patch_new)
    return new_patch 

from sklearn.feature_extraction.image import check_array,extract_patches,_compute_n_patches


def extract_patches_2dv2(image, patch_size, max_patches=None, random_state=None):

    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = extract_patches(image.astype('uint8'),
                                        patch_shape=(p_h, p_w, n_colors),
                                        extraction_step=1)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    print(n_patches)
    print(np.intp)
    extracted_patches=None
    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches

def extract_patches_Step(image, patch_size, step_patches=24):
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")
    print("PATCH SIZE")
    print(patch_size)
    cnt=0
    cnt_h=0
    cnt_w=0
    patches=[]

    border_crop=2
    
    #new_patch= np.zeros( (p_h,p_w,3))
    for w in range(i_w-p_w):
        if w==0 or w % step_patches==0 :
            if w==0:
                crop_w=0
            else:
                crop_w=0
            cnt_h=0
            for h in range(i_h-p_h):
                if h==0 or h % step_patches==0 :
                    if h==0:
                        crop_h=0
                    else:
                        crop_h=0

                    #patchul=image[h:h + p_h, w:w + p_w] 
                    patchul=image[h-crop_h:h + p_h +crop_h , w -crop_w :w + p_w +crop_w ] 

                    #patches[cnt]=patchul
                    patches.append(patchul)
                    #imgmap[cnt_h,cnt_w]=cnt
                    #print("Number of patches  = %d, Patch Shape h w= (%d, %d)" % (cnt, patchul.shape[0], patchul.shape[1]))
                    cnt+=1

                    cnt_h+=1


            cnt_w+=1        
    print(cnt_h)
    i=0
    out_patch= np.zeros( (cnt,p_h,p_w,3))
    for p in patches:
        out_patch[i]=p
        i+=1


    #return patches , (cnt_h,cnt_w)

    #return
    #Rebuild part
    cnt_bild=0
    #imgageRebuild = np.zeros( (i_h+p_h,i_w+p_w,3))
    imgageRebuild = np.zeros( (i_h,i_w,3))
    for w in range(cnt_w):
        for h in range(cnt_h):
            a=imgageRebuild[ h*step_patches:h*step_patches + p_h, w*step_patches:w*step_patches + p_w   ] 
            #print("pozitie h w of patches  = (%d,  %d) , Patch Shape h w= (%d, %d)" % (h*step_patches  ,w*step_patches , a.shape[0], a.shape[1]))
            print(a.shape)
            print(cnt_bild)
            imgageRebuild[ h*step_patches:h*step_patches + p_h, w*step_patches:w*step_patches + p_w   ] =out_patch[cnt_bild]
            cnt_bild+=1


    imsave('/home/www/imgsuper/val_images/test.png', imgageRebuild)

    return out_patch , (cnt_h,cnt_w)

    #B = view_as_windows(image, window_shape) 
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]
    patch_shape=(p_h, p_w, n_colors)
    print(image.ndim)
    print(image.shape)

    B = view_as_windows(image, patch_shape,step_patches)  
    print(B.shape) 
    patches = B.reshape(-1, p_h, p_w, n_colors) 
    print('----')
    print(patches.shape)
    return patches

def rebuild_from_patches_Step(img_initial,patches, patch_size,tupleinit, scale,step_patches_ini=24):
    cnt_bild=0
    i_h, i_w = img_initial.shape[:2]
    p_h_ini, p_w_ini = patch_size
    p_h=p_h_ini*scale
    p_w=p_w_ini*scale
    cnt_h, cnt_w = tupleinit
    step_patches=step_patches_ini*scale
    border_crop=8
    #imgageRebuild = np.zeros( (i_h+p_h,i_w+p_w,3))
    imgageRebuild = np.zeros( (i_h*scale,i_w*scale,3))
    for w in range(cnt_w):
        if w==0:
            crop_w=0
        else:
            crop_w=border_crop
        for h in range(cnt_h):
            if h==0:
                crop_h=0
            else:
                crop_h=border_crop
                
            a=imgageRebuild[ h*step_patches:h*step_patches + p_h, w*step_patches:w*step_patches + p_w   ] 
            localpatch=patches[cnt_bild]
            #print("pozitie h w of patches  = (%d,  %d) , Patch Shape h w= (%d, %d)" % (h*step_patches  ,w*step_patches , a.shape[0], a.shape[1]))
            print(a.shape)
            print(cnt_bild)
            #imgageRebuild[ h*step_patches:h*step_patches + p_h, w*step_patches:w*step_patches + p_w   ] =localpatch
            localpatch_croped=localpatch[crop_h : p_h-crop_h ,  crop_w : p_w -crop_w ]            
            imgageRebuild[ h*step_patches+crop_h : h*step_patches + p_h-crop_h, w*step_patches+crop_w :w*step_patches + p_w -crop_w  ] =localpatch_croped
            cnt_bild+=1

    return imgageRebuild        


if __name__ == "__main__":
    # Transform the images once, then run the main code to scale images

    # Change scaling factor to increase the scaling factor
    scaling_factor = 2

    # Set true_upscale to True to generate smaller training images that will then be true upscaled.
    # Leave as false to create same size input and output images
    true_upscale = False
    #true_upscale = True

    transform_images(input_path, output_path, scaling_factor=scaling_factor, max_nb_images=-1,
                     true_upscale=true_upscale)
    transform_images(validation_set5_path, validation_output_path, scaling_factor=scaling_factor, max_nb_images=-1,
                     true_upscale=true_upscale)
    pass
