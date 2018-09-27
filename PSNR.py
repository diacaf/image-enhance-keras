
import numpy as np
import math


#https://github.com/Jongchan/tensorflow-vdsr/blob/master/PSNR.py
def psnrVDSR(target, ref, scale):
	#assume RGB image
	target_data = np.array(target)
	target_data = target_data[scale:-scale, scale:-scale]

	ref_data = np.array(ref)
	ref_data = ref_data[scale:-scale, scale:-scale]
	
	diff = ref_data - target_data
	diff = diff.flatten('C')
	rmse = math.sqrt( np.mean(diff ** 2.) )
	return 20 * math.log10(255.0 / rmse)
	#return 20*math.log10(1.0/rmse)



#https://github.com/twtygqyy/pytorch-vdsr/blob/master/eval.py	
def PSNRTorch(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse) 
    


def psnrSVLAB(img1, img2):
    #local psnr = -10 * math.log10(diffShave:pow(2):mean())
    #mse = np.mean( (img1 - img2) ** 2 )
    #img1=im2double(np.squeeze(img1))
    #img2=im2double(np.squeeze(img2))
    #img1=rgb2y(img1)
    #img2=rgb2y(img2)
    img1=im2double(img1)
    img2=im2double(img2)
    mse = (np.mean((img1 - img2) ** 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return -10 * math.log10(mse)




def psnrNITRE(pred, gt, shave_border=0) : 
    height, width = pred.shape[:2]
    #print("height")
    #print(width)
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border] 



    if np.amax(pred) > 1:  
    	#print('im2')  
    	#pred = np.clip(pred, 16, 235)
    	#pred = np.rint(pred )
    	#pred = np.clip(pred, 16, 235)
    	pred=im2double(pred)

    if np.amax(gt) > 1:    
    	gt=im2double(gt)
    	#print('im1')  	

    imdff = pred - gt 
    rmse = math.sqrt(np.mean(imdff ** 2))
    #return 20 * math.log10(255.0 / rmse) 
    N = imdff.size  
    #print("NNNNNN")
    #print(N)
    #imdff = imdff.flatten('C')
    sumel=np.sum(imdff ** 2)
    #print('SUMMM')
    #print(sumel)
    return 10*math.log10(N / sumel) 
    #local psnr = -10 * math.log10(diffShave:pow(2):mean())

def im2doubleZ(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def im2double(im):

    #info = np.iinfo(im.dtype) # Get the data type of the input image
    #return im.astype(np.float) / info.max
    #return im
    return im.astype(np.float) / 255.0


def rgb2y(img): 
    #y = 16 + (65.481 * img[1]) + (128.553 * img[2]) + (24.966 * img[3])  
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
     # Y
    y[:,:,0] = 16 + (65.481 * r) + (128.553 * g) + (24.966 * b)  
    #cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    return y     

