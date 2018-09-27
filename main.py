import models
import argparse

parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
parser.add_argument("imgpath", type=str, help="Path to input image")
parser.add_argument("--model", type=str, default="eddsr", help="Use either image super resolution (sr), "
                        "expanded super resolution (esr), denoising auto encoder sr (dsr), "
                        "deep denoising sr (ddsr) or res net sr (rnsr)")
parser.add_argument("--scale", default=1, help='Scaling factor. Default = 2x')
parser.add_argument("--mode", default="fast", type=str, help='Mode of operation. Choices are "fast" or "patch"')
parser.add_argument("--save_intermediate", dest='save', default='False', type=str,
                        help="Whether to save bilinear upscaled image")
parser.add_argument("--suffix", default="scaled", type=str, help='Suffix of saved image')
parser.add_argument("--patch_size", type=int, default=8, help='Patch Size')

def strToBool(v):
    return v.lower() in ("true", "yes", "t", "1")

args = parser.parse_args()

path = args.imgpath
suffix = args.suffix

model_type = str(args.model).lower()
assert model_type in ["scax","xre","x53r","x53rmini","diff2","difv4" ,"difv", "scax4", "dsr", "ddsr", "rnsr","eddsr","dme","medd","sre","wei","sca","sca16","sca32","scag32","sca357"], 'Model type must be either "sr", "esr", "dsr", ' \
                                                           '"ddsr" or "rnsr"'

mode = str(args.mode).lower()
assert mode in ['fast', 'patch'], 'Mode of operation must be either "fast" or "patch"'

scale_factor = int(args.scale)
save = strToBool(args.save)

patch_size = int(args.patch_size)
assert patch_size > 0, "Patch size must be a positive integer"

if model_type == "scax":
    model = models.ScaleGenX(scale_factor)
elif model_type == "scax4":
    model = models.ScaleGenX4(scale_factor)
elif model_type == "dsr":
    model = models.DenoisingAutoEncoderSR(scale_factor)
elif model_type == "ddsr":
    model = models.DeepDenoiseSR(scale_factor)
elif model_type == "medd":
    model = models.MEDenoisingSR(scale_factor)    
elif model_type == "eddsr":
    model = models.DeepDenoiseSREPS(scale_factor)  
elif model_type == "dme":
    model = models.DenoisingMe(scale_factor)      
elif model_type == "wei":
    model = models.ModelWeight(scale_factor)
elif model_type == "sca":
    model = models.ScaleGen(scale_factor)  
elif model_type == "sca16":
    model = models.ScaleGen16(scale_factor)   
elif model_type == "sca32":
    model = models.ScaleGen32(scale_factor)   
elif model_type == "scag32":
    model = models.ScaleGen32gen(scale_factor)     
elif model_type == "sca357":
    model = models.ScaleGen357gen(scale_factor)  
elif model_type == "x53r":
    model = models.ScaleGenBIGBP(scale_factor)    
elif model_type == "x53rmini":
    model = models.ScaleGenSMALLBP(scale_factor)             
elif model_type == "rnsr":
    model = models.ResNetSR(scale_factor)   
elif model_type == "xre":
    model = models.ScaleGenX4Relu(scale_factor)    
elif model_type == "diff2":
    model = models.ScaleDiff(scale_factor)  
elif model_type == "difv":
    model = models.Difvdsr(scale_factor) 
elif model_type == "difv4":
    model = models.Difvdsr4(scale_factor)        
elif model_type == "diff234":
    model = models.Dif234(scale_factor)        
elif model_type == "sre":
    model = models.EfficientSubPixelConvolutionalSR(scale_factor)    
else:
    model = models.DeepDenoiseSR(scale_factor)

model.upscale(path, save_intermediate=save, mode=mode, patch_size=patch_size, suffix=suffix)
