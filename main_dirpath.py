import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import models

parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
parser.add_argument("imgpath", type=str, help="Path to input image")
parser.add_argument("--model", type=str, default="didbl", help="Use either image super resolution (sr), "
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
assert model_type in ["didbl"], 'Model type must be either "sr", "esr", "dsr", ' \
                                                           '"ddsr" or "rnsr"'

mode = str(args.mode).lower()
assert mode in ['fast', 'patch'], 'Mode of operation must be either "fast" or "patch"'

scale_factor = int(args.scale)
save = strToBool(args.save)

patch_size = int(args.patch_size)
assert patch_size > 0, "Patch size must be a positive integer"

if model_type == "Difvdsr":
    model = models.Difvdsr(scale_factor)   
elif model_type == "difv4":
    model = models.Difvdsr4(scale_factor)       
elif model_type == "didbl":
    model = models.DifvdsrDouble(scale_factor)                 
   
else:
    model = models.DeepDenoiseSR(scale_factor)


for file in os.listdir(path):    
    pathfile=path + file

    model.upscaleStepPatch(pathfile, save_intermediate=save,scalemulti=4,  patch_size=96, suffix=suffix)
    #model.upscalePatch(pathfile, save_intermediate=save,scalemulti=4,  patch_size=128, suffix=suffix)
    #model.upscale(pathfile, save_intermediate=save, mode=mode, patch_size=64, suffix=suffix)
