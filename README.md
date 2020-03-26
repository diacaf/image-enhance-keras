##  Image-enhance / scaleX4 -keras<br><br>
Image super-resolution (restoration of rich details in a low resolution image) <br><br>

##  Setup

Supports Keras with  Tensorflow backend.<br> 
By default GPU use(recommended) <br>
For CPU only please uncomment in main_dirpath.py 3 rd line <br> ` #os.environ['CUDA_VISIBLE_DEVICES'] = '-1' `

Model weights http://epsilonsys.com/weights025-17-0.93.h5<br>
Train data https://data.vision.ee.ethz.ch/cvl/DIV2K/

## Usage
The model weights can be downloaded from http://epsilonsys.com/weights025-17-0.93.h5 to /weights_Double/ folder :<br>
`python main_dirpath.py "imgpath"`, where imgpath is a full path to the images folder.



##  Our Quantitative results on Set5 with X4 computed on Y from YCbCr
SSIM  0.904<br>
Best SSIM (NTIRE 2017) 0.901<br>

##  Our Quantitative results on Set5 with X4 computed on RGB
SSIM  0.879<br>
Best SSIM (NTIRE 2017) 0.874<br>

Official SSIM score from  NTIRE 2017 :<br>
http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017suppl.pdf


