# Image-enhance / scaleX4 -keras<br><br>
Image super-resolution (restoration of rich details in a low resolution image) <br><br>
Setup

Supports Keras with  Tensorflow backend.<br> 
By default GPU use
For CPU only please uncomment in main_dirpath.py 3 rd line <br> ` #os.environ['CUDA_VISIBLE_DEVICES'] = '-1' `

Model weights http://epsilonsys.com/weights025-17-0.93.h5<br>

## Usage
The model weights can be downloaded from http://epsilonsys.com/weights025-17-0.93.h5 to /weights_Double/ folder :<br>
`python main_dirpath.py "imgpath"`, where imgpath is a full path to the images folder.
