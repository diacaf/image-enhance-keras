from __future__ import print_function, division

#rom keras.utils.visualize_util import plot
import models
import img_utils

if __name__ == "__main__":
    path = r""
    val_path = "val_images/"

    scale = 1








    ddsr = models.DifvdsrDouble(scale)
    ddsr.create_model()
    ddsr.fit(nb_epochs=180)    
