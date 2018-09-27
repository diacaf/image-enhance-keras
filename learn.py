from __future__ import print_function, division

#rom keras.utils.visualize_util import plot
import models
import img_utils

if __name__ == "__main__":
    path = r""
    val_path = "val_images/"

    scale = 1


    """
    Plot the models
    """

    # model = models.ImageSuperResolutionModel(scale).create_model()
    # plot(model, to_file="architectures/SRCNN.png", show_shapes=True, show_layer_names=True)

    # model = models.ExpantionSuperResolution(scale).create_model()
    # plot(model, to_file="architectures/ESRCNN.png", show_layer_names=True, show_shapes=True)

    # model = models.DenoisingAutoEncoderSR(scale).create_model()
    # plot(model, to_file="architectures/Denoise.png", show_layer_names=True, show_shapes=True)

    #model = models.DeepDenoiseSR(scale).create_model()
    # plot(model, to_file="architectures/Deep Denoise.png", show_layer_names=True, show_shapes=True)

    # model = models.ResNetSR(scale).create_model()
    # plot(model, to_file="architectures/ResNet.png", show_layer_names=True, show_shapes=True)

    # model = models.GANImageSuperResolutionModel(scale).create_model(mode='train')
    # plot(model, to_file='architectures/GAN Image SR.png', show_shapes=True, show_layer_names=True)




    """
    Train DenoisingAutoEncoderSR
    """

    #dsr = models.DenoisingAutoEncoderSR(scale)
    #dsr.create_model()
    #dsr.fit(nb_epochs=250) 

    """
    Train Deep Denoise SR
    """

    #ddsr = models.DeepDenoiseSR(scale)
    #ddsr.create_model()
    #ddsr.fit(nb_epochs=180)



    """
    Train Deep MEDenoisingSR
    """

    #Mdme = models.MEDenoisingSR(scale)
    #Mdme.create_model()
    #Mdme.fit(nb_epochs=180)

    """
    Train Deep DenoisingMe
    """

    #dme = models.DenoisingMe(scale)
    #dme.create_model()
    #dme.fit(nb_epochs=180)
    """
    Train Deep Denoise SR
    """

    #ddsr = models.DeepDenoiseSREPS(scale)
    #ddsr.create_model()
    #ddsr.fit(nb_epochs=180)

    """
    Train Res Net SR
    """

    #rnsr = models.ResNetSR(scale)
    #rnsr.create_model()
    #rnsr.fit(nb_epochs=150)

    """
    Train GAN Super Resolution
    """

    #gsr = models.GANImageSuperResolutionModel(scale)
    #gsr.create_model(mode='train')
    #gsr.fit(nb_pretrain_samples=10000, nb_epochs=10)  



    """
    Train EfficientSubPixelConvolutionalSR
    """

    #ddsr = models.EfficientSubPixelConvolutionalSR(scale)
    #ddsr.create_model()
    #ddsr.fit(nb_epochs=180)   



    #ddsr = models.ScaleGen(scale)
    #ddsr.create_model()
    #ddsr.fit(nb_epochs=180)  


    #ddsr = models.ScaleGen32(scale)
    #ddsr.create_model()
    #ddsr.fit(nb_epochs=180)     



    #ddsr = models.ScaleGen32gen(scale)
    #ddsr.create_model()
    #ddsr.fit(nb_epochs=180)  

    ddsr = models.ScaleGenX4(scale)
    ddsr.create_model()
    ddsr.fit(nb_epochs=180)    