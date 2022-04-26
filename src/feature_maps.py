import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model


class FeaturesExtraction(object):
    def __init__(self,model,layername):
        """Extracting features from a model at a given layer.

        Args:
            model (tf.keras.Model): keras model
            layername (str): name of layer to extract features from
        """
        self.model = model
        self.layername = layername
        
        self.feature_model = Model(inputs=model.inputs,outputs=model.get_layer(layername).output)
    
    def extract_features(self, img):
        return self.feature_model.predict(img)
    
def vis_feature_map(feature_maps, channel_cams=None, size_factor=1.5, alpha=0.8):
    if feature_maps.shape[3] < 8:
        ncols = feature_maps.shape[3]
        nrows = 1
        figsize = (size_factor*ncols, size_factor)
    else:
        ncols = min(8,int(np.floor(np.sqrt(feature_maps.shape[3]))))
        nrows = ncols
        figsize = (size_factor*ncols,ncols*size_factor)
    
    fig = plt.figure(figsize=figsize)
    axes = ImageGrid(
        fig, 111,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.05,
        share_all=True
    )
    
    for i in range(feature_maps.shape[3]):
        axes[i].imshow(feature_maps[0,:,:,i], cmap="gray")
        axes[i].axis("off")
        if channel_cams is not None:
            axes[i].imshow(channel_cams[0,:,:,i], alpha=alpha, vmin=0, vmax=np.max(channel_cams))