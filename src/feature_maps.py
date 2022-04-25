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
    
def vis_feature_map(feature_maps):
    ncol = min(8,int(np.floor(np.sqrt(feature_maps.shape[3]))))
    fig, ax = plt.subplots(ncol, ncol,figsize=(1.5*ncol,ncol*1.5))
    if ncol == 1:
        ax.imshow(feature_maps[0,:,:,0],cmap="gray")
    else:
        count = 0
        for i in range(ncol):
            for j in range(ncol):
                ax[j,i].imshow(feature_maps[0,:,:,count],cmap="gray")
                ax[j,i].axis("off")
                count += 1
    plt.subplots_adjust(wspace=0.1, hspace=0.1);
    plt.show()