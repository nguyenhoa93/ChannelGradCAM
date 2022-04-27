import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from skimage.transform import resize
from tensorflow.keras.applications.resnet50 import decode_predictions

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
    
    
class LayerCAM(object):
    def __init__(self, model, layername, inv_maps = {0: "cat", 1: "dog"}):
        self.model  = model
        self.layername = layername
        self.inv_maps = inv_maps
        
    def compute_heatmap(self, im, classIdx=None):
        laycamModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layername).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(im, tf.float32)
            (convOuts, preds) = laycamModel(inputs)  # preds after softmax
            if classIdx is None:
                classIdx = np.argmax(preds)
                if type(self.inv_maps) == dict:
                    print("Predicted class: {}".format(self.inv_maps[classIdx]))
                else:
                    print("Predicted class: {}".format(decode_predictions(preds)))
            loss = preds[:, classIdx]
            
        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # weights
        weights = tf.keras.activations.relu(grads)
        A = convOuts * weights
        
        M = tf.keras.activations.relu(tf.math.reduce_sum(A, axis=3)).numpy()
        
        cams = []
        for i in range(M.shape[0]):
            resized_cam = resize(M[i,:,:], output_shape=(im.shape[1], im.shape[2]))
            cams.append(resized_cam)
            
        channel_cams = []
        A = tf.keras.activations.relu(A)
        for i in range(A.shape[0]):
            X = np.empty((im.shape[1], im.shape[2], A.shape[3]))
            for j in range(A.shape[3]):
                X[:,:,j] = resize(A[i,:,:,j], output_shape=(im.shape[1], im.shape[2]))
            
            channel_cams.append(X)
        
        
        return np.array(cams), np.array(channel_cams)

class ChannelGradCAM(object):
    def __init__(self, model):
        self.model = model
        self.channel_model = self.__create_channel_model()
    
    def __create_channel_model(self):
        inputs = tf.keras.layers.Input((224,224,3), name="input")
        channels = tf.keras.layers.Conv2D(3, (1,1), name="channel_conv")(inputs)
        
        ori_model =  Model(inputs=self.model.layers[1].input, outputs=self.model.layers[-1].output)
        output = ori_model(channels)
        
        channel_model = Model(inputs=[inputs], outputs=[output])
        
        # Set weights for the channel_conv layer to output the identical input
        weights = np.zeros((1,1,3,3))
        weights[:,:,0,0] = 1.
        weights[:,:,1,1] = 1.
        weights[:,:,2,2] = 1.
        
        channel_model.get_layer("channel_conv").set_weights((weights, np.array([0.,0.,0.])))
        
        return channel_model
    
    def compute_heatmap(self, im):
        cams, channel_cams = LayerCAM(self.channel_model, "channel_conv").compute_heatmap(im)
        
        return cams, channel_cams
    
    def extract_features(self, im):
        return FeaturesExtraction(self.channel_model, "channel_conv").extract_features(im)
        
         
        
        