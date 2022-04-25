import os
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Model
import tensorflow as tf

def build_model():
    resnet = ResNet50(include_top=False, pooling="avg", weights="imagenet")

    for layer in resnet.layers:
        layer.trainable=True
        
    logits = tf.keras.layers.Dense(2)(resnet.layers[-1].output)
    output = tf.keras.layers.Activation('softmax')(logits)
    model = Model(resnet.input, output)
    
    return model

if __name__=="__main__":
    model = build_model()
    
    