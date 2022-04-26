import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Model
import tensorflow as tf

from src.datagen import HybridDataGen, PureDatagen

def build_model():
    resnet = ResNet50(include_top=False, pooling="avg", weights="imagenet")

    for layer in resnet.layers:
        layer.trainable=True
        
    logits = tf.keras.layers.Dense(2)(resnet.layers[-1].output)
    output = tf.keras.layers.Activation('softmax')(logits)
    model = Model(resnet.input, output)
    
    return model

parser = argparse.ArgumentParser("TRAINING CLASSIFICATION BASED ON CHANNEL 1")
parser.add_argument("--imdir", type=str, default="assets/dogs_and_cats/train", help="path to image directory")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")

args = parser.parse_args()

SEED = 42

if __name__=="__main__":
    # Prepare data
    filenames = sorted([x for x in os.listdir(args.imdir) if not x.startswith(".")])
    labels = [x.split(".")[0] for x in filenames]
    df = pd.DataFrame({"filename": filenames, "label": labels})
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.label)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df.label)

    train_df.sample(frac=1, random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    val_df.sample(frac=1, random_state=SEED)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print("Found {} samples: \n{}".format(len(df), df.label.value_counts()))
    print("Train: {}\nVal: {}\nTest: {}".format(train_df.label.value_counts(), train_df.label.value_counts(), train_df.label.value_counts()))
    
    # Data generators
    train_gen = HybridDataGen(imdir=args.imdir,
                              df=train_df,
                              stage="train",
                              preprocess=False,
                              batch_size=32)
    val_gen = HybridDataGen(imdir=args.imdir,
                            df=val_df,
                            stage="val",
                            preprocess=False,
                            batch_size=32)
    
    eval_gen = HybridDataGen(imdir=args.imdir,
                             df=test_df,
                             preprocess=False,
                             batch_size=32,
                             stage="eval")
    pure_eval_gen = PureDatagen(imdir=args.imdir,
                                df=test_df,
                                preprocess=False,
                                batch_size=32,
                                stage="eval")
    
    # Build model
    model = build_model()
    
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    
    sgd = tf.keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer=sgd, loss = "categorical_crossentropy", metrics=["accuracy"])
    
    # train
    model.fit(train_gen,
              epochs=args.epochs,
              validation_data=val_gen,
              callbacks=[cb])
    
    model.save("assets/model.h5")
    
    # Evaluate
    print("Evaluate hybrid test set")
    model.evaluate(eval_gen)
    print("Evaluate pure test set")
    model.evaluate(pure_eval_gen)
    
    