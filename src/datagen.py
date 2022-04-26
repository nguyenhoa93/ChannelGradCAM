import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

class HybridDataGen(tf.keras.utils.Sequence):
    def __init__(self, imdir, 
                 df,
                 target_size=(224,224,3),
                 preprocess = False,
                 n_classes=2,
                 batch_size=32,
                 seed=42, 
                 stage="train"):
        """Generating hybrid data: channel containing target for prediction, channel 2 and 3 may contain several types: dog, cat.

        Args:
            imdir (str): path to directory with images
            df (pd.DataFrame): dataframe with filenames and labels
            target_size (tuple, optional): input size. Defaults to (224,224,3).
            preprocess (bool, optional): preprocessing with preprocess_input of keras model. Defaults to False.
            n_classes (int, optional): number of classes. Defaults to 2.
            batch_size (int, optional): batch size. Defaults to 32.
            seed (int, optional): random seed. Defaults to 42.
            stage (str, optional): stage of data ["train", "val", "eval", "test"]. Defaults to "train".
        """
        self.imdir = imdir
        self.target_size = target_size
        self.batch_size = batch_size
        self.seed = seed
        self.stage = stage
        self.n_classes = n_classes
        self.preprocess = preprocess
        
        self.df = self.create_hybrid_df(df)
        print("Found {} samples: {}".format(len(self.df), self.df.label.value_counts()))
        
        self.channel1s = self.df.channel1.values
        self.channel2s = self.df.channel2.values
        self.channel3s = self.df.channel3.values
        self.list_IDs = [i for i in range(len(self.channel2s))]
        self.labels = self.df.label.values
        if n_classes > 1:
            self.labels = tf.keras.utils.to_categorical(self.labels, num_classes=n_classes)
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.channel2s) / np.float(self.batch_size)))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        list_ids_tmp = [self.list_IDs[k] for k in indexes]
        
        return self.__generate_data(list_ids_tmp)
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.stage in ["train", "val"]:
            np.random.seed(self.seed)
            np.random.shuffle(self.indexes)
        
    @staticmethod
    def hybrate(ls1, ls2, label, seed=2022):
        np.random.seed(seed)
        shuffled1_ls1 = np.random.permutation(ls1)
        shuffled1_ls2 = np.random.permutation(ls2)
        np.random.seed(seed*2)
        shuffled2_ls1 = np.random.permutation(ls1)
        shuffled2_ls2 = np.random.permutation(ls2)
        np.random.seed(seed*3)
        shuffled3_ls1 = np.random.permutation(ls1)
        shuffled3_ls2 = np.random.permutation(ls2)
        np.random.seed(seed*4)
        shuffled4_ls1 = np.random.permutation(ls1)
        shuffled4_ls2 = np.random.permutation(ls2)
        
        df1 = pd.DataFrame({"channel1": ls1, "label": [label]*len(ls1)}) # 3 channels: dog-dog-dog (if label="dog")
        df1["channel2"] = shuffled1_ls1
        df1["channel3"] = shuffled2_ls1
        
        df2 = pd.DataFrame({"channel1": ls1, "label": [label]*len(ls1)}) # dog-dog-cat
        df2["channel2"] = shuffled3_ls1
        df2["channel3"] = shuffled1_ls2
        
        df3 = pd.DataFrame({"channel1": ls1, "label": [label]*len(ls1)}) # dog-cat-dog
        df3["channel2"] = shuffled2_ls2
        df3["channel3"] = shuffled4_ls1
        
        df4 = pd.DataFrame({"channel1": ls1, "label": [label]*len(ls1)}) # dog-cat-cat
        df4["channel2"] = shuffled3_ls2
        df4["channel3"] = shuffled4_ls2
        
        return pd.concat([df1, df2,
                          df3, df4
                         ], ignore_index=True)
    
    
    def create_hybrid_df(self, df):
        
        dogs = df[df["label"] == "dog"]["filename"].values
        cats = df[df["label"] == "cat"]["filename"].values

        if len(cats) > len(dogs):
            cats = cats[:len(dogs)]
        else:
            dogs = dogs[:len(cats)]

        dog_df = self.hybrate(dogs, cats, label="dog", seed=2022)
        cat_df = self.hybrate(cats, dogs, label="cat", seed=2021)

        new_df = pd.concat([dog_df, cat_df], ignore_index=True)
        new_df["label"] = new_df["label"].map({"dog": 1, "cat": 0})
        
        return new_df
    
    def __load_img(self, imname):
        if imname is None:
            return np.zeros((self.target_size[0], self.target_size[1], 1)).astype("uint8")
        else:
            im = image.load_img(os.path.join(self.imdir, imname), target_size=self.target_size)
            channel = np.array(im)[:,:,:1]
            channel = (channel - channel.min()) / (channel.max() - channel.min()) * 255.
            
            return channel.astype("uint8")
    
    def __generate_data(self, list_ids_tmp):
        X = np.empty((self.batch_size, *self.target_size))
        y = np.empty((self.batch_size, self.n_classes))
        
        files = []
        
        for i, ID in enumerate(list_ids_tmp):
            channel2 = self.__load_img(self.channel2s[ID])
            channel1 = self.__load_img(self.channel1s[ID])
            channel3 = self.__load_img(self.channel3s[ID])
            files.append((self.channel1s[ID], self.channel2s[ID], self.channel3s[ID]))
            
            im = np.concatenate([channel1, channel2, channel3], axis=2)
            if self.preprocess:
                im = preprocess_input(im)
            else:
                im = im / 255.
            
            X[i,:] = im.astype(np.float32)
            y[i,:] = self.labels[ID]
            
        if self.stage in ["train", "val", "eval"]:
            return X, y
        else:
            return X, files
        
class PureDatagen(tf.keras.utils.Sequence):
    def __init__(self,
                 imdir,
                 df,
                 target_size=(224, 224, 3),
                 preprocess=False,
                 n_classes=2,
                 batch_size=32,
                 stage="train"
                ):
        """Generate pure data (3 channels are the same object, representing for color RGB)

        Args:
            imdir (str): path to directory with images
            df (pd.DataFrame): dataframe with filenames and labels
            target_size (tuple, optional): input size. Defaults to (224,224,3).
            preprocess (bool, optional): preprocessing with preprocess_input of keras model. Defaults to False.
            n_classes (int, optional): number of classes. Defaults to 2.
            batch_size (int, optional): batch size. Defaults to 32.
            seed (int, optional): random seed. Defaults to 42.
            stage (str, optional): stage of data ["train", "val", "eval", "test"]. Defaults to "train".
        """
        self.imdir = imdir
        self.target_size = target_size
        self.preprocess = preprocess
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.stage = stage
        
        self.df = df.copy()
        self.df["label"] = self.df["label"].map({"dog": 1, "cat": 0})
        print("Found {} samples: {}".format(len(self.df), self.df.label.value_counts()))
        self.filenames = self.df["filename"].values
        self.list_IDs = [i for i in range(len(self.filenames))]
        self.labels = self.df["label"].values
        if n_classes > 1:
            self.labels = tf.keras.utils.to_categorical(self.labels, n_classes)
            
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.filenames) / np.float(self.batch_size)))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        list_ids_tmp = [self.list_IDs[k] for k in indexes]
        
        return self.__generate_data(list_ids_tmp)
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.stage in ["train", "val"]:
            np.random.seed(self.seed)
            np.random.shuffle(self.indexes)
    
    def __generate_data(self, list_ids_tmp):
        X = np.empty((self.batch_size, *self.target_size))
        y = np.empty((self.batch_size, self.n_classes))
        
        files = []
        
        for i, ID in enumerate(list_ids_tmp):
            im = image.load_img(os.path.join(self.imdir, self.filenames[ID]), target_size=self.target_size)
            
            if self.preprocess:
                im = preprocess_input(im)
            else:
                im = np.array(im) / 255.
                
            X[i,:] = im.astype(np.float32)
            y[i,:] = self.labels[ID]
            
        if self.stage in ["train", "val", "eval"]:
            return X, y
        else:
            return X, files
        