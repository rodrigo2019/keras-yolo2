''' if your custom backend  file is in another folder, you can use sys.path.append to add the 
folder wheres the backend module from this repo is located, also it works for another imports ''' 
import sys 
sys.path.append("..")
#sys.path.append("path/to/backend") 
from keras_yolov2.backend import BaseFeatureExtractor 
from tensorflow.python.keras.models import Model, Sequential
import tensorflow as tf 
from tensorflow.python.keras.layers import Reshape, Conv2D, Input, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

SCALE=1 
class SuperTinyYoloFeature(BaseFeatureExtractor): 
    """ 
    It is a example from TinyTolo reduced around 16x times your size, also this network has 
    4 maxPoolings instead 5 as the original, with 4 maxpoolings this network will generate a different 
    grid size 
    """ 
    def __init__(self, input_size): 
        model = Sequential() 
     
        model.add(Conv2D(16//SCALE, (3, 3), padding="same",input_shape=input_size)) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(MaxPooling2D(pool_size=(3, 2), strides=(2, 2)))
                 
        model.add(Conv2D(16//SCALE, (1, 1), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(128//SCALE, (3, 3), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(16//SCALE, (1, 1), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(128//SCALE, (3, 3), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(MaxPooling2D(pool_size=(3, 2), strides=(2, 2)))
         
        model.add(Conv2D(32//SCALE, (1, 1), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(256//SCALE, (3, 3), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(32//SCALE, (1, 1), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(256//SCALE, (3, 3), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(MaxPooling2D(pool_size=(3, 2), strides=(2, 2)))
 
        model.add(Conv2D(64//SCALE, (1, 1), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(512//SCALE, (3, 3), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(64//SCALE, (1, 1), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(512//SCALE, (3, 3), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        model.add(Conv2D(128//SCALE, (1, 1), padding="same")) 
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.1)) 
        # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  
        try: 
            model.load_weights("www_ckp.h5") 
        except: 
            print("using fresh backend model") 
        self.feature_extractor = model  
 
    def normalize(self, image): 
        return image / 255.
