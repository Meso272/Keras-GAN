from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model,load_model
from keras.optimizers import RMSprop

import keras.backend as K
from sklearn.metrics import mean_squared_error

from datasets import * 
import numpy as np

orig_test=load_CLDHGH_orig(path="/home/jliu447/lossycompression/multisnapshot-data-cleaned/CLDHGH/",startnum=50,endnum=63,scale=False,size=64)
decomp_test=load_CLDHGH_decomp(path="/home/jliu447/lossycompression/multisnapshot-data-cleaned/CLDHGH_SZ/",startnum=50,endnum=63,size=64)
orig_test = np.expand_dims(orig_test, axis=3)
decomp_test = np.expand_dims(decomp_test, axis=3)
model=load_model("generator.h5")

recov_test=model.predict(decomp_test)
print(recov_test.shape

decomp_test=decomp_test/2+0.5
recov_test=recov_test/2+0.5
print(mean_squared_error(orig_test.flatten(),decomp_test.flatten()))
print(mean_squared_error(orig_test.flatten(),recov_test.flatten()))