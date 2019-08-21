# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, BatchNormalization
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.layers.advanced_activations import PReLU, LeakyReLU, ReLU, ELU
from keras import regularizers
from keras import backend as K
from keras import optimizers
from regularizers import l2_reg
from keras.utils import CustomObjectScope

image_size = (128,128)
#%%
input_img = Input(shape=(image_size[1], image_size[0], 3))
kernelInit  = 'glorot_uniform'
regLam = 0.00001

#activationFunc = ReLU()

x = Conv2D(24, (5, 5),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(input_img)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, (3, 3),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(48, (3, 3),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(96, (3, 3),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
encoded = ReLU(name='enc')(x)

###### DECODER ######

x = Conv2DTranspose(96, (3,3), padding='valid')(encoded)
x = BatchNormalization()(x)
x = ReLU()(x)

x = UpSampling2D()(x)

x = Conv2DTranspose(64, (3,3), padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = UpSampling2D()(x)

x = Conv2DTranspose(48, (3,3), padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = UpSampling2D()(x)

x = Conv2DTranspose(32, (3,3), padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = UpSampling2D()(x)

x = Conv2DTranspose(24, (3,3), padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

decoded = Conv2DTranspose(3, (3, 3),
                          activation='tanh',
                          padding='valid',
                          name='AE')(x)

autoEncoder = Model(input_img, decoded)
autoEncoder.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
#%%
autoEncoder.summary()
#%%
autoEncoder.save('saved_models/AE_expanded.h5')

