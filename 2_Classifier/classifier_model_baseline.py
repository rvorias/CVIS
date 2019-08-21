# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import Model
from keras import layers
from keras import models
from keras.metrics import categorical_accuracy
#%%

amountOfClasses = 15

modelName = 'baseline'
pretrainedModelPath = 'saved_models/autoEncoder_'+modelName+'_trained.h5'
#%%
encoder = load_model(pretrainedModelPath)
encoder.summary()
#%%
while encoder.layers[-1].name != 'enc':
    encoder.layers.pop()
    
preTrained = Model(encoder.input, encoder.layers[-1].output)
preTrained.summary()
#%%
# Freeze the layers except the last 4 layers
for layer in preTrained.layers:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in preTrained.layers:
    print(layer, layer.trainable)
#%%
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(preTrained)
 
# Add new layers
model.add(layers.Conv2D(256,(5,5), strides = 2, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(512,(3,3), strides = 2, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(amountOfClasses, activation='sigmoid'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()
#%%
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
# Train the model
#%%
from keras import backend as K
session = K.get_session()
for layer in model.layers: 
     for v in layer.__dict__:
         v_arg = getattr(layer,v)
         if hasattr(v_arg,'initializer'):
             initializer_method = getattr(v_arg, 'initializer')
             initializer_method.run(session=session)
             print('reinitializing layer {}.{}'.format(layer.name, v))
#%%
model.save('saved_models/classifier_'+modelName+'_finetune.h5')