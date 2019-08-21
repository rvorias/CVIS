#!/usr/bin/env python
# coding: utf-8
# In[1]:
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from data import getDF
from keras.callbacks import TensorBoard
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import CustomObjectScope
from losses import edgeLossMSE
from regularizers import l2_reg

#filter = ['aeroplane','bicycle','bird','boat','bottle',
#          'bus','car','cat','chair','cow',
#          'diningtable','dog','horse','motorbike','person',
#          'pottedplant','sheep','sofa','train','tvmonitor']
#filter = ['aeroplane','boat']
#filter = ['bicycle','boat','horse','motorbike','sofa','train','pottedplant','tvmonitor']     
filter = ['aeroplane','motorbike','bus','bicycle','cat','train','horse']

image_size = (128,128)
batchSize = 32

useBlankModel = True
useTrainedModel = not useBlankModel
# if you want to start from scratch, load:
# 'saved_models/autoEncoder_baseline.h5'

modelName = 'AE_combined'

blankModelPath = 'saved_models/'+modelName+'.h5'
trainedModelPath = 'saved_models/'+modelName+'_trained.h5'
#%%
if useBlankModel:
    with CustomObjectScope({'edgeLossMSE': edgeLossMSE, 'l2_reg': l2_reg}):
        autoEncoder = load_model(blankModelPath)
elif useTrainedModel:
    autoEncoder = load_model(trainedModelPath)
#%%
autoEncoder.summary()
#%%
def AE_generator(generator):
    '''Yields autoencoder X and Y'''
    for batch in generator:
        yield (batch[0], batch[0])
            
def plotPrediction(model,generator):
    '''Plots a prediction of the given model and generator'''
    img = next(generator)[0][:1]
    print('=== intermediate plot ===')
    decoded = model.predict(img, verbose=1)
    decoded = decoded[0]
    
    img[0] = (img[0]*scaleFac+mean_images)/255.0
    decoded = (decoded*scaleFac+mean_images)/255.0
    
    diff = img[0] - decoded
    plt.figure(figsize=(10, 3))
    plt.imshow(np.hstack((img[0], decoded, diff)))
    plt.title('Original, reconstructed, diff')
    plt.show()
    
class interPlot(Callback):
    '''
    keras callback for plotting prediction results between training epochs
    '''
    def on_epoch_end(self, epoch, logs={}):
        plotPrediction(autoEncoder,generator_test)
#%%
# retrieve training and test data and labels
mlb = MultiLabelBinarizer()
df, mlb = getDF(filter, mlb)
train, test = train_test_split(df, test_size=0.1, random_state=42)
#%%
# define generators
image_data_generator_train = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=.1,
    height_shift_range=.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    featurewise_center=True,
    featurewise_std_normalization=True,
    fill_mode='nearest')

image_data_generator_test = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

generator_train = image_data_generator_train.flow_from_dataframe(
    dataframe=train,
    directory='..//VOCdevkit/VOC2009/JPEGImages',
    x_col='filename',
    y_col='index',
    class_mode='other',
    color_mode="rgb",
    target_size=(image_size[1],image_size[0]),
    interpolation='lanczos',
    batch_size=batchSize)

generator_test = image_data_generator_test.flow_from_dataframe(
    dataframe=test,
    directory='..//VOCdevkit/VOC2009/JPEGImages',
    x_col='filename',
    y_col='index',
    class_mode='other',
    color_mode="rgb",
    target_size=(image_size[1],image_size[0]),
    interpolation='lanczos',
    batch_size=batchSize)

AE_generator_train = AE_generator(generator_train)
AE_generator_test = AE_generator(generator_test)
#%%
# generator in order to find the mean and variance values
fitImgGen = ImageDataGenerator()

fitGen = fitImgGen.flow_from_dataframe(
    dataframe=train,
    directory='..//VOCdevkit/VOC2009/JPEGImages',
    x_col='filename',
    y_col='index',
    class_mode='other',
    color_mode="rgb",
    target_size=(image_size[1],image_size[0]),
    interpolation='lanczos',
    batch_size=generator_train.n)
#%%
# original functions are commented out for speed

#images = next(fitGen)[0]
#mean_images = images.mean()
#mean_stds = images.std()
#print(mean_images)
#print(mean_stds)
mean_images = 114.661835
mean_stds = 68.592606
print(mean_images)
print(mean_stds)
#%%
#image_data_generator_train.mean = mean_images
#image_data_generator_train.std = mean_stds
#image_data_generator_test.mean = mean_images
#image_data_generator_test.std = mean_stds
#%%
# instead of scaling by 255 we scale by a mean subtracted value in order to
# the values in the range of [-1,1]
scaleFac = 255.0 - mean_images
print(scaleFac)
image_data_generator_train.mean = mean_images
image_data_generator_train.std = scaleFac
image_data_generator_test.mean = mean_images
image_data_generator_test.std = scaleFac
#%%
# show some images from the generator
images = next(generator_train)
for i in range(batchSize):
    plt.imshow(images[0][i,:,:,:])
    print(images[1][i])
    plt.show()
#%%
# show some images from the generator
images = next(generator_test)
for i in range(batchSize):
    plt.imshow(images[0][i,:,:,:])
    print(images[1][i])
    plt.show()
#%%
plotPrediction(autoEncoder,generator_test)
#%%
filepath = trainedModelPath
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
tensorboard = TensorBoard(log_dir = "./logs/AE/{}".format(trainedModelPath))
earlyStop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=500, 
                          verbose=1)
callbacks_list = [checkpoint, tensorboard, earlyStop, interPlot()]
#%%
nb_epoch = 500
hist = autoEncoder.fit_generator(
        AE_generator(generator_train),
        steps_per_epoch=int(generator_train.n/batchSize),
        validation_data=AE_generator(generator_test),
        validation_steps=int(generator_test.n/batchSize),
        nb_epoch=nb_epoch,
        callbacks=callbacks_list,
        use_multiprocessing=True,
        workers=5)
#%%
n = next(generator_test)[0]
plt.imshow(n[0])
plt.show()
for j in np.arange(1,40):
    try:
        if 'conv2d' in autoEncoder.layers[j].get_config()['name']:
            n_layer_output = K.function([autoEncoder.layers[0].input],[autoEncoder.layers[j].output])
            layer_output = n_layer_output([n])[0][0]
            for i in range(layer_output.shape[-1]):
                plt.subplot(4,layer_output.shape[-1]/4,i+1)
                plt.imshow(layer_output[:,:,i])
                plt.axis('off')
            plt.show()
    except:
        pass    