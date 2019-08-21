# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from data import getDF

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from keras.layers import Input, Flatten, Dense, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, add, Dropout
from keras.layers import Concatenate
from keras.models import Model
from keras.layers.advanced_activations import ReLU, PReLU, LeakyReLU, ELU
from keras import regularizers
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from regularizers import l2_reg
from keras.utils import plot_model

import tensorflow as tf

filter = ['aeroplane','motorbike','bus','bicycle','cat','train','horse']

image_size = (128,128)
batchSize = 32

# if you want to start from scratch, load:
# 'saved_models/autoEncoder_baseline.h5'
trainedModelPath = 'saved_models/DU_trained.h5'

#%%
def multilabel_flow_from_dataframe(data_generator, mlb):
    for x, y in data_generator:
        indices = y.astype(np.int).tolist()
        y_multi = mlb.transform(df.iloc[indices]['tags'].values.tolist())
        yield x, [x,y_multi]
            
def plotPredictionDual(model,generator):
    n = next(generator) # Get one image

    img = n[0][0:1]
    true_label = n[1][1][0:1]
    dec = model.predict(img, verbose=1)
    
    true_image = img[0]
    reconstructed_image = dec[0][0]
    diff = true_image-reconstructed_image
    
    combined = np.hstack([true_image, reconstructed_image, diff])
    combined =  (combined* scaleFac + mean_images)/255.0
    
    plt.figure(figsize=(6, 3))
    plt.imshow(combined)
    plt.title('Original  --  Reconstructed  --  Difference')
    plt.show()
    
    pred_label = dec[1]
    print('          class   true  pred')
    print('--------------------------------')
    #dec = np.exp(dec)/(np.exp(dec)+1)
    for i in range(len(filter)):
        print('{:>15} - {} --- {:.2f}'.format(filter[i],true_label[0][i], pred_label[0][i]), end =" ")
        if pred_label[0][i] == np.max(pred_label[0]):
            print('<--')
        else:
            print('')
    
class interPlot(Callback):
    '''
    keras callback for plotting prediction results between training epochs
    '''
    def on_epoch_end(self, epoch, logs={}):
        plotPredictionDual(dual,multilabel_generator_test)
#%%
mlb = MultiLabelBinarizer()
df, mlb = getDF(filter, mlb)
#train, test = train_test_split(df, test_size=0.2, random_state=42)
#%%
singleDf = df[df['tags'].map(len)==1]
train, test = train_test_split(singleDf, test_size=0.2, random_state=42)
#%%
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
    featurewise_std_normalization=True,)

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

multilabel_generator_train = multilabel_flow_from_dataframe(generator_train, mlb)
multilabel_generator_test = multilabel_flow_from_dataframe(generator_test, mlb)
#%%
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
input_img = Input(shape=(image_size[1], image_size[0], 3))
kernelInit  = 'glorot_uniform'
regLam = 0.001

x = Conv2D(16, (5, 5),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(input_img)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(24, (3, 3),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, (3, 3),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

skip_1 = Conv2D(48, (3, 3),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(skip_1)
x = ReLU()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
to_dec = ReLU(name='enc')(x)

x = Conv2D(72, (4, 4),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(to_dec)
x = BatchNormalization()(x)
encoded = ReLU()(x)

###### DECODER ######

x = Conv2DTranspose(64, (4,4), strides=2, padding='valid')(to_dec)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(64, (3,3), strides=1, padding='valid')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)
x = Concatenate(axis=-1)([x, skip_1])

x = Conv2DTranspose(48, (4,4), strides=2, padding='valid')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(48, (3,3), strides=1, padding='valid')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(32, (4,4), strides=2, padding='valid')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(32, (3,3), strides=1, padding='valid')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(24, (4,4), strides=2, padding='valid')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(24, (5,5), strides=1, padding='valid')(x)
#x = BatchNormalization()(x)
x = ReLU()(x)

decoded = Conv2DTranspose(3, (3, 3),
                          activation='tanh',
                          padding='valid',
                          name='AE')(x)

# Classification branch
x = Flatten()(encoded)
x = Dense(512,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Dropout(0.5)(x)
cl = Dense(len(filter), activation='softmax',name='classifier')(x)

# Compile the dual model

dual = Model(input_img, outputs=[decoded,cl])

dual.compile(optimizer='adam',
             loss=['mse','categorical_crossentropy'],
             loss_weights=[0.95,1.0],
             metrics=['accuracy'])

dual.summary()
#%%
n = next(multilabel_generator_test)
plotPredictionDual(dual,multilabel_generator_test)

img = n[0][0:1]
true_label = n[1][1][0:1]
dec = dual.predict(img, verbose=1)
#%%
from keras.models import load_model
dual = load_model('saved_models/dual_v3_trained.h5')
#%%
plot_model(dual, show_shapes=True, to_file='ae_baseline.png')
#%%
filepath = trainedModelPath
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_classifier_acc',
                             verbose=1,
                             save_best_only=True)
tensorboard = TensorBoard(log_dir = "./logs/DU/{}".format(trainedModelPath))
callbacks_list = [checkpoint, tensorboard, interPlot()]
#%%
nb_epoch = 250

hist = dual.fit_generator(
        multilabel_generator_train,
        steps_per_epoch=int(generator_train.n/batchSize),
        validation_data=multilabel_generator_test,
        validation_steps=int(generator_test.n/batchSize),
        nb_epoch=nb_epoch,
        callbacks=callbacks_list,
        use_multiprocessing=True,
        workers=5)
#%%
def plotWeights(layerNumber, model):
    try:
        x1w = model.get_weights()[layerNumber]
        noOfWeights = x1w.shape[3]
        print(model.layers[layerNumber].get_config()['name'])
        print('filter weights of layer {}'.format(layerNumber))
        print(x1w.shape)
        for i in range(noOfWeights):
            plt.subplot(4,noOfWeights/4,i+1)
            x = x1w[:,:,0:3,i]
            x = (x - np.min(x))/(np.max(x)-np.min(x))
            plt.imshow(x)
            plt.axis('off')
        plt.show() 
    except:
        pass
#%%
for i in range(10):
    plotWeights(i,dual)
#%%
n = next(multilabel_generator_test)[0]
plt.imshow(n[0])
plt.show()
for j in np.arange(1,20):
    try:
        if 'conv2d' in dual.layers[j].get_config()['name']:
            n_layer_output = K.function([dual.layers[0].input],[dual.layers[j].output])
            layer_output = n_layer_output([n])[0][0]
            for i in range(layer_output.shape[-1]):
                plt.subplot(4,layer_output.shape[-1]/4,i+1)
                plt.imshow(layer_output[:,:,i])
                plt.axis('off')
            plt.show()
    except:
        pass
#%%
yTs = None
yPs = None


generator_eval = image_data_generator_test.flow_from_dataframe(
    dataframe=test,
    directory='..//VOCdevkit/VOC2009/JPEGImages',
    x_col='filename',
    y_col='index',
    class_mode='other',
    color_mode="rgb",
    shuffle = False,
    target_size=(image_size[1],image_size[0]),
    interpolation='lanczos',
    batch_size=1)
multilabel_generator_eval = multilabel_flow_from_dataframe(generator_eval, mlb)
#%%
n = next(multilabel_generator_eval)
yTs=n[1][1]
yPs=dual.predict(n[0], steps=1)[1]

for i in range(1,generator_eval.n):
    n = next(multilabel_generator_eval)
    yTs = np.vstack([yTs,n[1][1]])
    pred = dual.predict(n[0], steps=1, verbose=1)[1]
    yPs = np.vstack([yPs,pred])
#%%
predRounded = np.round(yPs)

metrics = {}
metricOrder = ['tn', 'fp', 'fn', 'tp']
beta = 1
#%%
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
mcm = multilabel_confusion_matrix(yTs, predRounded)

#%%
import seaborn as sn

for i, f, cm in zip(range(len(filter)),filter,mcm):
    subMetric = {}
    for metricName, m in zip(metricOrder,cm.ravel()):
        subMetric[metricName] = m
    subMetric['precision'] = subMetric['tp']/(subMetric['tp']+subMetric['fp'])
    subMetric['recall'] = subMetric['tp']/(subMetric['tp']+subMetric['fn']) 
    subMetric['f1'] = (1+beta**2)*subMetric['precision']*subMetric['recall']/((beta**2)*subMetric['precision']+subMetric['recall'])
    metrics[f] = subMetric
#%%
    
f1s = [metrics[sf]['f1'] for sf in filter]
print('     Class   Prec    Rec     F1')
print('---------------------------------------')
for sf in filter:
    print('{:>10}   {:.2f}    {:.2f}    {:.2f}'.format(sf,
          metrics[sf]['precision'],
          metrics[sf]['recall'],
          metrics[sf]['f1']))
#%%
dual.summary()
#%%
predMax = [np.argmax(i) for i in yPs]
trueMax = [np.argmax(i) for i in yTs]

cm_sm = confusion_matrix(trueMax, predMax)
ax = sn.heatmap(cm_sm, annot=True, fmt="d", cmap="YlGnBu", yticklabels=filter)
ax.xaxis.set_ticks_position('top')
ax.set_xticklabels(filter, rotation=30)
ax.set_ylabel('True label')
ax.set_xlabel('Predicted')
ax.xaxis.set_label_position('top')
#%%
c = cm_sm.ravel()
c = np.reshape(c,(7,7))
cd = np.diag(c)
print(np.sum(cd)/np.sum(c))
#
#%%
generator_eval = image_data_generator_test.flow_from_dataframe(
    dataframe=test,
    directory='..//VOCdevkit/VOC2009/JPEGImages',
    x_col='filename',
    y_col='index',
    class_mode='other',
    color_mode="rgb",
    shuffle = False,
    target_size=(image_size[1],image_size[0]),
    interpolation='lanczos',
    batch_size=generator_test.n,
    seed=666)

yTs = next(generator_eval)[0]
#%%
  
yPs = []


for i in yTs:
    i = np.reshape(i,(1,128,128,3))
    yPs.append(dual.predict(i,verbose=0)[0])
yPs = np.vstack(yPs)
y_true = yTs.flatten()
y_pred = yPs.flatten()

plt.title('     original                          reconstructed')
imgs = np.hstack((yTs[0],yPs[0]))
imgs = (imgs*scaleFac+mean_images)/255.0
plt.imshow(np.hstack((yTs[0],yPs[0])))
plt.axis('off')
plt.show()

mse = np.mean((y_true-y_pred)**2)
print('MSE test set: {}'.format(mse))
#%%
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE

def plotTSNE(layerName, inputs, figsize=(10,10), zoom=0.25):
    try:
        for j in np.arange(1,40):
            if layerName in dual.layers[j].get_config()['name']:
                n_layer_output = K.function([dual.layers[0].input],[dual.layers[j].output])
    except:
        pass
    Xlist = []
    for i in inputs:
        i = np.reshape(i,(1,128,128,3))
        Xlist.append(n_layer_output([i])[0])
    x = np.vstack(Xlist)
    X = np.reshape(x,(x.shape[0],-1))
    print('stacked shape: {}'.format(X.shape))
    X_embedded = TSNE(n_components=2).fit_transform(X)
    plotImages = [i for i in inputs]
    plotImages = map(lambda x: (x * scaleFac+mean_images)/255.0, plotImages)
    
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_embedded, plotImages):
        x0, y0 = xy
        img = OffsetImage(i, zoom=zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_embedded)
    ax.autoscale()
    plt.axis('off')
    plt.show()
plotTSNE('enc',yTs)