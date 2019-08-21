# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.callbacks import TensorBoard


from keras.preprocessing.image import ImageDataGenerator

from data import getDF
#%%
from keras import backend as K
#https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d

def weighted_CE_logit(y_true, y_pred):
    #https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    q = 0.85
    l = (1 + (q-1)*y_true)
    #logits
    x = y_pred
    loss = (1-y_true)*x + l*(K.log(1+K.exp(-K.abs(x))) + K.max(-x, 0))
    
    return loss

import tensorflow as tf
def wBCE(target, output):
    #https://towardsdatascience.com/sigmoid-activation-and-binary-crossentropy-a-less-than-perfect-match-b801e130e31
    #basically, multi-label classification is really hard:
    # https://nickcdryan.com/2017/01/23/multi-label-classification-a-guided-tour/
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,logits=output)
    loss = tf.reduce_mean(loss)
    return loss
    
#%%
# when you change the amount of classes, you have to 
# regenerate the classifier model
# (because the final layer corresponds to the amount of classes)
#filter = ['aeroplane','bicycle','boat','bus','car','motorbike', 'bird', 'cat', 'cow', 'dog', 'horse'] 
filter = ['aeroplane','motorbike','bus','bicycle','cat','train','horse']

batchSize = 32


useBlankModel = False
useTrainedModel = not useBlankModel

modelName = 'frozen'

blankModelPath = 'saved_models/CL_'+modelName+'.h5'
trainedModelPath = 'saved_models/CL_'+modelName+'_trained.h5'
#%%from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, BatchNormalization
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, BatchNormalization
from keras.layers import Flatten, Dropout, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import PReLU, LeakyReLU, ReLU, ELU
from keras import regularizers
from keras import backend as K
from keras import optimizers

image_size = (128,128)
#%%
input_img = Input(shape=(image_size[1], image_size[0], 3))
kernelInit  = 'glorot_uniform'
regLam = 0.001

encoder = load_model('saved_models/AE_00001reg_trained.h5')
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
x = preTrained(input_img)

x = Conv2D(72, (4, 4),
           kernel_initializer=kernelInit,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
x = ReLU(name='latent_space')(x)

x = Flatten()(x)
x = Dense(512,
           kernel_regularizer=regularizers.l2(regLam))(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Dropout(0.5)(x)
x = Dense(len(filter), activation='softmax')(x)

#model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(amountOfClasses, activation=None))
 
# Show a summary of the model. Check the number of trainable parameters
classifier = Model(input_img, x)
classifier.summary()
classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%
mlb = MultiLabelBinarizer()
df, mlb = getDF(filter, mlb)
train, test = train_test_split(df, test_size=0.2, random_state=42)
#%%
singleDf = df[df['tags'].map(len)==1]
train, test = train_test_split(singleDf, test_size=0.2, random_state=42)
#%%
def multilabel_flow_from_dataframe(data_generator, mlb):
    for x, y in data_generator:
        indices = y.astype(np.int).tolist()
        y_multi = mlb.transform(df.iloc[indices]['tags'].values.tolist())
        yield x, y_multi
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
# show some images from the generator
images = next(generator_train)
for i in range(batchSize):
    plt.imshow(images[0][i,:,:,:])#*mean_stds+mean_images)/255.0)
    print(images[1][i])
    plt.show()
#%%
from keras.utils import CustomObjectScope

if useBlankModel:
    classifier = load_model(blankModelPath)
elif useTrainedModel:
    with CustomObjectScope({'wBCE': wBCE}):
        classifier = load_model(trainedModelPath)
#%%
class interPlot(Callback):
    '''
    keras callback for plotting prediction results between training epochs
    '''
    def on_epoch_end(self, epoch, logs={}):
        plotPrediction(classifier,class_tester)
#%%
filepath = trainedModelPath
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
tensorboard = TensorBoard(log_dir = "./logs/CL/{}".format(trainedModelPath))
earlyStop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=500, 
                          verbose=1)
callbacks_list = [checkpoint, tensorboard, earlyStop, interPlot()]
# In[72]:  
nb_epoch = 250
hist = classifier.fit_generator(
        multilabel_generator_train,
        steps_per_epoch = generator_train.n/batchSize,
        validation_data=multilabel_generator_test,
        validation_steps=int(generator_test.n/batchSize),
        nb_epoch = nb_epoch,
        callbacks = callbacks_list,
        use_multiprocessing=True,
        workers=5)
#%%
def multilabel_flow_test(data_generator, mlb):
    for x, y in data_generator:
        indices = y.astype(np.int).tolist()
        y_multi = mlb.transform(df.iloc[indices]['tags'].values.tolist())
        yield x, y_multi, y
class_tester = multilabel_flow_test(generator_test, mlb)
sortedFilter = [filter[i] for i in [0,1,2,3,4,5,6]]
#%%
def plotPrediction(classifier,classGenerator):
    n = next(classGenerator) # Get one image
    plt.imshow((n[0][0] * scaleFac + mean_images)/255.0)
    plt.title('index: {}'.format(n[2][0]))
    plt.show()
    img = n[0][0:1]
    true_label = n[1][0:1]
    dec = classifier.predict(img, verbose=1) # Decoded image
    print('          class   true  pred')
    print('--------------------------------')
    #dec = np.exp(dec)/(np.exp(dec)+1)
    for i in range(len(filter)):
        print('{:>15} - {} --- {:.2f}'.format(sortedFilter[i],true_label[0][i], dec[0][i]), end =" ")
        if dec[0][i] == np.max(dec[0]):
            print('<--')
        else:
            print('')
            
plotPrediction(classifier,class_tester)
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
yTs=n[1]
yPs=classifier.predict(n[0], steps=1)

for i in range(1,generator_eval.n):
    n = next(multilabel_generator_eval)
    yTs = np.vstack([yTs,n[1]])
    pred = classifier.predict(n[0], steps=1, verbose=1)
    yPs = np.vstack([yPs,pred])
#%%
# only if using logits
#yPs = np.exp(yPs)/(np.exp(yPs)+1)
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

for i, f, cm in zip(range(len(sortedFilter)),sortedFilter,mcm):
    subMetric = {}
    for metricName, m in zip(metricOrder,cm.ravel()):
        subMetric[metricName] = m
    subMetric['precision'] = subMetric['tp']/(subMetric['tp']+subMetric['fp'])
    subMetric['recall'] = subMetric['tp']/(subMetric['tp']+subMetric['fn']) 
    subMetric['f1'] = (1+beta**2)*subMetric['precision']*subMetric['recall']/((beta**2)*subMetric['precision']+subMetric['recall'])
    metrics[f] = subMetric
#%%
    
f1s = [metrics[sf]['f1'] for sf in sortedFilter]
print('     Class   Prec    Rec     F1')
print('---------------------------------------')
for sf in sortedFilter:
    print('{:>10}   {:.2f}    {:.2f}    {:.2f}'.format(sf,
          metrics[sf]['precision'],
          metrics[sf]['recall'],
          metrics[sf]['f1']))
    
#%%    
#w = 4
#h = 2
#f, ax = plt.subplots(w,h, figsize=(h*3,w*3))
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
#
#for i in range(w*h):
#    x = i%w
#    y = int((i)/w)
#    
#    title = '{}, prec:{:.2f}, rec:{:.2f}'.format(sortedFilter[i],metrics[sortedFilter[i]]['precision'],metrics[filter[i]]['recall'])
#    sn.heatmap(mcm[i], annot=True, fmt="d", ax=ax[x,y], cmap="YlGnBu")
#    ax[x,y].set_title(title)
#    ax[x,y].set_ylabel('True label')
#    ax[x,y].set_xlabel('Predicted label')
#    ax[x,y].xaxis.set_ticks_position('top')
#    ax[x,y].xaxis.set_label_position('top')
#%%
classifier.summary()
#%%

n = next(multilabel_generator_eval)[0]
plt.imshow(n[0])
plt.show()
for j in np.arange(1,20):
    if 'conv2d' in classifier.layers[j].get_config()['name']:
        n_layer_output = K.function([classifier.layers[0].input],[classifier.layers[j].output])
        layer_output = n_layer_output([n])[0][0]
        for i in range(layer_output.shape[-1]):
            plt.subplot(4,layer_output.shape[-1]/4,i+1)
            plt.imshow(layer_output[:,:,i])
            plt.axis('off')
        plt.show()   
#%%

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

print(cm_sm.ravel())
#%%
classifier.summary()
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
for i in range(20):
    plotWeights(i,classifier)

#%%
c = cm_sm.ravel()
c = np.reshape(c,(7,7))
cd = np.diag(c)
print(np.sum(cd)/np.sum(c))
























