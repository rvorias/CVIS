#!/usr/bin/env python
# coding: utf-8
# In[1]:
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from data import getDF,augmentDFSegmentation
from keras.callbacks import TensorBoard
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import CustomObjectScope
from regularizers import l2_reg
import tensorflow as tf

filter = ['aeroplane','bicycle','bird','boat','bottle',
          'bus','car','cat','chair','cow',
          'diningtable','dog','horse','motorbike','person',
          'pottedplant','sheep','sofa','train','tvmonitor']
#filter = ['aeroplane','boat']
#filter = ['bicycle','boat','horse','motorbike','sofa','train','pottedplant','tvmonitor']    
#filter = ['aeroplane','motorbike','bus','bicycle','cat','train','horse']

image_size = (128,128)
batchSize = 32
#%%
# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def dice_loss(y_true, y_pred):
    #https://arxiv.org/pdf/1707.03237.pdf
    e = K.epsilon()
    product1 = K.sum(y_true*y_pred+e, axis=-1)
    sum1 = K.sum(y_true+y_pred+e, axis=-1)
    
    product2 = K.sum((1.0-y_true)*(1.0-y_pred)+e, axis=-1)
    sum2 = K.sum(2-y_true-y_pred+e, axis=-1)
    
    return 1 - product1/sum1 - product2/sum2

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, BatchNormalization, Concatenate
from keras.layers import MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras.layers.advanced_activations import PReLU, LeakyReLU, ReLU
from keras import regularizers
from keras import optimizers
#from losses import edgeLoss

#x = Concatenate(axis=-1)([x, x_skip_1])

input_img = Input(shape=(image_size[1], image_size[0], 3))
compressionDepth = 256
dropoutRate = 0.5
poolSize = 2

x = Conv2D(16, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=poolSize)(x)

x = Conv2D(24, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

skip_1 = Conv2D(24, (3, 3), padding='same')(x)
x = BatchNormalization()(skip_1)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=poolSize)(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

skip_2 = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(skip_2)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=poolSize)(x)

x = Conv2D(48, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

skip_3 = Conv2D(48, (3, 3), padding='same')(x)
x = BatchNormalization()(skip_3)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=poolSize)(x)

x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

skip_4 = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(skip_4)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=poolSize)(x)

x = Conv2D(72, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(dropoutRate)(x)

### LATENT
x = Conv2D(72, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(name='enc')(x)
x = Dropout(dropoutRate)(x)

### DECODER
x = Conv2D(72, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = UpSampling2D()(x)

x = Conv2D(64, (3,3), padding='same')(x)
x = Concatenate(axis=-1)([x, skip_4])
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(64, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = UpSampling2D()(x)

x = Conv2D(48, (3,3), padding='same')(x)
x = Concatenate(axis=-1)([x, skip_3])
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(48, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = UpSampling2D()(x)

x = Conv2D(32, (3,3), padding='same')(x)
x = Concatenate(axis=-1)([x, skip_2])
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(32, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = UpSampling2D()(x)

x = Conv2D(24, (3,3), padding='same')(x)
x = Concatenate(axis=-1)([x, skip_1])
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(24, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = UpSampling2D()(x)

x = Conv2D(16, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(16, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

decoded = Conv2D(1, (1, 1), activation='sigmoid', padding='same', strides=1)(x)

segmenter = Model(input_img, decoded)
segmenter.compile(loss=dice_loss,optimizer='adam',metrics=['accuracy'])
#%%
segmenter.summary()
from keras.utils import plot_model
plot_model(segmenter, show_shapes=True, to_file='segmenter_baseline.png')
#%%
with CustomObjectScope({'dice_loss': dice_loss}):
    segmenter=load_model('saved_models/SE_dice.h5')
#%%
           
def plotPrediction(model,generator):
    img, mask = next(generator)
    figSize=(3,3)
    print('=== intermediate plot ===')
    print('=== Input ===')
    plt.figure(figsize=figSize)
    plt.imshow(img[0])
    plt.show()
    print('=== Target ===')
    plt.figure(figsize=figSize)
    plt.imshow(mask[0,:,:,0],cmap='gray')
    plt.show()
    print('=== Network output ===')
    plt.figure(figsize=figSize)
    decoded = model.predict(img[0:1], verbose=1)
    plt.imshow(decoded[0,:,:,0],cmap='gray')
    plt.show()
    print('=== Predicted overlay ===')
    plt.figure(figsize=figSize)
    plt.imshow(img[0])
    plt.imshow(decoded[0,:,:,0], alpha=0.5)
    plt.show()
    
class interPlot(Callback):
    '''
    keras callback for plotting prediction results between training epochs
    '''
    def on_epoch_end(self, epoch, logs={}):
        plotPrediction(segmenter,test_generator)
#%%
mlb = MultiLabelBinarizer()
df, mlb = getDF(filter, mlb)
#segmentation DF
sdf = augmentDFSegmentation(df)
train, test = train_test_split(sdf, test_size=0.2, random_state=42)
#%%

#%%
def seperateFrontBack(img):
    img[img > 0] = 1
    return img[:,:,0:1]

image_data_generator_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=.1,
    height_shift_range=.1,
    horizontal_flip=True,
    fill_mode='constant', cval=0)

mask_data_generator_train = ImageDataGenerator(
    preprocessing_function=seperateFrontBack,    
    rescale=1.,
    rotation_range=5,
    width_shift_range=.1,
    height_shift_range=.1,
    horizontal_flip=True,
    fill_mode='constant', cval=0)

seed = 1

train_generator_images = image_data_generator_train.flow_from_dataframe(
    dataframe=train,
    directory='..//VOCdevkit/VOC2009/JPEGImages',
    x_col='filename',
    class_mode=None,
    color_mode="rgb",
    target_size=(image_size[1],image_size[0]),
    batch_size=batchSize,
    seed=seed)

train_generator_mask = mask_data_generator_train.flow_from_dataframe(
    dataframe=train,
    directory='..//VOCdevkit/VOC2009/SegmentationClass',
    x_col='segmentation',
    class_mode=None,
    color_mode="grayscale",
    target_size=(image_size[1],image_size[0]),
    batch_size=batchSize,
    seed=seed)

train_generator = zip(train_generator_images, train_generator_mask)
#%%
image_data_generator_test = ImageDataGenerator(
    rescale=1./255)

mask_data_generator_test = ImageDataGenerator(
    preprocessing_function=seperateFrontBack)

seed = 2

test_generator_images = image_data_generator_test.flow_from_dataframe(
    dataframe=test,
    directory='..//VOCdevkit/VOC2009/JPEGImages',
    x_col='filename',
    class_mode=None,
    color_mode="rgb",
    target_size=(image_size[1],image_size[0]),
    batch_size=batchSize,
    seed=seed)

test_generator_mask = mask_data_generator_test.flow_from_dataframe(
    dataframe=test,
    directory='..//VOCdevkit/VOC2009/SegmentationClass',
    x_col='segmentation',
    class_mode=None,
    color_mode="grayscale",
    target_size=(image_size[1],image_size[0]),
    batch_size=batchSize,
    seed=seed)

test_generator = zip(test_generator_images, test_generator_mask)
#%%
# show some images from the generator
images, masks = next(train_generator)

for i in range(batchSize):
    plt.imshow(images[i,:,:,:])
    plt.show()
    plt.imshow(masks[i,:,:,0],cmap='gray')
    plt.show()

#%%
#img, mask = next(train_generator)
#print('=== intermediate plot ===')
#decoded = segmenter.predict(img[0:1], verbose=1)[0]
#m = mask[0,:,:,0]
#%%
filepath = 'saved_models/SE_dice_fixed.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
tensorboard = TensorBoard(log_dir = "./logs/SE/{}".format(filepath))
earlyStop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=250, 
                          verbose=1)
callbacks_list = [checkpoint, tensorboard, earlyStop, interPlot()]
#%%
nb_epoch = 250
hist = segmenter.fit_generator(
        train_generator,
        steps_per_epoch=int(train_generator_images.n/batchSize),
        validation_data=test_generator,
        validation_steps=int(test_generator_images.n/batchSize),
        nb_epoch=nb_epoch,
        callbacks=callbacks_list,
        use_multiprocessing=True,
        workers=4)
#%%
#check manually, interPlot() is not working well
plotPrediction(segmenter,test_generator)
#%%
# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch], print_table=True)
        metric.append(value)
    return np.mean(metric)
#%%
yT = []
yP = []
for i in range(int(test_generator_images.n/batchSize)):
    n,y_true = next(test_generator)
    y_pred = segmenter.predict(n, verbose=1)
    yT.append(y_true)
    yP.append(y_pred[:,:,:,0])

print(iou_metric_batch(np.vstack(yT),np.vstack(yP)))
































