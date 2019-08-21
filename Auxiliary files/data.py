from lxml import etree
import numpy as np
import os
from skimage import io
from skimage.transform import resize
import random
from shutil import copyfile
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

voc_root_folder = "../VOCdevkit"  # please replace with the location on your laptop where you unpacked the tarball

def ReadData(filter, image_size, offline=None):
    '''
    @param offline: if true, constructs an offline folder in /Temp_VOC_folders
    '''
    # parameters that you should set before running this script
        # select class, this default should yield 1489 training and 1470 validation images

    # step1 - build list of filtered filenames
    annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
    annotation_files = os.listdir(annotation_folder)
    filtered_filenames = []
    for a_f in annotation_files:
        tree = etree.parse(os.path.join(annotation_folder, a_f))
        if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
            filtered_filenames.append(a_f[:-4])

    # step2 - build (x,y) for TRAIN/VAL (classification)
    classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
    classes_files = os.listdir(classes_folder)
    train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt+'_' in c_f and '_train.txt' in c_f]
    val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt+'_' in c_f and '_val.txt' in c_f]
    if not offline:
        x_train, y_train = build_classification_dataset(train_files,filter, image_size)
        x_val, y_val = build_classification_dataset(val_files,filter, image_size)
        
        return x_train, y_train, x_val, y_val
    else:
        build_offline_folder('train',train_files,filter, image_size)
        build_offline_folder('test',val_files,filter, image_size)
        
def getDF(filter, mlb):
    '''
    @param offline: if true, constructs an offline folder in /Temp_VOC_folders
    '''
    # parameters that you should set before running this script
        # select class, this default should yield 1489 training and 1470 validation images

    # step1 - build list of filtered filenames
    annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
    annotation_files = os.listdir(annotation_folder)
    filtered_filenames = []
    for a_f in annotation_files:
        tree = etree.parse(os.path.join(annotation_folder, a_f))
        if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
            filtered_filenames.append(a_f[:-4])
            
    # step2 - build (x,y) for TRAIN/VAL (classification)
    classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
    classes_files = os.listdir(classes_folder)
    train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt+'_' in c_f and '_trainval.txt' in c_f]
    
    df = buildDF(filter, train_files)
    df = df.to_frame()
    df.reset_index(level=0, inplace=True)  
    df['index'] = df.index
    df['tags'] = [tuple(x[1:-1].split(',')) for x in df['tags']]
    
    mlb.fit(df['tags'].values.tolist())
    df['tags'].values.tolist()
    print('found class labels:')
    print(mlb.classes_)

    return (df,mlb)
    
def buildDF(filter, list_of_files):
    temp = []
    t_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt+'_' in f_cf][0]
            t_labels.append(len(temp[-1]) * [label_id])
    t_filter = [item for l in temp for item in l]
    
    df = pd.DataFrame(columns=['filename','tags'])
    
    for fn,label in zip(t_filter,np.hstack(t_labels)):
        #fileloc = os.path.join(voc_root_folder, "VOC2009/JPEGImages/",fn)
        df = df.append({'filename':fn+'.jpg','tags':str(label)}, ignore_index=True)
    dfT = df.groupby(['filename'])['tags'].apply(lambda x: "(%s)" % ','.join(x))
    
    return(dfT)    

def augmentDFSegmentation(df):
    df['segmentation'] = ''
    with open('../VOCdevkit/VOC2009/ImageSets/Segmentation/trainval.txt') as file:
        lines = file.read().splitlines()
    for i in range(len(df)):
        prefix = df.loc[i]['filename'][:-4]
        if prefix in lines:
            lines.remove(prefix)
            seg = prefix + '.png'
            df.at[i,'segmentation'] = seg
        else:
            df.at[i,'segmentation'] = np.nan
    sdf = df.dropna()
    return sdf

def build_classification_dataset(list_of_files, filter,image_size):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    print(list_of_files)

    temp = []
    t_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            t_labels.append(len(temp[-1]) * [label_id])
    t_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in t_filter for file in os.listdir(image_folder) if
                       f in file]
    x = np.array([resize(io.imread(img_f), (image_size[1], image_size[0], 3)) for img_f in image_filenames]).astype(
        'float32')
    # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in t_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)
    
    return x, y
        
def build_offline_folder(dest, list_of_files, filter, image_size):
    '''
    Builds an offline folder for Keras data flow
    '''
    
    print(list_of_files)

    temp = []
    t_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            t_labels.append(len(temp[-1]) * [label_id])
    t_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in t_filter for file in os.listdir(image_folder) if
                       f in file]
    
#    x = np.array([resize(io.imread(img_f), (image_size[1], image_size[0], 3)) for img_f in image_filenames]).astype(
#        'float32')
    
    print('flushing temporary folders')
    # missing, you'll have to delete the folders manually
    print('writing to offline folders')
    temp_folder = "../Temp_VOC_folders" 
    images = os.path.join(temp_folder, dest)
    try:
        os.mkdir(images)
    except:
        print('already exists')
    
    for f in filter:
        try:
            os.mkdir(os.path.join(images, f))
        except:
            print('already exists')
    
    lbls = np.hstack(t_labels)
    for i in range(len(lbls)):
#        x = resize(io.imread(image_filenames[i]), (image_size[1], image_size[0], 3))
#        io.imsave(os.path.join(images, f'{filter[lbls[i]]}/{t_filter[i]}.jpg'),x)
        target = '{}/{}.jpg'.format(filter[lbls[i]],t_filter[i])
        copyfile(image_filenames[i],os.path.join(images, target))
        x = resize(io.imread(os.path.join(images, target)), (image_size[1], image_size[0], 3))
        io.imsave(os.path.join(images, target),x)

def reshuffle_offline_folder(filter, ratio):
    print('======= Shuffling ========')
    offline_folder = "../Temp_VOC_folders"
    split = os.listdir(offline_folder)
    counts = []
    for a in split:
        for f in filter:
            folder = os.path.join(offline_folder,a,f)
            files = os.listdir(folder)
            counts.append(len(files))
    print('counters for TEST - TRAIN (per class)')
    print(counts)
    print('Total count: {}'.format(np.sum(counts)))
    pivot = int(len(counts)/2)
    sums = [counts[j]+counts[j+pivot] for j in range(pivot)]
    deltas = [0.2*sums[j] for j in range(pivot)]
    selections = [counts[j]-int(deltas[j]) for j in range(pivot)]
    
    source = "../Temp_VOC_folders/test"
    destination = "../Temp_VOC_folders/train"
    
    random.seed(4)
    
    selCounter = 0
    for f in filter:
        folder = os.path.join(source,f)
        files = os.listdir(folder)
        print("selecting {} instances from test folder".format(selections[selCounter]))
        sampling = random.sample(files, selections[selCounter])
        for s in sampling:
            os.rename(os.path.join(source,f,s),os.path.join(destination,f,s))
        selCounter += 1
        
    counts = []
    for a in split:
        for f in filter:
            folder = os.path.join(offline_folder,a,f)
            files = os.listdir(folder)
            counts.append(len(files))
    print('Results')
    print('counters for TEST - TRAIN (per class)')
    print(counts)
    print('Total count: {}'.format(np.sum(counts)))
    
def buildSegmentationFolders(mirror, image_size):
    print('=== Building offline segmentation folders ===')
    temp_folder = "../Temp_VOC_folders"
    segmentation = os.path.join(temp_folder, '{}Segmentation'.format(mirror))
    os.mkdir(segmentation)
    masks = os.path.join(temp_folder, '{}Masks'.format(mirror))
    os.mkdir(masks)
    combinedOrig = os.path.join(segmentation, 'combined')
    combinedMask = os.path.join(masks, 'combined')
    os.mkdir(combinedOrig)
    os.mkdir(combinedMask)
    
    source = os.path.join(temp_folder, mirror)
    filters = os.listdir(source)
    sourceFiles = []
    for f in filters:
        folder = os.path.join(source,f)
        files = os.listdir(folder)
        for file in files:
            sourceFiles.append(file[:-4])
    
    maskSource = '../VOCdevkit/VOC2009/SegmentationObject'
    maskFiles = os.listdir(maskSource)

    selectedMasks = []
    for m in maskFiles:
        if m[:-4] in sourceFiles:
            selectedMasks.append(m[:-4])
            maskUrl = os.path.join(maskSource,m)
#            x = resize(io.imread(maskUrl), (image_size[1], image_size[0], 3))
#            io.imsave(os.path.join(combinedMask, '{}.png'.format(m[:-4])),x)
            copyfile(maskUrl,os.path.join(combinedMask,m))
            
    for f in filters:
        folder = os.path.join(source,f)
        files = os.listdir(folder)
        for file in files:
            if file[:-4] in selectedMasks:
                copyfile(os.path.join(folder,file),os.path.join(combinedOrig,file))
               
        
def explore_image_sizes(filter):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    # step1 - build list of filtered filenames
    annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
    annotation_files = os.listdir(annotation_folder)
    filtered_filenames = []
    for a_f in annotation_files:
        tree = etree.parse(os.path.join(annotation_folder, a_f))
        if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
            filtered_filenames.append(a_f[:-4])

    # step2 - build (x,y) for TRAIN/VAL (classification)
    classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
    classes_files = os.listdir(classes_folder)
    train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt+'_' in c_f and '_train.txt' in c_f]
    val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt+'_' in c_f and '_val.txt' in c_f]
    x_train, y_train = get_shapes(train_files,filter)
    x_val, y_val = get_shapes(val_files,filter)
    
    x_shapes = np.vstack([x_train,x_val])
    labels = np.vstack([y_train,y_val])
    
    return x_shapes, labels

def get_shapes(list_of_files, filter):
    temp = []
    train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    x = []
    for img_f in image_filenames:
        x.append(np.asarray(io.imread(img_f)).shape)
    # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y
        
    
    






