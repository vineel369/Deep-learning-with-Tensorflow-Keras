import os
import numpy as np
from PIL import Image

from tqdm import tqdm

def process_train_data(Train_data, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    print('Resizing Train images & masks')
    train_ids = next(os.walk(Train_data))[1]
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT,IMG_WIDTH, 1), dtype=np.bool)
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = Train_data + id_
        img = Image.open(path + '/images/' + id_ +'.png')
        img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
        img_arr = np.array(img)[:,:,:IMG_CHANNELS]
        X_train[n] = img_arr
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path+'/masks/'))[2]:
            mask_ = Image.open(path+'/masks/'+mask_file)
            mask_ = mask_.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
            mask_arr = np.expand_dims(np.array(mask_), axis=-1)
            mask = np.maximum(mask, mask_arr)
        Y_train[n] = mask
    return X_train, Y_train

def process_test_data(Test_data, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    test_ids = next(os.walk(Test_data))[1]
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []   
    print('Resizing test images')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = Test_data + id_
        img = Image.open(path + '/images/' + id_ + '.png')
        sizes_test.append([img.size[0], img.size[1]])
        img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
        img_arr = np.array(img)[:,:,:IMG_CHANNELS]
        X_test[n] = img_arr
        return X_test