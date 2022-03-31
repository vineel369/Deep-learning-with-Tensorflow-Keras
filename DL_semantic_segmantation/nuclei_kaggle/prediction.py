import random
import numpy as np
from PIL import Image
import tensorflow as tf

from config import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, seed, Test_data, Train_data
from preprocess import process_train_data, process_test_data

np.random.seed = seed

X_train, Y_train = process_train_data(Train_data, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
X_test = process_test_data(Test_data, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = tf.keras.models.load_model('best_model_nuclei.h5')

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random samples
idx = random.randint(0, len(preds_train_t))
Image.fromarray(X_train[idx]).show()
Image.fromarray(np.squeeze(Y_train[idx])).show()
Image.fromarray(np.squeeze(preds_train_t[idx])).show()

ix = random.randint(0, len(preds_val_t))
Image.fromarray(X_train[int(X_train.shape[0]*0.9):][ix]).show()
Image.fromarray(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix])).show()
Image.fromarray(np.squeeze(preds_val_t[ix])).show()

