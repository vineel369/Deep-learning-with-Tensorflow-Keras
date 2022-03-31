import random
import numpy as np
import tensorflow as tf

from Unet import unet_model
from plots import loss_and_accuracy_plots

from config import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, seed, Train_data, epoch_num
from preprocess import process_train_data

np.random.seed = seed

train_data_path = Train_data

X_train, Y_train = process_train_data(train_data_path, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# image_x = random.randint(0, len(train_ids))
# Image.fromarray(X_train[image_x]).show()
# Image.fromarray(np.squeeze(Y_train[image_x])).show() #(128, 128, 1) to (128, 128)

model = unet_model(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True),
    tf.keras.callbacks.ModelCheckpoint('best_model_nuclei.h5', verbose=1, save_best_only=True)
]

results=model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=epoch_num, callbacks=callbacks)
loss_and_accuracy_plots(results)