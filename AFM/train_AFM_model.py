#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from data_generator import file_generator
from AFM_Model import AFM




# 1. Load data

train_path = "train.txt"
val_path = "test.txt"
batch_size = 1000

n_train = sum([1 for i in open(train_path)])
n_val = sum([1 for i in open(val_path)])

train_steps = n_train / batch_size
train_steps_ = n_train // batch_size
validation_steps = n_val / batch_size
validation_steps_ = n_val // batch_size


train_generator = file_generator(train_path, batch_size)
val_generator = file_generator(val_path, batch_size)

steps_per_epoch = train_steps_ if train_steps==train_steps_ else train_steps_ + 1
validation_steps = validation_steps_ if validation_steps==validation_steps_ else validation_steps_ + 1

print("n_train: ", n_train)
print("n_val: ", n_val)

print("steps_per_epoch: ", steps_per_epoch)
print("validation_steps: ", validation_steps)




# 2. Train model

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
callbacks = [early_stopping_cb]


model = AFM()
print(model.summary())
tf.keras.utils.plot_model(model, to_file='AFM_model.png', show_shapes=True)


model.compile(loss='binary_crossentropy', \
    optimizer=Adam(lr=1e-3), \
    metrics=['accuracy'])


history = model.fit(train_generator, \
                    epochs=1, \
                    steps_per_epoch = steps_per_epoch, \
                    callbacks = callbacks,
                    validation_data = val_generator, \
                    validation_steps = validation_steps, \
                    shuffle=True
                   )



model.save_weights('AFM_model.h5')
