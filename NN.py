import os

import numpy as np
from PIL import Image
import tensorflow as tf
from skimage import transform

import keras
from keras import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50V2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True,
                               rotation_range=20, width_shift_range=.2,
                               height_shift_range=.2, zoom_range=.2)
''' Here and further I commented fragments of code which we are not using having downloaded my pretrained model'''
# valid_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.4)

train_data = train_gen.flow_from_directory(directory='Original Images/Original Images',
                                           target_size=(256, 256), shuffle=True, class_mode='binary')

# valid_data = valid_gen.flow_from_directory(directory='Original Images/Original Images',
#                                            target_size=(256, 256), shuffle=True,
#                                            subset='training', class_mode='binary')
#
# test_data = valid_gen.flow_from_directory(directory='Original Images/Original Images',
#                                           target_size=(256, 256), shuffle=True,
#                                           subset='validation', class_mode='binary')

# label_map = train_data.class_indices
label_map = {'Monkey Pox': 0, 'Others': 1}
''' This the model I use for classification '''
# with tf.device('/GPU:0'):
#     base_model = ResNet50V2(
#         include_top=False,
#         input_shape=(256, 256, 3)
#     )
#     base_model.trainable = False
#
#     model = Sequential([
#         base_model,
#         GlobalAveragePooling2D(),
#         Dense(256, activation='relu'),
#         BatchNormalization(),
#         Dense(164, activation='relu'),
#         BatchNormalization(),
#         Dense(1, activation='sigmoid')
#     ])
#
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer='Adam',
#         metrics=['accuracy']
#     )
#
#     cb = [EarlyStopping(patience=5, monitor='val_accuracy', mode='max', restore_best_weights=True),
#           ModelCheckpoint("ResNet50V2-01.h5", save_best_only=True)]
#
#     model.fit(
#         train_data,
#         epochs=50,
#         validation_data=valid_data,
#         callbacks=cb
#     )

model = keras.models.load_model('ResNet50V2-01.h5')

''' Function to reshape input image to the standart shape for classification model'''
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

'''Loop to download absolute path to image +image name from your OS directory as a string value'''
while True:
    print('Print absolute path to your image in console and press ENTER')
    image = load(str(input()))
    y_prob = model.predict(image)
    y_classes = y_prob.argmax(axis=-1)
    if y_prob < .5:
        result = 100 - (100 * y_prob)
    elif y_prob > .5:
        result = y_prob * 100
    else:
        result = 50
    print(list(label_map.keys())[list(label_map.values()).index(y_classes)],
          'with probability: ', result[0][0], '%')
