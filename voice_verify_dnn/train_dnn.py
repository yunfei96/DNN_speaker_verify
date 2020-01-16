from __future__ import absolute_import, division, print_function, unicode_literals
import pre_proc
import numpy as np
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#prepare data
(data, label) = pre_proc.build_train_data()
(test_data, test_label) = pre_proc.build_test_data()
data = np.asarray(data)
#data = np.expand_dims(data, -1)

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(input_shape=(957,101,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"),
#   tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1,1)),
#   tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
#   tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1,1)),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(256, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(len(label), activation='softmax')
# ])

model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(957,101,3), pooling='max', classes=len(label))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(np.asarray(data), np.asarray(label), batch_size=5, epochs=10)
model.evaluate(np.asarray(test_data), np.asarray(test_label), verbose=2)