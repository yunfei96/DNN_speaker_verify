from __future__ import absolute_import, division, print_function, unicode_literals
import pre_proc
import numpy as np
import tensorflow as tf

#prepare data
(data, label) = pre_proc.build_train_data()
#(test_data, test_label) = pre_proc.build_test_data()

size_of_input = len(data[1])

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(size_of_input, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(80, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(60, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(40, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(len(label), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(np.asarray(data), np.asarray(label), epochs=10)
#model.evaluate(test_data,  test_label, verbose=2)