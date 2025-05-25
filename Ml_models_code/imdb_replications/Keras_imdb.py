import os

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_datasets as tfds

print("Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("Hub version: ", hub.__version__)

print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

train, validation,test = tfds.load(name="imdb_reviews",split=("train[:60%]","train[60%:]","test"),as_supervised=True)

#First ten examples

Features,labels = next(iter(train.batch(10)))

#Preparing model

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"

hub_layer = hub.KerasLayer(embedding, input_shape=[], 

                           dtype=tf.string, trainable=True)

hub_layer(Features[:3])

model = tf.keras.Sequential()

model.add(hub_layer)

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1))

#model.summary()

model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])

model.fit(train.shuffle(10000).batch(512),validation_data=validation.batch(512),epochs=10, verbose=1)

results = model.evaluate(test.batch(512),verbose=2)

 

for name,value in zip(model.metrics_names,results):

   print(name,":",value)

