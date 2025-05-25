from keras.datasets import mnist

import matplotlib.pyplot as plt

from keras import layers,models

import tensorflow as tf

import keras

from keras.utils import to_categorical as to_cat

(x_train,y_train),(x_test,y_test) = mnist.load_data()

# Info_data

print(x_train.shape)

print(y_train.shape)

print(y_test.shape)

print(x_train[0])

print(y_train[0])

# Plotting images

def plot(v):

  plt.figure(figsize=(5,5))

  plt.xticks([])

  plt.yticks([])

  plt.imshow(x_train[v],cmap=plt.cm.binary)

  plt.xlabel(y_train[v])

  plt.show()

plot(4)

for v in range(26):

  plot(v)

# Model

model = models.Sequential()

#Preparing elements for model

def x_con(val):

  Ans = val.astype("float32") /255

  return Ans

x_train = x_con(x_train)

x_test = x_con(x_test)

x_train = x_train.reshape((60000,(28*28)))

x_test = x_test.reshape((10000,(28*28)))

print(x_train.shape,x_test.shape)

y_train = to_cat(y_train)

y_test = to_cat(y_test)

 

print(y_train)

#Neural_network layers

model.add(layers.Flatten(input_shape=(28*28,)))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(10,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=10)

test_loss ,test_acc = model.evaluate(x_test,y_test)

print(test_acc)
