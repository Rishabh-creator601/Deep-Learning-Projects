from keras.datasets import fashion_mnist

from keras import layers,models

from keras.utils import to_categorical as to_cat

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_test.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(5,5))

plt.imshow(x_train[500]) 

plt.colorbar()

plt.show()

def plot(v):

  plt.figure(figsize=(5,5))

  plt.imshow(x_train[v])

  plt.colorbar() 

  plt.xlabel(class_names[y_train[v]]) 

  plt.show()

plot(5555)

model = models.Sequential()

x_train = x_train.astype("float32")/255

x_test = x_test.astype("float32")/255

y_train,y_test = to_cat(y_train),to_cat(y_test)

model.add(layers.Flatten(input_shape=(28,28)))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=10)

test_loss,test_acc = model.evaluate(x_test,y_test)

print(test_acc)
