import tensorflow as tf 

from keras.datasets import cifar10

from keras.models import Sequential 

from tensorflow.keras.utils import to_categorical as to_cat 

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

(x_train, y_train  ), (x_test , y_test )= cifar10.load_data()

print(x_train.shape)

print(x_test.shape)

print(x_train)

def conv(val):

    val = val /255.0

        return val

x_train, x_test = conv(x_train ), conv(x_test )

x_train.shape

y_train, y_test = to_cat(y_train), to_cat(y_test)

a = 'relu'

model = Sequential([

Conv2D(32,(5,5),activation=a, input_shape=(32, 32,3) ), 

MaxPooling2D((2,2) ), 

Dropout(0.3), 

Conv2D(64, (3,3) ,activation=a), 

MaxPooling2D((5,5)), 

Dropout(0.3),  

Flatten(), 

Dense(512, activation =a), 

Dense(256, activation =a), 

Dense(10,activation ='softmax') 

])

model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics =['accuracy'] )

model.fit(x_train,y_train,epochs=10)

loss, acc = model.evaluate(x_test, y_test) 

acc
