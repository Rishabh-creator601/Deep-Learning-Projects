from keras.datasets import boston_housing

from keras.models import Sequential

from keras.layers import Dense

(x_train,y_train),(x_test,y_test)= boston_housing.load_data()

model = Sequential()

model.add(Dense(13,input_dim=13,kernel_initializer='normal',activation='sigmoid'))

model.add(Dense(6,activation='sigmoid',kernel_initializer='normal'))

model.add(Dense(1, activation='softmax',kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam', 

metrics=['mean_absolute_percentage_error'])

model.fit(x_train,y_train[30,epochs=5,batch_size=32)

loss,acc = model.evaluate(x_test,y_test)

print(acc)
