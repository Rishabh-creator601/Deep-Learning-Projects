print('loading libraries!')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd




data = load_iris()
x = data.data
y = data.target

state = 42
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=state)

model = RandomForestRegressor(n_estimators=100,max_depth=4,random_state=state)
model.fit(x_train,y_train) print(model.score(x_test,y_test))
print(model.predict([[6,3,4,2]]))





