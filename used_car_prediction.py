from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

used_car_path=r"C:\Users\neerr\Documents\Datasets\archive-Used_cars\true_car_listings.csv"

used_car_data=pd.read_csv(used_car_path)

features=['Year','Mileage','State','Make','Model']

used_car_data=used_car_data.dropna(axis=0)

X=pd.get_dummies(used_car_data[features],drop_first=True)
y=used_car_data['Price']

train_X, test_X, train_y, test_y=train_test_split(X,y, random_state=1)

used_car_model=DecisionTreeRegressor(max_depth=10,random_state=1)

used_car_model.fit(train_X,train_y)

test_predictions=used_car_model.predict(test_X)

print("MAE : ",mean_absolute_error(test_y,test_predictions))
