import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import pickle
#### 1.import boston dataset########################
boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['MEDV'] = boston_dataset.target

#### 2. Prepare the dataset before model############
X = data.drop('MEDV', axis=1).to_numpy()
y = data['MEDV'].to_numpy()

##### 3. Split the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
##### 4. Choose the Model
regressor = SVR(kernel='rbf', C=10)#, gamma=0.1)
regressor.fit(X_train, y_train)

##### 5. Evaluate the model and make predictions
y_pred = regressor.predict(X_test)
print(F'The fit of trained data: {r2_score(y_true= y_test , y_pred= y_pred)}')
######################
### pickling the model file for deployment
######################

