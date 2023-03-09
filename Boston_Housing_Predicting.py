import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
#### 1.import boston dataset########################
boston=load_boston()
dataset=pd.DataFrame(boston.data,columns=boston.feature_names)
#### 2. Prepare the dataset before model############
dataset['Price']=boston.target
## Independent and Dependent features
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

##### 3. Split the dataset into training and test
##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#############################################
## Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#############################################
import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))
############################################
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)
reg_pred=regression.predict(X_test)
##### 5. Evaluate the model and make predictions
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))

from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)
## Pickling The Model file For Deployment
import pickle
pickle.dump(regression,open('regmodel.pkl','wb'))
pickled_model=pickle.load(open('regmodel.pkl','rb'))
## Prediction
print(pickled_model.predict(scaler.transform(boston.data[0].reshape(1,-1))))