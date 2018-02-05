import pandas as pd
import numpy as np


dataset=pd.read_csv('/home/kartik/Documents/data_set.csv')
y=dataset['Exited'].values
dataset=dataset.drop(['RowNumber','Surname','CustomerId','Exited'],axis=1)
dataset=dataset.values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
dataset[:, 1]=le.fit_transform(dataset[:, 1])
le2=LabelEncoder()
dataset[:, 2]=le2.fit_transform(dataset[:, 2])
ohe=OneHotEncoder(categorical_features = [1])
dataset=ohe.fit_transform(dataset).toarray()
dataset=dataset[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn import svm
model=svm.SVC(C=25.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.12, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
model.fit(X_train,y_train)
training_accuracy=model.score(X_train,y_train)

prediction=model.predict(X_test)

error=np.mean(y_test!=prediction)
error

from sklearn.metrics import accuracy_score
testing_accuracy=accuracy_score(y_test,prediction)

