import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

x=np.arange(1,20).reshape(-1,1)
y=np.array([c**2 + random.randint(-4,4)*2.5*c**0.75 for c in x]).reshape(-1,1)
X=x
n_params=6
for i in range(2,n_params):
    X=np.concatenate((X,x**i),axis=-1)
X=pd.DataFrame(X)
print(X)
X.columns=['x^'+str(i) for i in range(1,n_params)]
print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

model_1=LinearRegression()
model_2=LinearRegression()
model_3=LinearRegression()

model_1.fit(np.array(X_train['x^1']).reshape(-1,1), y_train)
model_2.fit(X_train[['x^1','x^2']],y_train)
model_3.fit(X_train,y_train)
print(model_1.score(np.array(X_test['x^1']).reshape(-1,1), y_test))
print(model_2.score(X_test[['x^1','x^2']],y_test))
print(model_3.score(X_test,y_test))

plt.scatter(x,y)
plt.plot(x.reshape(-1,1),model_1.predict(np.array(X['x^1']).reshape(-1,1)))
plt.plot(x.reshape(-1,1),model_2.predict(np.array(X[['x^1','x^2']])))
plt.plot(x.reshape(-1,1),model_3.predict(np.array(X)))
plt.show()
