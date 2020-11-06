import pandas as pd

base = pd.read_csv(r"C:\Users\Victor\Downloads\USA_Housing.csv")
base.drop_duplicates(inplace=True)
base.drop(['Address'], inplace=True, axis=1)
X = base.iloc[:,0:5].values
y = base.iloc[:,5].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
previsao = reg.predict(X_test)
score = reg.score(X_train, y_train)

import matplotlib.pyplot as plt
plt.scatter(y_test, previsao)
plt.xlabel('y_test')
plt.ylabel('previsao')