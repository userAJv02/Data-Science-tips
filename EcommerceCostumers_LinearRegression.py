import pandas as pd

base = pd.read_csv(r"C:\Users\Victor\Downloads\Ecommerce Customers.csv")
base.drop_duplicates(inplace=True)
base.drop(['Email', 'Address', 'Avatar'], inplace=True, axis=1)
base.isnull().sum()

import seaborn as sns
sns.pairplot(base)

X = base.iloc[:,0:4].values
y = base.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
score = reg.score(X_train, y_train)
previsao = reg.predict(X_test)

coefs=reg.coef_
interc=reg.intercept_

import matplotlib.pyplot as plt
plt.scatter(y_test, previsao)
