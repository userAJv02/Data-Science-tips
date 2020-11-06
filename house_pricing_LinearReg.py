import pandas as pd

base = pd.read_csv(r"C:\Users\Victor\Downloads\house_prices.csv")
base.drop(['id', 'date'], inplace=True, axis=1)
base.drop_duplicates(inplace=True)
X = base.iloc[:,1:19].values
y = base.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
score = reg.score(X_train, y_train)
previsao = reg.predict(X_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, previsao)


