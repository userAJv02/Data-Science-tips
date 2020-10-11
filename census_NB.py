import pandas as pd
import numpy as np

base = pd.read_csv(r"C:\Users\Victor\Downloads\census.csv")
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

column = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = column.fit_transform(previsores).toarray()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split

prev_train, prev_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.naive_bayes import GaussianNB   

nb = GaussianNB()
nb.fit(prev_train, classe_train)
previsao = nb.predict(prev_test)

from sklearn.metrics import accuracy_score, confusion_matrix

precisao = accuracy_score(classe_test, previsao)
matriz = confusion_matrix(classe_test, previsao)