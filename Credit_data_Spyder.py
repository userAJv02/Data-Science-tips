import pandas as pd
import numpy as np

base = pd.read_csv(r"C:\Users\Victor\Downloads\credit_data.csv")

c = base[base['age']>0]['age'].mean()

base.fillna(c, inplace=True)

base.loc[base.age<0, 'age']=40.9277

previsores = base.iloc[:,1:4]
classe = base.iloc[:,4]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split

prev_train, prev_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(prev_train, classe_train)
previsoes = nb.predict(prev_test)

from sklearn.metrics import accuracy_score, confusion_matrix

precisao = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)
