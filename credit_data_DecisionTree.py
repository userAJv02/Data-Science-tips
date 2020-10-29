import pandas as pd

base = pd.read_csv(r"C:\Users\Victor\Downloads\credit_data.csv")

AgeAvg = base[base['age']>0]['age'].mean()
base.fillna(AgeAvg, inplace=True)
base.loc[base['age']<0, 'age']=40.9277

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(previsores_train, classe_train)
previsao = tree.predict(previsores_test)

from sklearn.metrics import accuracy_score, confusion_matrix
precisao = accuracy_score(classe_test, previsao)
matriz = confusion_matrix(classe_test, previsao)