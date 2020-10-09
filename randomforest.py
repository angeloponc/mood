import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

nameFile = 'data.csv'
df = pd.read_csv(nameFile)
print(nameFile)
print(df.shape)
print(df.head())
print(df['Clase'].value_counts())

X = df.drop(['Emocion','Path','Clase'],axis=1)
y = df.Clase

# separar entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

print('XX_train, X_test, y_train, y_test')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#defino el algoritmo a utilizar
#bosques aleatorios
from sklearn.ensemble import RandomForestClassifier
algoritmo = RandomForestClassifier(n_estimators = 100, random_state = 42)

#entreno el modelo
algoritmo.fit(X_train, y_train)

#realizo una predicción
y_pred = algoritmo.predict(X_test)

#evalúo el algoritmo
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

eti = ['positivo', 'negativo']
tn, fp, fn, tp = confusion_matrix(y_test,y_pred,labels = eti).ravel()
print('tn: ', tn ,' fp: ', fp, ' fn: ',  fn, ' tp:', tp)

cm1 = pd.DataFrame(confusion_matrix(y_test,y_pred), index = eti, columns = eti)
print(cm1)

print(classification_report(y_test,y_pred))
print(nameFile)
