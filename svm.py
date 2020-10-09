from time import time #importamos la función time para capturar tiempos

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline


start_time = time()
#df = pd.read_csv("dataset.csv")
namefile = 'data_w.csv'
df = pd.read_csv(namefile)
print(namefile)
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

# Lts do data normalization 
# mean = np.mean(X_train, axis=0)
# std = np.std(X_train, axis=0)

# X_train = (X_train - mean)/std
# X_test = (X_test - mean)/std

# fin Lts do data normalization 

#Clasificar
start_time_trainer = time()
from sklearn.svm import SVC
model = SVC(kernel='linear',gamma = 'auto')
model.fit(X_train, y_train)

#almacenar modelo model #


# #predicciones
start_time_predicciones = time()
y_pred = model.predict(X_test)

# #evaluar el algoritmo
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

eti = ['positivo', 'negativo']
tn, fp, fn, tp = confusion_matrix(y_test,y_pred,labels = eti).ravel()
print('tn: ', tn ,' fp: ', fp, ' fn: ',  fn, ' tp:', tp)
# print('precision: ', precision_score(y_test, y_pred))
# print('recall: ', recall_score(y_test, y_pred))


# print(confusion_matrix(y_test,y_pred,labels = eti))


cm1 = pd.DataFrame(confusion_matrix(y_test,y_pred), index = eti, columns = eti)
print(cm1)


print(classification_report(y_test,y_pred))
end_time = time()
ejecucion = end_time - start_time
print('tiempo inicial: ', start_time)
print('tiempo final: ', end_time)
print('tiempo desde entrenamiento hasta final: ', end_time - start_time_trainer)
print('tiempo desde predecir: ', end_time - start_time_predicciones)
print('tiempo de ejecucion total: ', ejecucion)


print(namefile)
plt.figure(figsize = (10, 8))
#sns.heatmap(cm1, annot = True, cbar = False, fmt = 'g')
sns.heatmap(cm1, annot = True, fmt = 'd')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()
exit()