import pandas as pd
import numpy as np

nameFile = 'data.csv'
df = pd.read_csv(nameFile)
print(nameFile)
print(df.shape)
print(df.head())
print(df['Clase'].value_counts())

X = df.drop(['Emocion','Path','Clase'],axis=1)
y = df.Clase

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

print('X_train, X_test, y_train, y_test')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

oculta = int (len(X_train[1]) / 2)
print('oculta: ', oculta)


from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes= (oculta,oculta), activation='relu', max_iter=500, alpha=0.0001,
                     solver='adam', random_state=42,tol=0.000000001, verbose = True)

mlp.fit(X_train, y_train)

print("Score con training: %f" % mlp.score(X_train, y_train))
print("Score con test: %f" % mlp.score(X_test, y_test))

predictions = mlp.predict(X_test)

# #evaluar el algoritmo
from sklearn.metrics import classification_report, confusion_matrix

eti = ['positivo', 'negativo']
tn, fp, fn, tp = confusion_matrix(y_test,predictions,labels = eti).ravel()
print('tn: ', tn ,' fp: ', fp, ' fn: ',  fn, ' tp:', tp)
print(classification_report(y_test,predictions))
cm1 = pd.DataFrame(confusion_matrix(y_test,predictions), index = eti, columns = eti)
print('confusion_matrix')
print(cm1)
print(nameFile)
