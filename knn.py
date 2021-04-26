import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


df=pd.read_csv("C:\\Users\\xxx\\Desktop\\Yapay Zeka Final Ödev\\Yapay_Proje\\data.csv")

df.head()


X=df.drop(["target"],axis=1)
y=df["target"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
#Ölçeklendirme
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

import math
#toplam veri sayısının karekökü sayesinde k değerini seçeceğim.
#k çıkan sonuca en yakın tek sayı olmalı 
print (math.sqrt(len(X.values)))  #====17.406895185529212 çıktı verdi sonuç k=17 olarak alındı

classifier=KNeighborsClassifier(n_neighbors=17,metric='euclidean')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier,X_test,y_test,display_labels=["(0) Normal","(1) Hasta"],cmap=plt.cm.Blues)

cm=confusion_matrix(y_test,y_pred)
print(cm)
#Accuracy, Sensitivity, Specificity, Recall, Precision
print("Sensivity")
print(cm[0][0]/(cm[0][1]+cm[0][0]))
print("Accuracy")
print((cm[0][0]+cm[1][1])/X_test.shape[0])
print("Specificity")
print(cm[1][1]/(cm[1][0]+cm[1][1]))
print("Recall")
print(cm[0][0]/(cm[0][1]+cm[0][0]))
print("Precision")
print(cm[0][0]/(cm[0][0]+cm[1][0]))


