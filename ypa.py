import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df=pd.read_csv("E:\\MASAÜSTÜ\\Ders Kitap ve Notları\\4. Sınıf 1. DÖNEM\\Yapay Zeka Final Ödev\Yapay_Proje\\data.csv")

X=df.drop(["target"],axis=1)
y=df["target"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

clf = MLPClassifier(activation="relu",alpha=0.001,solver="sgd",hidden_layer_sizes=(100,100,100))

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf,X_test,y_test,display_labels=["(0) Normal","(1) Hasta"],cmap=plt.cm.Blues)

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