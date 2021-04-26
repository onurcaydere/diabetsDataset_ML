import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\xxx\\Desktop\\Yapay Zeka Final Ödev\\Yapay_Proje\\data.csv")
y=df["target"]
X=df.drop(["target"],axis=1)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(solver="liblinear",max_iter=500)
logmodel.fit(X_train,y_train)
pred=logmodel.predict(X_test)
print("Teta 0 Değeri:  ")

print(logmodel.intercept_) # Q0 değeri
print("Geri kalan teta değerleri :" )
print(logmodel.coef_) #geri kalan teta değerleri



from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(logmodel,X_test,y_test,display_labels=["(0) Normal","(1) Hasta"],cmap=plt.cm.Blues)
cm=confusion_matrix(y_test,pred)
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








