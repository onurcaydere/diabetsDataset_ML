import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd



df=pd.read_csv("C:\\Users\\xxx\\Desktop\\Yapay Zeka Final Ödev\\Yapay_Proje\\data.csv")


X=df.drop(["target"],axis=1)
y=df["target"]



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.naive_bayes import GaussianNB
naive_bayes_Classifier = GaussianNB()
naive_bayes_Classifier.fit(X_train, y_train)

y_pred = naive_bayes_Classifier.predict(X_test)
pred=naive_bayes_Classifier.predict_proba(X_test)
y_compare = np.vstack((y_test,y_pred)).T

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(naive_bayes_Classifier,X_test,y_test,display_labels=["(0) Normal","(1) Hasta"],cmap=plt.cm.Blues)

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



