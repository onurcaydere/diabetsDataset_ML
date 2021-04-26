import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

data=pd.read_csv("C:\\Users\\xxx\\Desktop\\Yapay Zeka Final Ödev\\Yapay_Proje\\data.csv")
df=data.copy()
print(df.head())
print(df.isnull())
print(df.info())
#Kaç adet hasta Sağlıklı var grafik
print(df["target"].value_counts())

#sns.countplot(x="target",hue="target",data=df)
plt.show()

#Cinsiyete Göre
#sns.countplot(x="sex",hue="target",data=df)
plt.show()

#Yaş Dağılımı
#sns.countplot(x="age",data=df)
plt.show()

#yasa göre hastalıklı olanlar

filtre = df["target"] ==1
print(df[filtre])
sns.countplot(x="age",hue="target",data=df[filtre])

