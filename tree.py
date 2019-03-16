import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv('loan_data.csv')
loans.info()
print(loans.describe())
print(loans.head())

#exploratory data analysis
sns.set()
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue', bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red', bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue', bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red', bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

plt.figure(figsize=(10,6))
sns.countplot('purpose', data=loans, hue= 'not.fully.paid')
plt.show()

sns.jointplot('fico','int.rate',data=loans)
plt.show()

sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy', col='not.fully.paid',scatter_kws={'s':10})
plt.show()

#categorical features
feat = ['purpose']
final_data = pd.get_dummies(loans,columns=feat,drop_first=True)
final_data.info()

#train test split
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=55)

#decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))