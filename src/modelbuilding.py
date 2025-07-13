import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x = pd.read_csv(r'C:\Users\Dell\Desktop\MLendtpend\data\preprocessdata\x.csv')
y = pd.read_csv(r'C:\Users\Dell\Desktop\MLendtpend\data\preprocessdata\y.csv')

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state= 42,test_size=.3, stratify=y)
x_train.shape, y_train.shape

scaler = StandardScaler()
x_train_scaled  = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_scaled, y_train)
predict = lr.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))

from sklearn.tree import DecisionTreeClassifier
lr = DecisionTreeClassifier()
lr.fit(x_train_scaled, y_train)
predict = lr.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))

from sklearn.ensemble import RandomForestClassifier
lr = RandomForestClassifier()
lr.fit(x_train_scaled, y_train)
predict = lr.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))

from sklearn.ensemble import AdaBoostClassifier
lr = AdaBoostClassifier()
lr.fit(x_train_scaled, y_train)
pridict = lr.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))

from sklearn.ensemble import GradientBoostingClassifier
lr = GradientBoostingClassifier()
lr.fit(x_train_scaled, y_train)
pridict = lr.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))

from sklearn.svm import SVC
lr = SVC()
lr.fit(x_train_scaled, y_train)
pridict = lr.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))

import pickle
with open (r'C:\Users\Dell\Desktop\MLendtpend\models\model.pkl','wb') as file:
    pickle.dump(lr, file)

with open (r'C:\Users\Dell\Desktop\MLendtpend\models\model.pkl','rb') as file:
    loaded_model = pickle.load(file)

loaded_model.predict(x_test_scaled)    

