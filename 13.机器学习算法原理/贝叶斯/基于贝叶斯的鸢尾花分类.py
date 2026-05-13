import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


dataset = pd.read_csv('./dataset/iris.data')

column_names = ['sepal_width','sepal_length','petal_width','petal_length','class']
dataset.columns = column_names

X = dataset.drop('class',axis = 1)
Y = dataset['class']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


nb_classifier = GaussianNB()

nb_classifier.fit(X_train_scaled,Y_train)

Y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(Y_test,Y_pred)

conf_matrix = confusion_matrix(Y_test,Y_pred)

class_report = classification_report(Y_test,Y_pred)

print(f'Accuracy:{accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classfication Report:{class_report}')