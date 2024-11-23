from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("output",exist_ok=True)

iris = load_iris()
x,y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)

clf = RandomForestClassifier()

clf.fit(x_train,y_train)

joblib.dump(clf, 'output/model.pkl', compress=9)