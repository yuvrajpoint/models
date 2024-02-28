import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

dataset = pd.read_csv("MalwareDataset.csv", sep='|')
x = dataset.drop(columns=['legitimate', 'Name', 'md5'], axis=1)
y = dataset['legitimate']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)
joblib.dump(model, 'malware_detection_model_new.joblib')
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report")
print(report)