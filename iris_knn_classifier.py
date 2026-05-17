# Noor Ul Huda
# DecodeLabs Internship Project 2
# Iris Flower Classification using KNN

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


print("====================================")
print(" DecodeLabs - AI Project 2")
print(" Submitted by: Noor Ul Huda")
print("====================================\n")


# loading iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# creating dataframe
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

print("First 5 rows of dataset:\n")
print(df.head())


# checking dataset info
print("\nDataset Information:\n")
print(df.info())

# checking missing values
print("\nMissing Values:\n")
print(df.isnull().sum())


# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\nTraining data size:", len(X_train))
print("Testing data size :", len(X_test))


# scaling data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData scaling completed")


# training model
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

print("KNN model trained successfully")


# predictions
predictions = model.predict(X_test)


# accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# confusion matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))


# classification report
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))


# visualization
plt.figure(figsize=(8, 5))

plt.scatter(
    df["sepal length (cm)"],
    df["petal length (cm)"],
    c=y
)

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Iris Flower Classification - Noor Ul Huda")

plt.show()


# custom prediction
sample = [[5.1, 3.5, 1.4, 0.2]]

sample = scaler.transform(sample)

result = model.predict(sample)

print("\nCustom Prediction Result:")
print("Predicted Flower Type:", iris.target_names[result][0])
