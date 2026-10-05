import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

print("\n========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Information ==========")
df.info()

print("\n========== Statistical Summary ==========")
print(df.describe())

print("\n========== Dataset Shape ==========")
print(df.shape)

print("\n========== Missing Values ==========")
print(df.isnull().sum())

print("\n========== Duplicate Rows ==========")
print(df.duplicated().sum())

print("\n========== Target Distribution ==========")
print(df["target"].value_counts())

print("\n========== Target Names ==========")
print(data.target_names)

X = df.drop("target", axis=1)
y = df["target"]

print("\n========== Feature / Target Shapes ==========")
print("Feature Shape :", X.shape)
print("Target Shape  :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== Train/Test Shapes ==========")
print("Training Feature Shape :", X_train.shape)
print("Testing Feature Shape  :", X_test.shape)
print("Training Target Shape :", y_train.shape)
print("Testing Target Shape  :", y_test.shape)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions=model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

classification_rep = classification_report(y_test, predictions, target_names=data.target_names)
print("Classification Report:\n", classification_rep)

k_values = range(1, 21)
accuracies = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)


print("\n========== K vs Accuracy ==========")

for k, accuracy in zip(k_values, accuracies):
    print(f"K={k}: Accuracy={accuracy:.2f}")

    plt.figure(figsize=(8, 5))

plt.plot(k_values, accuracies, marker="o")

plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("KNN: K vs Accuracy")

plt.xticks(k_values)
plt.grid(True)

plt.show()

from sklearn.model_selection import cross_val_score

k_values = range(1, 21)
cv_scores = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_scores.append(scores.mean())


print("\n========== K vs Cross-Validation Accuracy ==========")

for k, score in zip(k_values, cv_scores):
    print(f"K={k}: CV Accuracy={score:.3f}")