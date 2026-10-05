# KNN Iris Classification

A machine learning classification project using **K-Nearest Neighbors (KNN)** to classify Iris flowers into three species using the Iris dataset and `scikit-learn`.

## 📌 Project Overview

This project demonstrates the complete workflow of a KNN classification model:

* Loading the Iris dataset
* Exploring and inspecting the data
* Checking missing values and duplicates
* Separating features and target
* Splitting data into training and testing sets
* Standardizing features using `StandardScaler`
* Training a KNN classifier
* Making predictions
* Evaluating model performance
* Visualizing the confusion matrix
* Generating a classification report

## 🌸 Dataset

The project uses the built-in **Iris dataset** from `scikit-learn`.

The dataset contains **150 samples** and 4 features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The target contains three Iris species:

* Setosa
* Versicolor
* Virginica

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## 🤖 Machine Learning Model

### K-Nearest Neighbors (KNN)

The model is implemented using:

```python
KNeighborsClassifier(n_neighbors=5)
```

The model uses **K = 5**, meaning it considers the 5 nearest training samples when making a prediction.

### Feature Scaling

Since KNN is a distance-based algorithm, the features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler
