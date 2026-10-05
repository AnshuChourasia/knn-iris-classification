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
* Using cross-validation to select the best value of K

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

The final model uses **K = 5**, meaning it considers the 5 nearest training samples when making a prediction.

### Feature Scaling

Since KNN is a distance-based algorithm, the features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then used to transform both the training and test data.

## 🔍 K Selection with Cross-Validation

Different values of K were evaluated using cross-validation to determine which value provided the best generalization performance.

|     K | CV Accuracy |
| ----: | ----------: |
|     1 |       94.2% |
|     2 |       95.8% |
|     3 |       95.8% |
|     4 |       95.8% |
| **5** |   **96.7%** |
|     6 |   **96.7%** |
|     7 |       95.8% |
|     8 |       95.8% |
|     9 |       95.8% |
|    10 |   **96.7%** |
|    11 |       95.8% |
|    12 |   **96.7%** |
|    13 |       95.0% |
|    14 |       95.8% |
|    15 |       95.0% |
|    16 |       95.8% |
|    17 |       95.8% |
|    18 |       94.2% |
|    19 |       94.2% |
|    20 |       93.3% |

The highest cross-validation accuracy was **96.7%**, achieved by K = 5, 6, 10, and 12.

**K = 5 was selected for the final model** because it achieved the maximum CV accuracy while keeping the model relatively simple and local.

## 📊 Model Evaluation

The model was evaluated using:

* Accuracy
* Confusion matrix
* Classification report

Cross-validation was also used to evaluate how model performance changes with different values of K.

## 📈 Key Learning Outcomes

Through this project, I practiced:

* Implementing KNN classification
* Understanding distance-based algorithms
* Feature scaling
* Train-test splitting
* Model evaluation
* Confusion matrices
* Classification reports
* Cross-validation
* Hyperparameter selection using K

## 📁 Project Structure

```text
knn-iris-classification/
│
├── knn.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python knn.py
```

## 📌 Conclusion

This project demonstrates how KNN can be used for multi-class classification and how **feature scaling and cross-validation** can improve the model selection process.

The final KNN model uses **K = 5** and achieved a maximum cross-validation accuracy of **96.7%** during K selection.
