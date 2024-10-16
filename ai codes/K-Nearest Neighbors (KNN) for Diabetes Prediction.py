import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Load dataset and split into features and target
df = pd.read_csv('diabetes.csv')
print("Data Preview:\n", df.head(), "\n")
print("Data Shape:", df.shape)
print("Data Types:\n", df.dtypes)

# Split data into features and target
X, y = df.drop('Outcome', axis=1), df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train and evaluate k-NN for k from 1 to 9
train_acc = [KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_train, y_train) for k in range(1, 10)]
test_acc = [KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test) for k in range(1, 10)]

# Plot accuracy
plt.plot(range(1, 10), train_acc, label='Train Acc')
plt.plot(range(1, 10), test_acc, label='Test Acc')
plt.title('k-NN Accuracy for Varying k'), plt.xlabel('k'), plt.ylabel('Accuracy'), plt.legend(), plt.show()

# Fit best k-NN model and evaluate
knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred), "\n", classification_report(y_test, y_pred))

# ROC Curve and AUC Score
y_proba = knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='k-NN (n_neighbors=7)')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR'), plt.ylabel('TPR'), plt.title('ROC Curve'), plt.legend(), plt.show()
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Hyperparameter tuning with GridSearchCV
best_knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': np.arange(1, 50)}, cv=5).fit(X, y)
print(f"Best Score: {best_knn.best_score_}, Best Params: {best_knn.best_params_}")
