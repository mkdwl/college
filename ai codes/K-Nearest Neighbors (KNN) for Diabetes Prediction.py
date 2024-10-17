import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, 
roc_auc_score 
 
# Load the diabetes dataset 
df = pd.read_csv('diabetes.csv') 
print(df.head()) 
print(df.shape) 
print(df.dtypes) 
 
# Split the data into features (X) and target (y) 
X, y = df.drop('Outcome', axis=1), df['Outcome'] 
 
# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
random_state=42) 
 
# Train and evaluate k-NN for different values of k 
neighbors = np.arange(1, 10) 
train_accuracy = [] 
test_accuracy = [] 
 
for k in neighbors: 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train) 
    train_accuracy.append(knn.score(X_train, y_train)) 
    test_accuracy.append(knn.score(X_test, y_test)) 
 
# Plot the accuracy 
plt.title('k-NN varying number of neighbors') 
plt.plot(neighbors, train_accuracy, label='Training accuracy') 
plt.plot(neighbors, test_accuracy, label='Testing accuracy') 
plt.legend() 
plt.xlabel('Number of neighbors (k)') 
plt.ylabel('Accuracy') 
plt.show() 
 
# Fit k-NN with the best k and evaluate 
best_k = 7 
knn = KNeighborsClassifier(n_neighbors=best_k) 
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test) 
 
# Confusion matrix and classification report 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred)) 
 
# ROC Curve 
y_pred_proba = knn.predict_proba(X_test)[:, 1] 
fpr, tpr, _ = roc_curve(y_test, y_pred_proba) 
plt.plot([0, 1], [0, 1], 'k--') 
plt.plot(fpr, tpr, label='k-NN (n_neighbors=7)') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend() 
plt.show() 
 
# AUC Score 
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba)) 
 
# Hyperparameter tuning with GridSearchCV 
param_grid = {'n_neighbors': np.arange(1, 50)} 
knn = KNeighborsClassifier() 
knn_cv = GridSearchCV(knn, param_grid, cv=5) 
knn_cv.fit(X, y) 
 
print("Best Score:", knn_cv.best_score_) 
print("Best Params:", knn_cv.best_params_) 
