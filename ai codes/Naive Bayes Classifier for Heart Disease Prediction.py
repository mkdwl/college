# practical_7_naive_bayes.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset and encode categorical variables
df = pd.read_csv('C:/Users/nupur/sem 4 journals/AI/heart.csv')
le = LabelEncoder()
df[['Sex', 'ChestPainType', 'RestingECG', 'Diagnosis']] = df[['Sex', 'ChestPainType', 'RestingECG', 'Diagnosis']].apply(le.fit_transform)

# Display data info and plot counts
print(df.head(11), df.info())
for col in ['Sex', 'ChestPainType', 'RestingECG', 'Diagnosis']:
    sns.countplot(x=df[col])
    plt.title(f"Count of '{col}'")
    plt.show()

# Split data and train model
X, y = df.drop('Diagnosis', axis=1), df['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
