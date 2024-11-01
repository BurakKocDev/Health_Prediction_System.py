import pandas as pd


df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")

print(df)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Drop patientid as it is not useful for modeling
df = df.drop(columns=['patientid'])

# Split features and target
X = df.drop(columns=['target'])
y = df['target']

# Standardize numerical columns
numerical_cols = ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of training and testing data
X_train.shape, X_test.shape, y_train.shape, y_test.shape



print(df.columns)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize models

rf_clf = RandomForestClassifier(random_state=42)


# Train and evaluate models
models = { 'Random Forest': rf_clf}
results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

print(results)



import joblib

# Random Forest modelini kaydet
joblib.dump(rf_clf, 'random_forest_cardiovascular_model.pkl')

# Scaler'ı kaydet
joblib.dump(scaler, 'scaler_cardiovascular.pkl')
print("kaydedildi")