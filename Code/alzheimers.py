import pandas as pd

data = pd.read_csv("alzheimers_disease_data.csv")




data_info = data.info()
data_head = data.head()

print(data.info())
print(data.head())


missing_values = data.isnull().sum()

data_cleaned = data.drop(columns=['PatientID', 'DoctorInCharge'])

print(missing_values)
print(data_cleaned.head())
print(data_cleaned.columns)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


X = data_cleaned.drop('Diagnosis', axis=1)
y = data_cleaned['Diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


random_forest_model = RandomForestClassifier(random_state=42)


random_forest_model.fit(X_train, y_train) 


y_pred_rf = random_forest_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)


rf_report = classification_report(y_test, y_pred_rf)

print("rf_accuracy :",rf_accuracy) 
print("rf_report :",rf_report)



import joblib

# Eğitilmiş Random Forest modelini kaydetme
joblib.dump(random_forest_model, 'alzheimers_model.pkl')

print("Model başarıyla 'alzheimers_model.pkl' olarak kaydedildi.")


