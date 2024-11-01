import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix


parkinsons_data = pd.read_csv("parkinsons_disease_data.csv")


parkinsons_data_cleaned = parkinsons_data.drop(['PatientID', 'DoctorInCharge'], axis=1)


X = parkinsons_data_cleaned.drop('Diagnosis', axis=1)
y = parkinsons_data_cleaned['Diagnosis']


categorical_columns = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'FamilyHistoryParkinsons',
                       'TraumaticBrainInjury', 'Hypertension', 'Diabetes', 'Depression', 'Stroke', 'Tremor',
                       'Rigidity', 'Bradykinesia', 'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100)
xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)

print(parkinsons_data_cleaned.columns)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)


import joblib

# Modeli kaydet
joblib.dump(xgb_model, 'xgb_parkinsons_model.pkl')
print("model kaydeildi")
# Label encoders'ı kaydet
joblib.dump(label_encoders, 'label_encoders_parkinsons.pkl')


