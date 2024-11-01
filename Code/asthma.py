import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix


asthma = pd.read_csv("asthma_disease_data.csv")
print(asthma)
print(asthma.columns)
# Kategorik verileri sayısal verilere dönüştür
label_encoder = LabelEncoder()

# Cinsiyet, Sigara Kullanımı gibi kategorik değişkenleri sayısal hale getirelim
categorical_columns = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'PhysicalActivity', 'DietQuality',
                       'SleepQuality', 'PollutionExposure', 'PollenExposure', 'DustExposure', 'PetAllergy', 
                       'FamilyHistoryAsthma', 'HistoryOfAllergies', 'Eczema', 'HayFever', 'GastroesophagealReflux',
                       'Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing', 'NighttimeSymptoms', 
                       'ExerciseInduced']

for col in categorical_columns:
    asthma[col] = label_encoder.fit_transform(asthma[col])

# Bağımsız ve bağımlı değişkenleri ayır
X = asthma.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis=1)  # Diagnosis hedef değişken, PatientID ve Doctor dışarıda bırakılıyor
y = asthma['Diagnosis']

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri ölçeklendir (opsiyonel ama önerilir)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost modelini tanımla ve eğit
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Test verileri üzerinde tahmin yap
y_pred = xgb_model.predict(X_test)

# Modelin doğruluğunu ve karışıklık matrisini göster
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Doğruluk: {accuracy}")
print("Karışıklık Matrisi:")
print(conf_matrix)



import joblib

# Modeli kaydet
joblib.dump(xgb_model, 'xgb_asthma_model.pkl')

# Label encoders'ı kaydet
joblib.dump(label_encoder, 'label_encoder_asthma.pkl')

# Scaler'ı kaydet
joblib.dump(scaler, 'scaler_asthma.pkl')
print("kaydedildi")