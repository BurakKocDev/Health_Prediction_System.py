from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np


heart=pd.read_csv('heart.csv')
print(heart.head()) 


# Özellik ve hedef ayrımı
X = heart[['age', 'sex', 'cp', 'trtbps', 'chol', 'thalachh', 'exng', 'oldpeak', 'caa', 'thall']]
y = heart['output']

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verilerin ölçeklenmesi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM modelini oluşturma ve eğitme
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Test verisi ile tahmin yapma
y_pred = svm_model.predict(X_test_scaled)

# Model performansı
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("accuracy :",accuracy)
print("report :", report)



# Modeli kaydet
import joblib

try:
    joblib.dump(svm_model, 'svm_heart_attack_model.pkl')
    print("Model kaydedildi.")
except Exception as e:
    print(f"Model kaydedilirken hata oluştu: {e}")



