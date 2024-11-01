import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import numpy as np
import joblib

# Model dosyalarını yükle
models = {
    "Alzheimer": 'alzheimers_model.pkl',
    "Parkinson": 'xgb_parkinsons_model.pkl',
    "Heart Disease": 'svm_heart_attack_model.pkl',
    "Cardiovascular Disease": 'random_forest_cardiovascular_model.pkl',
    "Asthma": 'xgb_asthma_model.pkl'
}

loaded_models = {}
for disease, model_file in models.items():
    try:
        if disease == "Heart Disease":
            loaded_models[disease] = joblib.load(model_file)
        else:
            with open(model_file, 'rb') as file:
                loaded_models[disease] = pickle.load(file)
    except Exception as e:
        print(f"{disease} model could not be loaded: {e}")

# Özellikler ve seçenekler

disease_features = {
    "Alzheimer": {
        "Age": None,
        "Gender": {0: "Male", 1: "Female"},  
        "Ethnicity": {0: "White", 1: "African American", 2: "Asian", 3: "Other"},  
        "Education Level": {1: "High School", 2: "Bachelor's", 3: "Master's", 4: "PhD", 5: "Other"},
        "BMI": None,
        "Smoking": {1: "Yes", 0: "No"},
        "Alcohol Consumption": {1: "Yes", 0: "No"},
        "Physical Activity": {1: "Regular", 0: "Irregular"},
        "Diet Quality": {1: "Good", 0: "Poor"},
        "Sleep Quality": {1: "Good", 0: "Poor"},
        "Family History of Alzheimer's": {1: "Yes", 0: "No"},
        "Cardiovascular Disease": {1: "Yes", 0: "No"},
        "Diabetes": {1: "Yes", 0: "No"},
        "Depression": {1: "Yes", 0: "No"},
        "Head Injury": {1: "Yes", 0: "No"},
        "Hypertension": {1: "Yes", 0: "No"},
        "Systolic BP": None,
        "Diastolic BP": None,
        "Cholesterol Total": None,
        "Cholesterol LDL": None,
        "Cholesterol HDL": None,
        "Cholesterol Triglycerides": None,
        "MMSE": None,
        "Functional Assessment": None,
        "Memory Complaints": {1: "Yes", 0: "No"},
        "Behavioral Problems": {1: "Yes", 0: "No"},
        "ADL": {1: "Independent", 0: "Assistance Needed"},
        "Confusion": {1: "Yes", 0: "No"},
        "Disorientation": {1: "Yes", 0: "No"},
        "Personality Changes": {1: "Yes", 0: "No"},
        "Difficulty Completing Tasks": {1: "Yes", 0: "No"},
        "Forgetfulness": {1: "Yes", 0: "No"}
    },
    "Parkinson": {
        "Age": None,
        "Gender": {0: "Male", 1: "Female"},
        "Ethnicity": {0: "White", 1: "African American", 2: "Asian", 3: "Other"},
        "Education Level": {1: "High School", 2: "Bachelor's", 3: "Master's", 4: "PhD"},
        "BMI": None,
        "Smoking": {1: "Yes", 0: "No"},
        "Alcohol Consumption": {1: "Yes", 0: "No"},
        "Physical Activity": {1: "Regular", 0: "Irregular"},
        "Diet Quality": {1: "Good", 0: "Poor"},
        "Sleep Quality": {1: "Good", 0: "Poor"},
        "Family History of Parkinson's": {1: "Yes", 0: "No"},
        "Traumatic Brain Injury": {1: "Yes", 0: "No"},
        "Hypertension": {1: "Yes", 0: "No"},
        "Diabetes": {1: "Yes", 0: "No"},
        "Depression": {1: "Yes", 0: "No"},
        "Stroke": {1: "Yes", 0: "No"},
        "Systolic BP": None,
        "Diastolic BP": None,
        "Cholesterol Total": None,
        "Cholesterol LDL": None,
        "Cholesterol HDL": None,
        "Cholesterol Triglycerides": None,
        "UPDRS": None,
        "MoCA": None,
        "Functional Assessment": None,
        "Tremor": {1: "Yes", 0: "No"},
        "Rigidity": {1: "Yes", 0: "No"},
        "Bradykinesia": {1: "Yes", 0: "No"},
        "Postural Instability": {1: "Yes", 0: "No"},
        "Speech Problems": {1: "Yes", 0: "No"},
        "Sleep Disorders": {1: "Yes", 0: "No"},
        "Constipation": {1: "Yes", 0: "No"}
    },
    "Heart Disease": {
        "Age": None,
        "Gender": {"Male": 1, "Female": 2},
        "Chest Pain Type": {"Type 1": 1, "Type 2": 2, "Type 3": 3},
        "Blood Pressure": None,
        "Cholesterol": None,
        "Maximum Heart Rate": None,
        "Pain on Exercise": {"Yes": 1, "No": 0},
        "ST Depression": None,
        "Number of Vessels": None,
        "Thal": {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    },
   "Cardiovascular Disease": {
        "Age": None,
        "Gender": {1: "Male", 0: "Female"},
        "Chest Pain Type": {0: "Type 1", 1: "Type 2", 2: "Type 3", 3: "Type 4"},
        "Resting Blood Pressure": None,
        "Serum Cholesterol": None,
        "Fasting Blood Sugar": {1: "> 120 mg/dl", 0: "<= 120 mg/dl"},
        "Resting ECG Results": {0: "Normal", 1: "Abnormal", 2: "Probable Hypertrophy"},
        "Maximum Heart Rate": None,
        "Exercise-Induced Angina": {1: "Yes", 0: "No"},
        "Old Peak": None,
        "Slope of the ST Segment": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
        "Number of Major Vessels": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"},
        "Target": {1: "Disease", 0: "No Disease"}
    },
    "Asthma": {
        "Age": None,
        "Gender": {1: "Male", 0: "Female"},
        "Ethnicity": {1: "White", 2: "African American", 3: "Asian", 4: "Other"},
        "Education Level": {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"},
        "BMI": None,
        "Smoking": {1: "Yes", 0: "No"},
        "Physical Activity": None,
        "Diet Quality": None,
        "Sleep Quality": None,
        "Pollution Exposure": None,
        "Pollen Exposure": None,
        "Dust Exposure": None,
        "Pet Allergy": {1: "Yes", 0: "No"},
        "Family History of Asthma": {1: "Yes", 0: "No"},
        "History of Allergies": {1: "Yes", 0: "No"},
        "Eczema": {1: "Yes", 0: "No"},
        "Hay Fever": {1: "Yes", 0: "No"},
        "Gastroesophageal Reflux": {1: "Yes", 0: "No"},
        "Lung Function (FEV1)": None,
        "Lung Function (FVC)": None,
        "Wheezing": {1: "Yes", 0: "No"},
        "Shortness of Breath": {1: "Yes", 0: "No"},
        "Chest Tightness": {1: "Yes", 0: "No"},
        "Coughing": {1: "Yes", 0: "No"},
        "Nighttime Symptoms": {1: "Yes", 0: "No"},
        "Exercise-Induced Symptoms": {1: "Yes", 0: "No"},
        "Diagnosis": {1: "Asthma", 0: "No Asthma"},
        "Doctor in Charge": None
    }
}


class DiseasePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disease Prediction Application")
        self.root.geometry("600x600")

        self.selected_disease = tk.StringVar()
        self.selected_disease.trace("w", self.update_features)

        ttk.Label(self.root, text="Select Disease:").pack(pady=10)
        self.disease_dropdown = ttk.Combobox(self.root, textvariable=self.selected_disease, values=list(disease_features.keys()))
        self.disease_dropdown.pack()

        self.feature_frame = ttk.Frame(self.root)
        self.feature_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.feature_frame)
        self.scrollbar = tk.Scrollbar(self.feature_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.entries = {}

        self.predict_button = ttk.Button(self.root, text="Predict", command=self.predict_disease)
        self.predict_button.pack(pady=20)

    def update_features(self, *args):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        disease = self.selected_disease.get()
        if disease:
            self.entries.clear()
            for feature, options in disease_features[disease].items():
                label = ttk.Label(self.scrollable_frame, text=feature)
                label.pack(pady=5)

                if isinstance(options, dict):
                    var = tk.StringVar(value="Select")
                    self.entries[feature] = var
                    option_menu = ttk.OptionMenu(self.scrollable_frame, var, "Select", *options.keys())
                    option_menu.pack(fill=tk.X, padx=5, pady=2)
                else:
                    entry = ttk.Entry(self.scrollable_frame)
                    entry.pack(fill=tk.X, padx=5, pady=2)
                    self.entries[feature] = entry

    def predict_disease(self):
        try:
            disease = self.selected_disease.get()
            if disease not in loaded_models:
                raise ValueError("Model for the selected disease is not loaded.")

            input_features = []
            for feature, entry in self.entries.items():
                if isinstance(entry, tk.StringVar):  # Dropdown (Seçmeli)
                    value = entry.get()
                    if value == "Select":
                        raise ValueError(f"{feature} seçimi yapılmamış.")
                    input_features.append(disease_features[disease][feature][value])
                else:  # Text girişi
                    value = entry.get()
                    if not value.strip():
                        raise ValueError(f"{feature} değeri boş bırakılamaz.")
                    input_features.append(float(value))

            features_array = np.array(input_features).reshape(1, -1)

            model = loaded_models[disease]
            prediction = model.predict(features_array)
            
            # Tahmin sonucunu açıklayıcı mesajla dön
            if prediction[0] == 1:
                result_message = f"{disease} için risk taşıyorsunuz. Lütfen doktorunuza danışın."
            else:
                result_message = f"{disease} için risk taşımıyorsunuz. Sağlıklı kalmaya devam edin!"

            messagebox.showinfo("Prediction Result", result_message)
        
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Prediction Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictionApp(root)
    root.mainloop()