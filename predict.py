import joblib
import numpy as np

# Load saved model
model = joblib.load("xgboost_model.pkl")

# Sample patient data
sample = np.array([[
    35,     # Age
    2,      # Number of sexual partners
    18,     # First sexual intercourse
    2,      # Num of pregnancies
    0,      # Smokes
    0,      # Smokes (years)
    0,      # Smokes (packs/year)
    1,      # Hormonal Contraceptives
    5,      # Hormonal Contraceptives (years)
    0,      # IUD
    0,      # IUD (years)
    0,      # STDs
    0,      # STDs(number)
    0,      # STDs:condylomatosis
    0,      # STDs:cervical condylomatosis
    0,      # STDs:vaginal condylomatosis
    0,      # STDs:vulvo-perinealcondylomatosis
    0,      # STDs:syphilis
    0,      # STDs:pelvic inflammatory disease
    0,      # STDs:genital herpes
    0,      # STDs:molluscum contagiosum
    0,      # STDs:AIDS
    0,      # STDs:HIV
    0,      # STDs:Hepatitis B
    0,      # STDs:HPV
    0,      # STDs: Number of diagnosis
    0,      # Dx:Cancer
    0,      # Dx:CIN
    0,      # Dx:HPV
    0,      # Dx
    0,      # Hinselmann
    0,      # Schiller
    0       # Citology
]])

prediction = model.predict(sample)

print("Prediction:", prediction)

if prediction[0] == 1:
    print("High Risk")
else:
    print("Low Risk")