import streamlit as st
import joblib
import numpy as np
import time

model = joblib.load("xgboost_model.pkl")

st.set_page_config(
    page_title="Cervical Cancer Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Cervical Cancer Prediction System")
st.write("XGBoost Based Cervical Cancer Risk Prediction")

# -------------------
# Demographics
# -------------------

with st.expander("📋 Demographics", expanded=True):

    age = st.number_input("Age", 0.0, 100.0, 30.0)

    partners = st.number_input(
        "Number of sexual partners",
        0.0,
        20.0,
        1.0
    )

    intercourse = st.number_input(
        "First sexual intercourse",
        0.0,
        50.0,
        18.0
    )

    pregnancies = st.number_input(
        "Num of pregnancies",
        0.0,
        20.0,
        0.0
    )

# -------------------
# Smoking
# -------------------

with st.expander("🚬 Smoking Information"):

    smokes = st.number_input("Smokes", 0.0, 1.0, 0.0)

    smokes_years = st.number_input(
        "Smokes (years)",
        0.0,
        60.0,
        0.0
    )

    smokes_packs = st.number_input(
        "Smokes (packs/year)",
        0.0,
        200.0,
        0.0
    )

# -------------------
# Contraceptives
# -------------------

with st.expander("💊 Contraceptive Information"):

    hormonal = st.number_input(
        "Hormonal Contraceptives",
        0.0,
        1.0,
        0.0
    )

    hormonal_years = st.number_input(
        "Hormonal Contraceptives (years)",
        0.0,
        50.0,
        0.0
    )

    iud = st.number_input(
        "IUD",
        0.0,
        1.0,
        0.0
    )

    iud_years = st.number_input(
        "IUD (years)",
        0.0,
        50.0,
        0.0
    )

# -------------------
# STD Information
# -------------------

with st.expander("🦠 STD Information"):

    stds = st.number_input("STDs", 0.0, 1.0, 0.0)

    std_number = st.number_input(
        "STDs(number)",
        0.0,
        20.0,
        0.0
    )

    condylomatosis = st.number_input(
        "STDs:condylomatosis",
        0.0,
        1.0,
        0.0
    )

    cervical_condylomatosis = st.number_input(
        "STDs:cervical condylomatosis",
        0.0,
        1.0,
        0.0
    )

    vaginal_condylomatosis = st.number_input(
        "STDs:vaginal condylomatosis",
        0.0,
        1.0,
        0.0
    )

    vulvo = st.number_input(
        "STDs:vulvo-perinealcondylomatosis",
        0.0,
        1.0,
        0.0
    )

    syphilis = st.number_input(
        "STDs:syphilis",
        0.0,
        1.0,
        0.0
    )

    pid = st.number_input(
        "STDs:pelvic inflammatory disease",
        0.0,
        1.0,
        0.0
    )

    herpes = st.number_input(
        "STDs:genital herpes",
        0.0,
        1.0,
        0.0
    )

    molluscum = st.number_input(
        "STDs:molluscum contagiosum",
        0.0,
        1.0,
        0.0
    )

    aids = st.number_input(
        "STDs:AIDS",
        0.0,
        1.0,
        0.0
    )

    hiv = st.number_input(
        "STDs:HIV",
        0.0,
        1.0,
        0.0
    )

    hepatitis = st.number_input(
        "STDs:Hepatitis B",
        0.0,
        1.0,
        0.0
    )

    hpv = st.number_input(
        "STDs:HPV",
        0.0,
        1.0,
        0.0
    )

    diagnosis_count = st.number_input(
        "STDs: Number of diagnosis",
        0.0,
        20.0,
        0.0
    )

# -------------------
# Diagnosis
# -------------------

with st.expander("🏥 Diagnosis Information"):

    dx_cancer = st.number_input("Dx:Cancer", 0.0, 1.0, 0.0)
    dx_cin = st.number_input("Dx:CIN", 0.0, 1.0, 0.0)
    dx_hpv = st.number_input("Dx:HPV", 0.0, 1.0, 0.0)
    dx = st.number_input("Dx", 0.0, 1.0, 0.0)

# -------------------
# Screening Tests
# -------------------

with st.expander("🔬 Screening Test Results"):

    hinselmann = st.number_input(
        "Hinselmann",
        0.0,
        1.0,
        0.0
    )

    schiller = st.number_input(
        "Schiller",
        0.0,
        1.0,
        0.0
    )

    citology = st.number_input(
        "Citology",
        0.0,
        1.0,
        0.0
    )

# -------------------
# Prediction
# -------------------

if st.button("🚑 Predict Risk"):

    try:
        st.image("assets/ambulance.gif", width=250)
    except:
        pass

    with st.spinner("Analyzing Patient Data..."):

        time.sleep(2)

        data = np.array([[
            age,
            partners,
            intercourse,
            pregnancies,
            smokes,
            smokes_years,
            smokes_packs,
            hormonal,
            hormonal_years,
            iud,
            iud_years,
            stds,
            std_number,
            condylomatosis,
            cervical_condylomatosis,
            vaginal_condylomatosis,
            vulvo,
            syphilis,
            pid,
            herpes,
            molluscum,
            aids,
            hiv,
            hepatitis,
            hpv,
            diagnosis_count,
            dx_cancer,
            dx_cin,
            dx_hpv,
            dx,
            hinselmann,
            schiller,
            citology
        ]])

        prediction = model.predict(data)
        probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ High Risk Detected")
    else:
        st.success("✅ Low Risk")

    st.metric(
        "Cancer Risk Probability",
        f"{probability*100:.2f}%"
    )