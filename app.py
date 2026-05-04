import streamlit as st
import pandas as pd
import joblib #to unpkl the 3 files

model = joblib.load('logRegressionHeart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.title("Heart Stroke Prediction System (biraj)")
st.markdown("Please enter the following details: ")

age = st.slider("Age" , 18 , 100 , 40)
sex = st.selectbox("SEX" , ['M' , 'F'])
chestpain = st.selectbox("CHEST PAIN TYPE" , ['ATA' , 'NAP' , 'TA' , 'ASY'])
restingbp = st.number_input("Resting Blood Pressure (mm Hg)" , 80 , 200 , 120)
cholesterol = st.number_input("Cholesterol (mg/dL)" , 100 , 600 , 200)
fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL" , [0 , 1])
restingecg = st.selectbox("Resting ECG" , ['Normal' , 'ST' , 'LVH'])
maxhr = st.slider("Max Heart Rate" , 50 , 220 , 150)
exerciseangina = st.selectbox("Exercise-Induced Angina" , ['Y' , 'N'])
oldpeak = st.slider("OldPeak (ST Depression)" , 0.0 , 6.0 , 1.0)
stslope = st.selectbox("ST Slope" , ['Up' , 'Flat' , 'Down'])

if st.button("Predict"):
    raw_input = {
        'Age' : age,
        'RestingBP' : restingbp,
        'Cholesterol' : cholesterol,
        'FastingBS' : fastingbs,
        'MaxHR' : maxhr,
        'Oldpeak' : oldpeak,
        'Sex_' + sex : 1,
        'ChestPainType_' + chestpain : 1,
        'RestingECG_' + restingecg : 1,
        'ExerciseAngina_' + exerciseangina : 1,
        'ST_Slope_' + stslope : 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    #logistic regression model ma scale not compulsory, knn, svm ma chai parxa, still doing it (better doing in any case ig)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease.")
    else:
        st.success("Low Risk of Heart Disease")