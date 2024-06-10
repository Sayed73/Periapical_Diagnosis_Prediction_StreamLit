import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
model_periapical = pickle.load(open('Periapical_Diagnosis_Prediction.sav', 'rb'))
model_pulpal = pickle.load(open('pulpal_Diagnosis_Prediction.sav', 'rb'))


st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    st.title('Periapical & Pulpal Diagnosis Prediction')

    st.header('Enter Features')
    
    # Create two columns
    col1, col2, col3 = st.columns(3)
    
    # Column 1 for general features
    with col1:
        st.subheader('Clinical')
        pain_score = st.number_input('Pain Score', min_value=0, max_value=10)
        palpation = st.checkbox('Palpation')
        percussion = st.checkbox('Percussion')
        mobility = st.checkbox('Mobility')
        swelling_eo = st.checkbox('Swelling EO')
        swelling_io = st.checkbox('Swelling IO')
        sinus_tract = st.checkbox('Sinus Tract')
    
    # Column 2 for PAI and new features
    with col2:
        st.subheader('PAI Features')
        # st.markdown(
        #     """
        #     <div style="margin-top: 85px;"></div>
        #     """,
        #     unsafe_allow_html=True
        # )
        pai = st.radio('PAI Score', ['PAI 1', 'PAI 2', 'PAI 3', 'PAI 4', 'PAI 5'])

        pai_1 = 1 if pai == 'PAI 1' else 0
        pai_2 = 1 if pai == 'PAI 2' else 0
        pai_3 = 1 if pai == 'PAI 3' else 0
        pai_4 = 1 if pai == 'PAI 4' else 0
        pai_5 = 1 if pai == 'PAI 5' else 0

    # st.header('Quality of Filling')
    with col3:
        st.subheader('Quality of Filling')
        acceptability = st.selectbox('Acceptability', ['Yes', 'No', 'NA'])

    # Convert the input string to 
    acceptability = 0 if acceptability == 'No' else acceptability
    acceptability = 1 if acceptability == 'Yes' else acceptability
    acceptability = 99 if acceptability == 'NA' else acceptability


    # Predict when button is clicked
    if st.button('Predict'):
        features = [
            pain_score, palpation, percussion, mobility, pai_1, pai_2, pai_3, pai_4, pai_5, 
            swelling_eo, swelling_io, sinus_tract, acceptability
        ]

        # Predict periapical diagnosis
        prediction_periapical = model_periapical.predict([features])[0]
        probabilities_periapical = model_periapical.predict_proba([features])[0]

        if prediction_periapical == 0:
            predicted_class_periapical = 'Acute apical abscess'
        elif prediction_periapical == 1:
            predicted_class_periapical = 'Asymptomatic apical periodontitis'
        elif prediction_periapical == 2:
            predicted_class_periapical = 'Chronic apical abscess'
        elif prediction_periapical == 3:
            predicted_class_periapical = 'Normal periapical tissues'
        else:
            predicted_class_periapical = 'Symptomatic apical periodontitis'

        # Predict pulpal diagnosis
        prediction_pulpal = model_pulpal.predict([features])[0]
        probabilities_pulpal = model_pulpal.predict_proba([features])[0]

        if prediction_pulpal == 0:
            predicted_class_pulpal = 'Asymptomatic irreversible pulpitis'
        elif prediction_pulpal == 1:
            predicted_class_pulpal = 'Necrotic pulp'
        elif prediction_pulpal == 2:
            predicted_class_pulpal = 'Previously initiated therapy'
        elif prediction_pulpal == 3:
            predicted_class_pulpal = 'Previously treated tooth'
        else:
            predicted_class_pulpal = 'Symptomatic irreversible pulpitis'

        # Display predicted class and probabilities
        st.markdown(f'<p style="font-size:20px; color:#2e6c80;"><strong>Periapical Predicted Value:</strong> {predicted_class_periapical} ({probabilities_periapical[prediction_periapical].round(2) * 100}%) </p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:20px; color:#2e6c80;"><strong>Pulpal Predicted Value:</strong> {predicted_class_pulpal} ({probabilities_pulpal[prediction_pulpal].round(2) * 100}%) </p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
