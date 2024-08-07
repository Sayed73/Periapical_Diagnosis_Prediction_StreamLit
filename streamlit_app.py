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


# def predict_diagnosis(features, swelling_eo, sinus_tract):
#     # Force prediction based on features
#     if swelling_eo:
#         return 'Acute apical abscess'
#     elif sinus_tract:
#         return 'Chronic apical abscess'
#     else:
#         # Perform model prediction if forced conditions are not met
#         prediction_periapical = model_periapical.predict([features])[0]
#         if prediction_periapical == 0:
#             return 'Acute apical abscess'
#         elif prediction_periapical == 1:
#             return 'Asymptomatic apical periodontitis'
#         elif prediction_periapical == 2:
#             return 'Chronic apical abscess'
#         elif prediction_periapical == 3:
#             return 'Normal periapical tissues'
#         else:
#             return 'Symptomatic apical periodontitis'



def main():
    st.title('Periapical & Pulpal Diagnosis Prediction')

    st.header('Enter Features')

    # Create four columns
    col1, col2, col3, col4 = st.columns(4)

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

    # Column 2 for additional general features
    with col2:
        Painkiller_usage = st.selectbox('Pain killer usage', ['Yes', 'No'])
        Pain_duration = st.selectbox('how long experiencing the pain', ['less than 2 days', 'more than 2 days', 'NA'])
        affected_tooth = st.selectbox('can you point out the tooth', ['Yes', 'No', 'NA'])
        tooth_open_history = st.selectbox('was the tooth open earlier', ['Yes', 'No', 'NA'])
        Pulp_Vitality = st.selectbox('Pulp Vitality (Cold)', ['+ve', '-ve', 'NA'])

    # Convert the input strings to numeric values
    Painkiller_usage = 0 if Painkiller_usage == 'No' else 1
    Pain_duration = 0 if Pain_duration == 'more than 2 days' else 1 if Pain_duration == 'less than 2 days' else 99
    affected_tooth = 0 if affected_tooth == 'No' else 1 if affected_tooth == 'Yes' else 99
    tooth_open_history = 0 if tooth_open_history == 'No' else 1 if tooth_open_history == 'Yes' else 99
    Pulp_Vitality = 0 if Pulp_Vitality == '-ve' else 1 if Pulp_Vitality == '+ve' else 99

    # Column 3 for PAI features
    with col3:
        st.subheader('PAI Features')
        show_pai_options = st.selectbox('Is there Periapical RL changes ?', ['No', 'Yes'])

        # Initialize PAI features
        pai_features = [0] * 5

        if show_pai_options == 'Yes':
            pai = st.radio('PAI Score', ['PAI 3', 'PAI 4', 'PAI 5'])
            pai_features = [1 if pai == f else 0 for f in ['PAI 1', 'PAI 2','PAI 3', 'PAI 4', 'PAI 5']]

        elif show_pai_options == 'No':
            pai = st.radio('PAI Score', ['PAI 1', 'PAI 2'])
            pai_features = [1 if pai == f else 0 for f in ['PAI 1', 'PAI 2','PAI 3', 'PAI 4', 'PAI 5']]

    # Column 4 for Quality of Filling
    with col4:
        st.subheader('Root Canal Filling')
        is_rcf = st.selectbox('Is there a RCF ?', ['No','Yes'])
    
        if is_rcf == 'No':
            acceptability = 99
        else:
            # If Yes, show the second question
            quality_of_acceptability = st.selectbox('Acceptability', ['Yes', 'No'])
            
            if quality_of_acceptability == 'Yes':
                acceptability = 1
            else:
                acceptability = 0


    # Predict when button is clicked
    if st.button('Predict'):
        features = [
            pain_score, Painkiller_usage, Pain_duration, affected_tooth, tooth_open_history, palpation, percussion, mobility, *pai_features, 
            acceptability, swelling_eo, swelling_io, sinus_tract, Pulp_Vitality
        ]


        # Predict periapical diagnosis
        prediction_periapical = model_periapical.predict([features])[0]
        # probabilities_periapical = model_periapical.predict_proba([features])[0]

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

        # Perform forced prediction based on features
        # predicted_class_periapical = predict_diagnosis(features, swelling_eo, sinus_tract)

        # Predict pulpal diagnosis
        prediction_pulpal = model_pulpal.predict([features])[0]
        # probabilities_pulpal = model_pulpal.predict_proba([features])[0]

        if prediction_pulpal == 0:
            predicted_class_pulpal = 'Asymptomatic irreversible pulpitis'
        elif prediction_pulpal == 1:
            predicted_class_pulpal = 'Necrotic pulp'
        elif prediction_pulpal == 2:
            predicted_class_pulpal = 'Previously initiated therapy'
        elif prediction_pulpal == 3:
            predicted_class_pulpal = 'Previously treated tooth'
        elif prediction_pulpal == 4:
            predicted_class_pulpal = 'Symptomatic irreversible pulpitis'
        else:
            predicted_class_pulpal = 'reversible pulpitis'

        # Display predicted class and probabilities
        st.markdown(f'<p style="font-size:22px; color:#2e6c80;"><strong>Periapical Predicted Value:</strong> {predicted_class_periapical}  </p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:22px; color:#2e6c80;"><strong>Pulpal Predicted Value:</strong> {predicted_class_pulpal}  </p>', unsafe_allow_html=True)
        # st.markdown(f'<p style="font-size:20px; color:#2e6c80;"><strong>Periapical Predicted Value:</strong> {predicted_class_periapical} ({probabilities_periapical[prediction_periapical].round(2) * 100}%) </p>', unsafe_allow_html=True)
        # st.markdown(f'<p style="font-size:20px; color:#2e6c80;"><strong>Pulpal Predicted Value:</strong> {predicted_class_pulpal} ({probabilities_pulpal[prediction_pulpal].round(2) * 100}%) </p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
