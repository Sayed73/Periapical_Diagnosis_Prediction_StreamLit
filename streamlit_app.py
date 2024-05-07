import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
model = pickle.load(open('Periapical_Diagnosis_Prediction.sav', 'rb'))

# Define the Streamlit app
def main():
    st.title('Periapical Diagnosis Prediction')

    # Add input widgets for user to enter features
    st.header('Enter Features')
    pain_score = st.number_input('Pain Score', min_value=0, max_value=10)
    palpation = st.checkbox('Palpation')
    percussion = st.checkbox('Percussion')
    mobility = st.checkbox('Mobility')
    pai_1 = st.checkbox('PAI 1')
    pai_2 = st.checkbox('PAI 2')
    pai_3 = st.checkbox('PAI 3')
    pai_4 = st.checkbox('PAI 4')
    pai_5 = st.checkbox('PAI 5')
    sinus_tract = st.checkbox('Sinus Tract')

    # Predict when button is clicked
    if st.button('Predict'):
        features = [pain_score, palpation, percussion, mobility, pai_1, pai_2, pai_3, pai_4, pai_5, sinus_tract]
        prediction = model.predict([features])[0]
        if prediction == 2:
            prediction = 'Normal periapical tissues'
        elif prediction == 0:
            prediction = 'Asymptomatic apical periodontitis'
        elif prediction == 3:
            prediction = 'Symptomatic apical periodontitis'
        else:
            prediction = 'Chronic apical abscess'
            
        st.write(f'The predicted value is: {prediction}')

if __name__ == "__main__":
    main()
