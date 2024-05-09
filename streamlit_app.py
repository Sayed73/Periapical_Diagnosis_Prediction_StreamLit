import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
model = pickle.load(open('Periapical_Diagnosis_Prediction.sav', 'rb'))


def main():
   
    st.title('Periapical Diagnosis Prediction')

    st.header('Enter Features')
    pain_score = st.number_input('Pain Score', min_value=0, max_value=4)
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
        probabilities = model.predict_proba([features])[0]

        if prediction == 2:
            predicted_class = 'Normal periapical tissues'
        elif prediction == 0:
            predicted_class = 'Asymptomatic apical periodontitis'
        elif prediction == 3:
            predicted_class = 'Symptomatic apical periodontitis'
        else:
            predicted_class = 'Chronic apical abscess'

        # Display predicted class and probabilities
        st.markdown(f'<p style="font-size:20px; color:#2e6c80;"><strong>Predicted Value:</strong> {predicted_class} ({probabilities[prediction].round(2) * 100}%) </p>', unsafe_allow_html=True)
        # st.write('Class Probabilities:')
        # st.write('- Asymptomatic apical periodontitis:', probabilities[0].round(2) *100 ,"%")
        # st.write('- Normal periapical tissues:', probabilities[2].round(2)*100 ,"%")
        # st.write('- Symptomatic apical periodontitis:', probabilities[3].round(2)*100 ,"%")
        # st.write('- Chronic apical abscess:', probabilities[1].round(2)*100 ,"%")


if __name__ == "__main__":
    main()
