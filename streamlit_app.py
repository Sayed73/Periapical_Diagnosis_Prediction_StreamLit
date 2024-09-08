import streamlit as st
import pandas as pd
import numpy as np
import pickle
import boto3
from io import BytesIO
from botocore.exceptions import ClientError
from datetime import datetime
import base64

# Load the saved model
model_periapical = pickle.load(open('Periapical_Diagnosis_Prediction.sav', 'rb'))
model_pulpal = pickle.load(open('pulpal_Diagnosis_Prediction.sav', 'rb'))

# Access the secrets
aws_access_key_id = st.secrets["aws_credentials"]["access_key_id"]
aws_secret_access_key = st.secrets["aws_credentials"]["secret_access_key"]
region_name = st.secrets["aws_credentials"]["region_name"]
    
# Initialize an S3 client
s3 = boto3.client('s3', 
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name)

bucket_name = st.secrets["s3"]["bucket_name"]
file_name = st.secrets["s3"]["file_name"]

# Function to check if file exists in S3
def check_file_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False

# Function to read Parquet from S3
def read_parquet_from_s3(bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(BytesIO(response['Body'].read()))
    except ClientError:
        return pd.DataFrame()

# Function to write Parquet to S3
def write_parquet_to_s3(df, bucket, key):
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=parquet_buffer.getvalue())

# Test function to view Parquet file contents
def test_view_parquet_file():
    if check_file_exists(bucket_name, file_name):
        df = read_parquet_from_s3(bucket_name, file_name)
        st.write("Contents of the Parquet file:")
        st.dataframe(df)
        st.write(f"Total number of records: {len(df)}")
    else:
        st.write("Parquet file does not exist in the S3 bucket.")

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
    st.title('DIAGNOSIS PULPAL AND PERIAPICAL DISEASES')

    # Create four columns
    col1, col2, col3= st.columns([2, 3, 3])

    # Column 1 for general features
    with col1:
        st.markdown('<p style="font-size:26px; font-weight:bold;">A. Pain</p>', unsafe_allow_html=True)
        pain_score = st.number_input('Pain Score', min_value=0, max_value=10)
        Painkiller_usage = st.selectbox('Have you taken pain killers ?', ['Yes', 'No'])
        Pain_duration = st.selectbox('When did the pain start ?', ['less than 2 days', 'more than 2 days', 'NA'])
        Pulp_Vitality = st.selectbox('Pulp Vitality test(Cold)', ['+ve', '-ve', 'NA'])
        affected_tooth = st.selectbox('Can you point out the offending tooth ?', ['Yes', 'No', 'NA'])
        
    # Convert the input strings to numeric values
    Painkiller_usage = 0 if Painkiller_usage == 'No' else 1
    Pain_duration = 0 if Pain_duration == 'more than 2 days' else 1 if Pain_duration == 'less than 2 days' else 99
    Pulp_Vitality = 0 if Pulp_Vitality == '-ve' else 1 if Pulp_Vitality == '+ve' else 99
    affected_tooth = 0 if affected_tooth == 'No' else 1 if affected_tooth == 'Yes' else 99

    # Column 2 for additional general features
    with col2:
        # Center the "B. Clinical Examination" header
        st.markdown('<p style="text-align:center; font-size:26px; font-weight:bold;">B. Clinical Examination</p>', unsafe_allow_html=True)

        # Separating the features into two sub-columns
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            swelling_eo = 1 if st.checkbox('Extra-Oral Swelling') else 0
            swelling_io = 1 if st.checkbox('Intra-Oral Swelling') else 0
            sinus_tract = 1 if st.checkbox('Sinus Tract') else 0
            tooth_open_history = st.selectbox('Was the tooth open Earlier ?', ['Yes', 'No', 'NA'])
        with sub_col2:
            palpation = 1 if st.checkbox('Palpation') else 0
            percussion = 1 if st.checkbox('Percussion') else 0
            mobility = 1 if st.checkbox('Mobility') else 0
        
    # Convert the input strings to numeric values
    tooth_open_history = 0 if tooth_open_history == 'No' else 1 if tooth_open_history == 'Yes' else 99

    # Column 3 for PAI features
    with col3:
        st.markdown('<p style="text-align:center; font-size:26px; font-weight:bold;">C. Radiographic Examination</p>', unsafe_allow_html=True)
        # Separating the features into two sub-columns
        sub_col3, sub_col4 = st.columns(2)
        with sub_col3:
            show_pai_options = st.selectbox('Is there a Periapical Radiolucency ?', ['No', 'Yes'])

            # Initialize PAI features
            pai_features = [0] * 5

            if show_pai_options == 'Yes':
                pai = st.radio('PAI Score', ['PAI 3', 'PAI 4', 'PAI 5'])
                pai_features = [1 if pai == f else 0 for f in ['PAI 1', 'PAI 2','PAI 3', 'PAI 4', 'PAI 5']]

            elif show_pai_options == 'No':
                pai = st.radio('PAI Score', ['PAI 1', 'PAI 2'])
                pai_features = [1 if pai == f else 0 for f in ['PAI 1', 'PAI 2','PAI 3', 'PAI 4', 'PAI 5']]
        with sub_col4:
            is_rcf = st.selectbox('Is there a previous Root Canal Filling ?', ['No','Yes'])
    
            if is_rcf == 'No':
                acceptability = 99
            else:
                # If Yes, show the second question
                quality_of_acceptability = st.selectbox('Is it Acceptable Root Canal Filling ?', ['Yes', 'No'])
                
                if quality_of_acceptability == 'Yes':
                    acceptability = 1
                else:
                    acceptability = 0

    # New section for user diagnosis
    st.markdown('<p style="font-size:26px; font-weight:bold;">Your Diagnosis</p>', unsafe_allow_html=True)
    
    # Options for pulpal diagnosis
    pulpal_options = [
        'Asymptomatic irreversible pulpitis',
        'Necrotic pulp',
        'Previously initiated therapy',
        'Previously treated tooth',
        'Symptomatic irreversible pulpitis',
        'Reversible pulpitis'
    ]
    
    # Options for periapical diagnosis
    periapical_options = [
        'Acute apical abscess',
        'Asymptomatic apical periodontitis',
        'Chronic apical abscess',
        'Normal periapical tissues',
        'Symptomatic apical periodontitis'
    ]

    
    user_pulpal_diagnose = st.selectbox("What is the pulpal diagnosis?", pulpal_options)
    user_periapical_diagnose = st.selectbox("What is the periapical diagnosis?", periapical_options)

    # Predict when button is clicked
    if st.button('Diagnose'):
        features = [
            pain_score, Painkiller_usage, Pain_duration, affected_tooth, tooth_open_history, palpation, percussion, mobility, *pai_features, 
            acceptability, swelling_eo, swelling_io, sinus_tract, Pulp_Vitality
        ]


        # Predict periapical diagnosis
        prediction_periapical = model_periapical.predict([features])[0]

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
            predicted_class_pulpal = 'Reversible pulpitis'

        # Display predicted class and probabilities
        st.markdown(f'<p style="font-size:22px; color:#2e6c80;"><strong>Pulpal diagnosis:</strong> {predicted_class_pulpal}  </p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:22px; color:#2e6c80;"><strong>Periapical diagnosis:</strong> {predicted_class_periapical}  </p>', unsafe_allow_html=True)
        # Compare user input with predictions
        if user_pulpal_diagnose == predicted_class_pulpal and user_periapical_diagnose == predicted_class_periapical:
            st.success("Well done! Your diagnoses match the prediction.🥳🎉")
            st.balloons()
        else:
            st.warning("Your diagnosis does not match the predicted diagnosis.😔")

        # Save the record to S3
        columns = [
            'pain_score', 'Painkiller_usage', 'Pain_duration', 'affected_tooth', 'tooth_open_history', 
            'palpation', 'percussion', 'mobility', 'PAI_1', 'PAI_2', 'PAI_3', 'PAI_4', 'PAI_5', 
            'acceptability', 'swelling_eo', 'swelling_io', 'sinus_tract', 'Pulp_Vitality'
        ]
        # Convert features to a list of 'NA' if the value is 99
        features_na = ['NA' if x == 99 else x for x in features]
        
        new_record = pd.DataFrame([features_na], columns=columns)
        # Convert columns to appropriate types
        numeric_columns = ['Pain_duration', 'affected_tooth', 'tooth_open_history', 'acceptability', 'Pulp_Vitality']
        
        for col in numeric_columns:
            new_record[col] = new_record[col].astype(str)
        
        new_record['pulpal_diagnosis'] = predicted_class_pulpal
        new_record['periapical_diagnosis'] = predicted_class_periapical
        new_record['user_pulpal_diagnose'] = user_pulpal_diagnose
        new_record['user_periapical_diagnose'] = user_periapical_diagnose
        new_record['record_date'] = datetime.now().strftime("%Y-%m-%d")

        try:
            if check_file_exists(bucket_name, file_name):
                existing_df = read_parquet_from_s3(bucket_name, file_name)
                updated_df = pd.concat([existing_df, new_record], ignore_index=True)
                # Remove duplicates cases, keeping the first occurrence
                updated_df = updated_df.drop_duplicates(subset=columns, keep='first')
            
            else:
                updated_df = new_record

            write_parquet_to_s3(updated_df, bucket_name, file_name)
            st.success("Record saved successfully!")
        except Exception as e:
            st.error(f"An error occurred while saving the record: {str(e)}")
        

if __name__ == "__main__":
    main()
    test_view_parquet_file()

