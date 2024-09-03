import streamlit as st
import pandas as pd
import numpy as np
import pickle
import boto3
from io import StringIO
from botocore.exceptions import ClientError

# Load the saved model
model_periapical = pickle.load(open('Periapical_Diagnosis_Prediction.sav', 'rb'))
model_pulpal = pickle.load(open('pulpal_Diagnosis_Prediction.sav', 'rb'))

aws_access_key_id = 'AKIAVRUVTPOM3YR6OWHO'
aws_secret_access_key = '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'
region_name = 'ap-southeast-2'
    
    # Initialize an S3 client
s3 = boto3.client('s3', 
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name)

bucket_name = 'foodqast'
file_name = 'patients.csv'

# Function to check if file exists in S3
def check_file_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False

# Function to read CSV from S3
def read_csv_from_s3(bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    except ClientError:
        return pd.DataFrame()

# Function to write CSV to S3
def write_csv_to_s3(df, bucket, key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

def test_s3_record_addition():
    # aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    # aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    # region_name = os.getenv('AWS_REGION', 'ap-southeast-2')
    # bucket_name = os.getenv('S3_BUCKET_NAME')
    # file_name = 'patients.csv'

    # s3 = boto3.client('s3', 
    #                   aws_access_key_id=aws_access_key_id,
    #                   aws_secret_access_key=aws_secret_access_key,
    #                   region_name=region_name)

    # Read the file from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

    # Print the last record (the newly added one)
    st.write("Newly added record:")
    st.dataframe(df)

    # Print the total number of records
    st.write(f"\nTotal number of records: {len(df)}")


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
            swelling_eo = st.checkbox('Extra-Oral Swelling')
            swelling_io = st.checkbox('Intra-Oral Swelling')
            sinus_tract = st.checkbox('Sinus Tract')
            tooth_open_history = st.selectbox('Was the tooth open Earlier ?', ['Yes', 'No', 'NA'])
        with sub_col2:
            palpation = st.checkbox('Palpation')
            percussion = st.checkbox('Percussion')
            mobility = st.checkbox('Mobility')
        
    # Convert the input strings to numeric values
    tooth_open_history = 0 if tooth_open_history == 'No' else 1 if tooth_open_history == 'Yes' else 99
    # Convert the input values to numeric
    boolean_features = {
        'swelling_eo': swelling_eo,
        'swelling_io': swelling_io,
        'sinus_tract': sinus_tract,
        'palpation': palpation,
        'percussion': percussion,
        'mobility': mobility
    }
    
    # Convert boolean values to 0 or 1
    numeric_features = {k: int(v) for k, v in boolean_features.items()}
    st.write('palpation : ', palpation)
    st.write('percussion : ', percussion)
    st.write('mobility : ', mobility)
    st.write('swelling_eo : ', swelling_eo)
    st.write('swelling_io : ', swelling_io)
    st.write('sinus_tract : ', sinus_tract)

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
            predicted_class_pulpal = 'reversible pulpitis'

        # Display predicted class and probabilities
        # st.markdown(f'<p style="font-size:24px; color:black; font-weight:bold;">Diagnosis :</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:22px; color:#2e6c80;"><strong>Pulpal diagnosis:</strong> {predicted_class_pulpal}  </p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:22px; color:#2e6c80;"><strong>Periapical diagnosis:</strong> {predicted_class_periapical}  </p>', unsafe_allow_html=True)

        # Save the record to S3
        new_record = pd.DataFrame([features], columns=[
            'pain_score', 'Painkiller_usage', 'Pain_duration', 'affected_tooth', 'tooth_open_history', 
            'palpation', 'percussion', 'mobility', 'PAI_1', 'PAI_2', 'PAI_3', 'PAI_4', 'PAI_5', 
            'acceptability', 'swelling_eo', 'swelling_io', 'sinus_tract', 'Pulp_Vitality'
        ])
        new_record['periapical_diagnosis'] = predicted_class_periapical
        new_record['pulpal_diagnosis'] = predicted_class_pulpal

        try:
            if check_file_exists(bucket_name, file_name):
                existing_df = read_csv_from_s3(bucket_name, file_name)
                updated_df = pd.concat([existing_df, new_record], ignore_index=True)
            else:
                updated_df = new_record

            write_csv_to_s3(updated_df, bucket_name, file_name)
            st.success("Record saved successfully!")
        except Exception as e:
            st.error(f"An error occurred while saving the record: {str(e)}")
        

if __name__ == "__main__":
    main()
    test_s3_record_addition()
