# Import libraries
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the Model and Preprocess from file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Display sidebar
st.write("""
# Valeo Defect Prediction App
This app predicts the defects on the Valeo production line!
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/nicolas-szb/Valeo-defect-prediction/main/data/defect_examples.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_features = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        PROC_TRACEINFO = st.sidebar.text_input('PROC_TRACEINFO', 'I-B-XA1207672-190422-00287')

        OP070_V_1_angle_value = st.sidebar.slider('OP070_V_1_angle_value', 0.0, 200.0, 100.0)
        OP090_SnapRingPeakForce_value = st.sidebar.slider('OP090_SnapRingPeakForce_value', 0.0, 200.0, 100.0)
        OP070_V_2_angle_value = st.sidebar.slider('OP070_V_2_angle_value', 0.0, 200.0, 100.0)
        OP120_Rodage_I_mesure_value = st.sidebar.slider('OP120_Rodage_I_mesure_value', 0.0, 200.0, 100.0)
        OP090_SnapRingFinalStroke_value = st.sidebar.slider('OP090_SnapRingFinalStroke_value', 0.0, 200.0, 100.0)
        OP110_Vissage_M8_torque_value = st.sidebar.slider('OP110_Vissage_M8_torque_value', 0.0, 200.0, 100.0)
        OP100_Capuchon_insertion_mesure = st.sidebar.slider('OP100_Capuchon_insertion_mesure', 0.0, 200.0, 100.0)
        OP120_Rodage_U_mesure_value = st.sidebar.slider('OP120_Rodage_U_mesure_value', 0.0, 200.0, 100.0)
        OP070_V_1_torque_value = st.sidebar.slider('OP070_V_1_torque_value', 0.0, 200.0, 100.0)
        OP090_StartLinePeakForce_value = st.sidebar.slider('OP090_StartLinePeakForce_value', 0.0, 200.0, 100.0)
        OP110_Vissage_M8_angle_value = st.sidebar.slider('OP110_Vissage_M8_angle_value', 0.0, 200.0, 100.0)
        OP090_SnapRingMidPointForce_val = st.sidebar.slider('OP090_SnapRingMidPointForce_val', 0.0, 200.0, 100.0)
        OP070_V_2_torque_value = st.sidebar.slider('OP070_V_2_torque_value', 0.0, 200.0, 100.0)

        data = {'PROC_TRACEINFO': PROC_TRACEINFO,
                'OP070_V_1_angle_value': OP070_V_1_angle_value,
                'OP090_SnapRingPeakForce_value': OP090_SnapRingPeakForce_value,
                'OP070_V_2_angle_value': OP070_V_2_angle_value,
                'OP120_Rodage_I_mesure_value': OP120_Rodage_I_mesure_value,
                'OP090_SnapRingFinalStroke_value': OP090_SnapRingFinalStroke_value,
                'OP110_Vissage_M8_torque_value': OP110_Vissage_M8_torque_value,
                'OP100_Capuchon_insertion_mesure': OP100_Capuchon_insertion_mesure,
                'OP120_Rodage_U_mesure_value': OP120_Rodage_U_mesure_value,
                'OP070_V_1_torque_value': OP070_V_1_torque_value,
                'OP090_StartLinePeakForce_value': OP090_StartLinePeakForce_value,
                'OP110_Vissage_M8_angle_value': OP110_Vissage_M8_angle_value,
                'OP090_SnapRingMidPointForce_val': OP090_SnapRingMidPointForce_val,
                'OP070_V_2_torque_value': OP070_V_2_torque_value}

        features = pd.DataFrame(data, index=[0])
        return features
    input_features = user_input_features()

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_features)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_features)

# Select features after testing with different model's parameters
features_dropped = []
features_dropped.extend(['PROC_TRACEINFO',
                         'OP070_V_1_angle_value',
                         'OP070_V_2_angle_value',
                         'OP120_Rodage_I_mesure_value',
                         'OP110_Vissage_M8_torque_value',
                         'OP100_Capuchon_insertion_mesure',
                         'OP120_Rodage_U_mesure_value',
                         'OP110_Vissage_M8_angle_value',
                         'OP090_SnapRingMidPointForce_val'])
input_features_dropped = input_features.drop(features_dropped,
                      axis=1)

# Apply model to make predictions
prediction = model.predict(input_features_dropped)
prediction_proba = model.predict_proba(input_features_dropped)

# Display predictions
st.subheader('Prediction')
defect_yes_no = np.array(['No Defect','Defect'])
st.write(defect_yes_no[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
