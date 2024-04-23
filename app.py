import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Initialize Spark
spark = SparkSession.builder.appName("Model Deployment").getOrCreate()

# Load the trained GBT model
model_path = "/path/to/save/model_rf"
model = PipelineModel.load(model_path)

def make_prediction(features):
    # Convert input data to Spark DataFrame
    columns = ['age', 'gender', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 
               'restingrelectro', 'maxheartrate', 'oldpeak', 'slope', 'noofmajorvessels', 
               'chestpain_x_exerciseangia']
    data = spark.createDataFrame([features], schema=columns)
    # Predict using the loaded model
    prediction = model.transform(data)
    return prediction.select('prediction').collect()[0]['prediction']

st.set_page_config(layout="wide")  # Use the full page width
# Define your custom CSS
custom_css = """
    <style>
        /* Add your custom CSS styles here */
        .title {
            color: #FF5733; /* Change title color to orange */
            font-size: 36px; /* Increase font size of the title */
            text-align: center; /* Center-align the title */
        }

        /* Add more custom styles as needed */
    </style>
"""

# Render the custom CSS using st.markdown
st.markdown(custom_css, unsafe_allow_html=True)
st.title('Heart Disease Prediction')

with st.form(key='input_form'):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=50, step=1, help="Enter your age")
        gender = st.selectbox('Gender', options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female', help="Select your gender")
        restingBP = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120, help="Enter your resting blood pressure")

    with col2:
        serumcholestrol = st.number_input('Serum Cholesterol', min_value=0, max_value=600, value=200, help="Enter your serum cholesterol level")
        fastingbloodsugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[1, 0], help="Select if your fasting blood sugar is greater than 120 mg/dl")
        restingrelectro = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2], help="Select your resting electrocardiographic results")

    with col3:
        maxheartrate = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, value=150, help="Enter your maximum heart rate achieved")
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Enter ST depression induced by exercise relative to rest")
        slope = st.selectbox('Slope of the peak exercise ST segment', options=[0, 1, 2], help="Select the slope of the peak exercise ST segment")

    with col4:
        noofmajorvessels = st.number_input('Number of major vessels', min_value=0, max_value=4, value=0, step=1, help="Enter the number of major vessels colored by flourosopy")
        chestpain_x_exerciseangia = st.selectbox('Exercise induced angina', options=[0, 1], help="Select if exercise induced angina")

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    features = [age, gender, restingBP, serumcholestrol, fastingbloodsugar, 
                restingrelectro, maxheartrate, oldpeak, slope, noofmajorvessels, 
                chestpain_x_exerciseangia]
    prediction = make_prediction(features)
    if prediction == 1:
        st.write("There is a significant chance that the person has heart disease.")
    else:
        st.write("There is a low chance that the person has heart disease.")


# Add any custom CSS to the Streamlit app using st.markdown or other Streamlit commands
