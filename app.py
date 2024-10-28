import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"C:\Users\ivika\Downloads\diabetes.csv")

# Streamlit app title and data description
st.title("Diabetes Checkup")
st.subheader("Training Data")
st.write(df.describe())

# Visualization
st.subheader("Visualization")
st.bar_chart(df)

# Prepare the data for modeling
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to capture user input via sidebar sliders
def user_report():
    Pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    Glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    BloodPressure = st.sidebar.slider("BloodPressure", 0, 122, 70)
    SkinThickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    Insulin = st.sidebar.slider("Insulin", 0, 846, 79)
    BMI = st.sidebar.slider("BMI", 0, 67, 20)
    DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.4, 0.47)
    Age = st.sidebar.slider("Age", 21, 88, 33)

    user_report_data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age,
    }

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Get user input data
user_data = user_report()

# Train the Random Forest model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Display model accuracy
st.subheader("Model Accuracy:")
st.write(str(accuracy_score(y_test, rf.predict(x_test)) * 100) + '%')

# Make a prediction for the user input
user_result = rf.predict(user_data)

# Display the result to the user
st.subheader("Your Report:")
output = 'The Person is Not diabetic' if user_result[0] == 0 else 'The Person is diabetic'
st.write(output)
