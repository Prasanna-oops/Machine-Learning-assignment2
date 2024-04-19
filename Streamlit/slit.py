import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('C:\Front-Back\Streamlit\Prasanna raj KJ_717822I239 (1).csv')
data = data.replace(np.nan, 0)

# Select features and target variable
x = data[['Move Minutes count', 'Distance (m)', 'Heart Points', 'Average speed (m/s)']]
y = data['Calories (kcal)']

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=1)

# Train the polynomial regression model
poly = PolynomialFeatures(degree=6)
x_poly = poly.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, y)

# Define the function to make predictions
def predict_calories(move_minutes, distance, heart_points, avg_speed):
    # Transform input features into polynomial features
    input_features = np.array([[move_minutes, distance, heart_points, avg_speed]])
    input_poly = poly.transform(input_features)
    # Make prediction using the trained model
    prediction = model.predict(input_poly)
    return prediction[0]

# Streamlit app
st.set_page_config(
    page_title="Calories Prediction App",
    page_icon=":running:",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Add some extra information
st.sidebar.markdown('### About')
st.sidebar.info(
    "This web app allows you to predict the calories burned based on "
    "the input values for Move Minutes count, Distance, Heart Points, and Average speed."
)

# Add related content
st.markdown('## Related Content')
st.write('Here are some related articles and tips to help you stay fit and healthy:')

# Add some images or links to related content
st.image('10321.jpg', use_column_width=True)
st.write('Check the Below articles and vedio maintain your fitness!!!')
st.write('**Article:** [10 Tips for a Healthy Lifestyle](https://www.healthline.com/nutrition/27-health-and-nutrition-tips)')
st.write('**Video:** [10 Minute Full Body Workout](https://www.youtube.com/watch?v=5Z7vzv9OBrM)')
st.title('Lets burn calories together...\nCheck your effort in maintaining fitness...')

# Add input widgets for user input
move_minutes = st.number_input('Move Minutes count', value=0)
distance = st.number_input('Distance (m)', value=0)
heart_points = st.number_input('Heart Points', value=0)
avg_speed = st.number_input('Average speed (m/s)', value=0)

# Make prediction when button is clicked
if st.button('Predict Calories', key='predict_button'):
    prediction = predict_calories(move_minutes, distance, heart_points, avg_speed)
    st.success(f'Predicted Calories: {prediction:.2f} kcal')

# Add a footer
st.sidebar.markdown('---')
st.sidebar.markdown(
    "Created with ❤️ by [Prasanna raj KJ](http://localhost:8501)"
)
