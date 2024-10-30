import pandas as pd
import streamlit as st
import numpy as np
import joblib
import openai
import requests
from streamlit_chat import message
from dotenv import load_dotenv
import os

## Load environment variables
load_dotenv()

AZURE_OPENAI_ENDPOINT=os.getenv("MY_ENDPOINT_KEY")    
API_KEY=os.getenv("MY_API_KEY")

# Load the trained model and scaler
# Ensure that `model.pkl` and `scaler.pkl` exist in the `pkl` directory within your main directory
try:
    Model = joblib.load("pkl/model.pkl")
    Scaler = joblib.load("pkl/scaler.pkl")
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# Cache data loading for efficiency
@st.cache_data
def load_data():
    # Read split CSV files
    try:
        df1 = pd.read_csv('datasets/split_1.csv')
        df2 = pd.read_csv('datasets/split_2.csv')
        df3 = pd.read_csv('datasets/split_3.csv')
        df4 = pd.read_csv('datasets/split_4.csv')
        df5 = pd.read_csv('datasets/split_5.csv')

    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.stop()
    
    # Concatenate them into one DataFrame
    final_df = pd.concat([df1, df2, df3,df4,df5])
    return final_df

# Load recommendations dictionary
@st.cache_data
def load_recommendations():
    try:
        return joblib.load("pkl/recommendations_compressed.pkl")
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

# Initialize data
df = load_data()
recommendations_dict = load_recommendations()

# Mapping for activity intensity and gender (for encoding)
intensity_multipliers = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extra_active': 1.9
}
objective_adjustments = {
    'weight_loss': 0.8,
    'muscle_gain': 1.2,
    'health_maintenance': 1
}

# BMR and caloric intake calculations
def compute_bmr(gender, body_weight, body_height, age):
    if gender == 'male':
        bmr_value = 10 * body_weight + 6.25 * body_height - 5 * age + 5
    else:
        bmr_value = 10 * body_weight + 6.25 * body_height - 5 * age - 161
    return bmr_value

def compute_daily_caloric_intake(bmr, activity_intensity, objective):
    maintenance_calories = bmr * intensity_multipliers[activity_intensity]
    total_caloric_intake = maintenance_calories * objective_adjustments[objective]
    return round(total_caloric_intake)

# Main recommendation function
def suggest_recipes(gender, body_weight, body_height, age, activity_intensity, objective):
    bmr = compute_bmr(gender, body_weight, body_height, age)
    total_calories = compute_daily_caloric_intake(bmr, activity_intensity, objective)
    user_input_features = np.array([[total_calories, 0, 0, 0, 0, 0, 0, 0, 0]])
    scaled_input_features = Scaler.transform(user_input_features)
    predicted_latent_features = Model.predict(scaled_input_features)
    top_prediction_index = np.argmax(predicted_latent_features.flatten())
    similar_recipe_indices = np.array(recommendations_dict[top_prediction_index])
    recommended_recipes = df.iloc[similar_recipe_indices[:, 1].astype(int)][['Name', 'Calories']]
    return recommended_recipes.head(5)

# Get response from OpenAI API
def get_response_from_openai(prompt):
    headers = {
        'Content-Type': 'application/json',
        'api-key': API_KEY
    }
    data = {
        'messages': [
            {"role": 'system', 'content': 'You are a diet assistant.'},
            {"role": 'user', 'content': prompt}
        ],
        'max_tokens': 150,
        'temperature': 0.7
    }
    response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]["message"]["content"]
    else:
        return 'Error: unable to connect to Azure service for chatting.'

# Streamlit UI Code
st.title("Diet Recommender Web App")
st.image("Images/logo.png", width=300)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Diet Page", "Disease Page", "Chatbot"])

if page == "Home":
    st.header("Welcome to the Diet Recommender!")

elif page == "Diet Page":
    st.header("Diet Page")
    st.image("Images/diet.png", width=300)

    gender = st.radio("Gender", ["Male", "Female"], key="gender_input").lower()
    weight = st.number_input("Weight (kg)", min_value=1.0, key="weight_input")
    height = st.number_input("Height (cm)", min_value=1.0, key="height_input")
    age = st.slider("Age", 1, 100, key="age_input")
    intensity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"], key="intensity_input").lower().replace(' ', '_')
    objective = st.selectbox("Objective", ["Weight Loss", "Muscle Gain", "Health Maintenance"], key="objective_input").lower().replace(' ', '_')

    if st.button("Submit"):
        required_calories = compute_daily_caloric_intake(
            bmr=compute_bmr(gender, weight, height, age),
            activity_intensity=intensity,
            objective=objective
        )
        st.subheader(f"Required Daily Calories: {required_calories} kcal")
        suggested_recipes = suggest_recipes(gender, weight, height, age, intensity, objective)
        if suggested_recipes is not None and not suggested_recipes.empty:
            st.subheader("Top 5 Suggested Recipes:")
            for idx, recipe in suggested_recipes.iterrows():
                st.write(f"*{idx + 1}. {recipe['Name']}* - {recipe['Calories']} Calories")
                st.write('-' * 40)
        else:
            st.warning("No recipes found for your preferences.")

elif page == "Disease Page":
    st.header("Disease Page")
    st.image("Images/doctor.png", width=300)
    health_conditions = st.multiselect("Select health conditions:", ['HeartDisease', 'Diabetes', 'HighBloodPressure', 'Obesity', 'Hyperglycemia', 'KidneyDisease'], key="health_conditions_input")
    calorie_range = st.slider("Calorie Range", 0, 1500, (200, 800), key="calories_input")
    allergies = st.text_input("Enter allergies (e.g., nuts, milk):", key="allergies_input")

    if st.button("Filter Recipes", key="filter_submit"):
        filtered_recipes = df[(df['Calories'] >= calorie_range[0]) & (df['Calories'] <= calorie_range[1])]
        if not filtered_recipes.empty:
            st.write("Recipes suitable for you:")
            st.dataframe(filtered_recipes[['Name', 'Calories', 'RecipeIngredientParts']])
            meal_counts = filtered_recipes['Name'].value_counts()
            st.bar_chart(meal_counts)
            st.download_button(
                label="Download Recipes as CSV",
                data=filtered_recipes.to_csv(index=False),
                file_name='filtered_recipes.csv',
                mime='text/csv',
            )
        else:
            st.warning("No recipes found for your preferences.")

elif page == "Chatbot":
    st.header("Chat with DietBot")
    st.image("Images/file.png", width=300)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if st.session_state['messages']:
        st.subheader('Chat History')
        for msg in st.session_state['messages']:
            if msg['role'] == 'user':
                st.write(f"*You*: {msg['content']}")
            else:
                st.write(f"*DietBot*: {msg['content']}")

    user_input = st.text_input('Enter your question: ')
    if st.button('Send'):
        if user_input:
            st.session_state['messages'].append({'role': 'user', 'content': user_input})
            bot_response = get_response_from_openai(user_input)
            st.session_state['messages'].append({'role': 'assistant', 'content': bot_response})
            st.write(f"*DietBot*: {bot_response}")
        else:
            st.warning("Please enter a message.")
