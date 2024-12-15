import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os
from lightgbm import LGBMClassifier

# --- Configurations ---
st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Add CSS for styling ---
st.markdown(
    """
    <style>
        .main {
            background-color: #f9f9f9;
            font-family: Arial, sans-serif;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Title Section ---
st.title("🧠 Mental Health Prediction Dashboard")
st.markdown(
    """
    Welcome to the **Mental Health Prediction Dashboard**.  
    This tool helps predict depression likelihood based on lifestyle and personal factors.  
    **Fill in the form below to get started!**
    """
)

# Title
st.title("Understanding and Predicting Depression to Enhance Mental Health Interventions")

# --- Layout using Tabs ---
tab1, tab2, tab3 = st.tabs(["📋 Form Input", "📊 Insights & Prediction", "📈 Feature Importances"])

# --- Tab 1: Form Input ---
# --- Tab 1: Form Input ---
with tab1:
    col1, col2, col3 = st.columns([2, 1, 1], gap="large")

    # Load and display training dataset
    with col1:
        try:
            path = '../Data/train.csv'  # Update this to the correct path if necessary
            df = pd.read_csv(path)

            st.subheader("Dataset Preview")
            st.dataframe(df)  # Display the first 10 rows of the dataset
        except Exception as e:
            st.error("Failed to load dataset. Please check the file path.")

    # Enter Input
    with col2:
        st.subheader("📋 Input Form")
        gender = st.multiselect("**Gender**", df['Gender'].unique())
        city = st.multiselect("**City**", df['City'].unique())
        working = st.multiselect("**Working Professional or Student**", df['Working Professional or Student'].unique())
        profession = st.multiselect("**Profession**", df['Profession'].unique())
        sleep = st.multiselect("**Sleep Duration**", df['Sleep Duration'].unique())
        age = st.slider("**Age**", min_value=18, max_value=60, value=25)
        work_pressure = st.slider("**Work Pressure**", min_value=1, max_value=5, value=3)

    with col3:
        habit = st.multiselect("**Dietary Habits**", df['Dietary Habits'].unique())
        degree = st.multiselect("**Degree**", df['Degree'].unique())
        thoughts = st.multiselect("**Have you ever had suicidal thoughts ?**", df['Have you ever had suicidal thoughts ?'].unique())
        history = st.multiselect("**Family History of Mental Illness**", df['Family History of Mental Illness'].unique())
        satisfaction = st.slider("**Job Satisfaction**", min_value=1, max_value=5, value=4)
        work_hours = st.slider("**Work/Study Hours (per day)**", min_value=0, max_value=12, value=8)
        stress = st.slider("**Financial Stress**", min_value=1, max_value=5, value=3)

    # Submit button
    predict_button = st.button("🚀 Predict")

# --- Tab 2: Insights and Prediction ---
with tab2:
    st.header("🔍 Prediction Result")

    if 'predict_button' in locals() and predict_button:
        try:
            # Load encoders, models
            lgbm_clf = joblib.load('../Weights/best_model.joblib')
            encoders = joblib.load('../Encoders/label_encoders.joblib')

            # Create Input
            columns = ['Gender', 'Age', 'City', 'Working Professional or Student', 'Profession', 'Work Pressure', 
                       'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Degree', 
                       'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 
                       'Family History of Mental Illness']
            data_input = [gender[0], age, city[0], working[0], profession[0], work_pressure, satisfaction,
                          sleep[0], habit[0], degree[0], thoughts[0], work_hours, stress, history[0]]

            x_test = pd.DataFrame([data_input], columns=columns)

            # Using encoders
            for name, encoder in encoders.items():
                x_test[name] = encoder.transform(x_test[name])
            
            # Predict
            yhat = lgbm_clf.predict(x_test)
            
            # Mapping
            id_to_name = {
                0: "No Depression",
                1: "Depression"
            }

            # Display predicted and true results
            predicted_result = id_to_name[yhat[0]]
            # Fetch True Result from the dataset
            true_result = None
            if 'df' in locals():
                # Find the row in the dataset matching the input values
                filter_condition = (
                    (df['Gender'] == gender[0]) & 
                    (df['Age'] == age) &
                    (df['City'] == city[0]) &
                    (df['Working Professional or Student'] == working[0]) &
                    (df['Profession'] == profession[0]) &
                    (df['Work Pressure'] == work_pressure) &
                    (df['Job Satisfaction'] == satisfaction) &
                    (df['Sleep Duration'] == sleep[0]) &
                    (df['Dietary Habits'] == habit[0]) &
                    (df['Degree'] == degree[0]) &
                    (df['Have you ever had suicidal thoughts ?'] == thoughts[0]) &
                    (df['Work/Study Hours'] == work_hours) &
                    (df['Financial Stress'] == stress) &
                    (df['Family History of Mental Illness'] == history[0])
                )

                matching_row = df[filter_condition]
                if not matching_row.empty:
                    true_result = matching_row['Depression'].values[0]  # Assuming 'Depression' column is the true result
                
            # Display prediction and true result
            st.markdown(f"### Predicted Result: **{predicted_result}**")
            if true_result is not None:
                st.markdown(f"### True Result: **{'Depression' if true_result == 1 else 'No Depression'}**")
                # Provide feedback on accuracy
                if (predicted_result == "Depression" and true_result == 1) or (predicted_result == "No Depression" and true_result == 0):
                    st.success("The prediction matches the true result! 🎉")
                else:
                    st.warning("The prediction does not match the true result. Consider reviewing the input data.")
            else:
                st.error("Could not find the true result in the dataset. Please check the input values.")

            # Visualization
            st.subheader("📊 Insights")
            depression_prob = lgbm_clf.predict_proba(x_test)[0][1]  # "Depression" probability
            no_depression_prob = lgbm_clf.predict_proba(x_test)[0][0]  # "No Depression" probability

            fig, ax = plt.subplots()
            sns.barplot(x=["No Depression", "Depression"], y=[no_depression_prob * 100, depression_prob * 100], ax=ax)
            ax.set_title("Depression Prediction Distribution")
            ax.set_ylabel("Probability (%)")
            st.pyplot(fig)
        except Exception as e:
            st.error("An error occurred while making predictions. Please check the input values and try again.")

# --- Tab 3: Feature Importances ---
with tab3:
    st.header("📈 Depression Rate by Feature")

    try:
        # Load dataset
        path = '../Data/train.csv'
        df = pd.read_csv(path)
        depression_column = 'Depression'  

        if depression_column not in df.columns:
            st.error("Cột 'Depression' không có trong dataset. Vui lòng kiểm tra lại.")
        else:
            features = [
                'Gender', 'Age', 'City', 'Working Professional or Student', 'Profession',
                'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 'Dietary Habits',
                'Degree', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
                'Financial Stress', 'Family History of Mental Illness'
            ]

            for feature in features:
                if feature in df.columns:
                    feature_mean = df.groupby(feature)[depression_column].mean().reset_index()
                    fig, ax = plt.subplots(figsize=(18, 6))  
                    sns.lineplot(data=feature_mean, x=feature, y=depression_column, marker="o", ax=ax)
                    ax.set_title(f"Depression Rate by {feature}", fontsize=14)
                    ax.set_xlabel(feature, fontsize=12)
                    ax.set_ylabel("Mean Depression Rate", fontsize=12)
                    plt.xticks(rotation=60, ha="right", fontsize=10)  
                    st.pyplot(fig)
                else:
                    st.warning(f"Feature '{feature}' không tồn tại trong dataset. Bỏ qua.")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi xử lý: {e}")

# --- Footer Section ---
st.divider()
st.markdown(
    """
    Built with ❤️ using [Streamlit](https://streamlit.io).  
    Disclaimer: This is a demonstration tool and not a substitute for professional mental health advice.
    """
)
