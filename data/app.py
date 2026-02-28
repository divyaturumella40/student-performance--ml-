import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Features and Target
X = data[['study_hours', 'attendance', 'previous_score']]
y = data['final_score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor()

# Train models
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Predictions
lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Accuracy
lr_r2 = r2_score(y_test, lr_pred)
dt_r2 = r2_score(y_test, dt_pred)

# Streamlit UI
st.title("🎓 Student Performance Prediction System")

st.write("This Machine Learning model predicts the final score based on:")
st.write("- Study Hours")
st.write("- Attendance")
st.write("- Previous Score")

# Sliders
study_hours = st.slider("Study Hours", 0, 12, 4)
attendance = st.slider("Attendance (%)", 50, 100, 80)
previous_score = st.slider("Previous Score", 0, 100, 70)

model_choice = st.selectbox("Choose Model", ["Linear Regression", "Decision Tree"])

if st.button("Predict Final Score"):

    input_data = [[study_hours, attendance, previous_score]]

    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)
    else:
        prediction = dt_model.predict(input_data)

    st.success(f"Predicted Final Score: {prediction[0]:.2f}")

# Show model accuracy
st.subheader("📊 Model Accuracy (R2 Score)")
st.write(f"Linear Regression R2 Score: {lr_r2:.2f}")
st.write(f"Decision Tree R2 Score: {dt_r2:.2f}")

# Visualization
st.subheader("📈 Study Hours vs Final Score")

fig, ax = plt.subplots()
ax.scatter(data['study_hours'], data['final_score'])
ax.set_xlabel("Study Hours")
ax.set_ylabel("Final Score")
st.pyplot(fig)
