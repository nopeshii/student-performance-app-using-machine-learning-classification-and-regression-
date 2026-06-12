import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

model = joblib.load("student_score_model.pkl")

st.title("🎓 Student Performance Prediction")

st.write("Predict Math Score using Reading and Writing Scores")

reading = st.slider(
    "Reading Score",
    0,
    100,
    70
)

writing = st.slider(
    "Writing Score",
    0,
    100,
    70
)

if st.button("Predict"):

    data = pd.DataFrame({
        "reading score": [reading],
        "writing score": [writing]
    })

    prediction = model.predict(data)

    score = round(prediction[0], 2)

    st.success(
        f"Predicted Math Score: {score}"
    )

    if score >= 40:
        st.success("PASS ✅")
    else:
        st.error("FAIL ❌")

st.subheader("Visualizations")

st.image("reading_vs_math.png")
st.image("writing_vs_math.png")
st.image("math_distribution.png")