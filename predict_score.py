import pandas as pd
import joblib

model = joblib.load("student_score_model.pkl")

reading = float(input("Enter Reading Score: "))
writing = float(input("Enter Writing Score: "))

data = pd.DataFrame({
    "reading score": [reading],
    "writing score": [writing]
})

prediction = model.predict(data)

print("\nPredicted Math Score:", round(prediction[0], 2))