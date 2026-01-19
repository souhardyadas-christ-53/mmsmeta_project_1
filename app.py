from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# =====================================
# LOAD FINAL LINEAR REGRESSION MODEL
# =====================================
with open("model/linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# =====================================
# HOME ROUTE
# =====================================
@app.route("/")
def home():
    return render_template("index.html")

# =====================================
# PREDICTION ROUTE (API)
# =====================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # 🔒 FEATURE ORDER MUST MATCH TRAINING
        features = np.array([[
            float(data["study_hours_per_week"]),
            float(data["attendance_percentage"]),
            float(data["previous_score"]),
            float(data["assignments_completed"]),
            float(data["sleep_hours"]),
            float(data["class_participation"]),
            float(data["internet_quality"]),
            float(data["extracurricular_hours"]),
            float(data["assignment_completion_rate"])
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": "Invalid input data"
        }), 400

# =====================================
# RUN SERVER
# =====================================
if __name__ == "__main__":
    app.run(debug=True)
