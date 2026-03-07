from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load models
pcos_model = joblib.load("RandomForest_PCOS.pkl")
scaler = joblib.load("scaler.pkl")
cycle_model = joblib.load("cycle_model.pkl")
cycle_imputer = joblib.load("cycle_imputer.pkl")


@app.route('/')
def home():
    return render_template('index.html')


# ==========================
# 🔹 PREDICT PCOS ROUTE
# ==========================
@app.route('/predict_pcos', methods=['POST'])
def predict_pcos():
    try:
        # Collect input data
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        cycle_length = float(request.form['cycle_length'])
        hirsutism = int(request.form['hirsutism'])
        acne = int(request.form['acne'])
        hair_thinning = int(request.form['hair_thinning'])
        fast_food = int(request.form['fast_food'])
        exercise = int(request.form['exercise'])

        # ✅ Age validation
        if age < 9:
            return render_template(
                'result.html',
                age=age,
                prediction_text="⚠️ Age must be 9 or above to perform PCOS analysis."
            )

        # Calculate BMI
        bmi = round(weight / ((height / 100) ** 2), 2)

        # Prepare features
        features = np.array([[age, weight, height, bmi, cycle_length,
                              hirsutism, acne, hair_thinning, fast_food, exercise]])
        scaled_features = scaler.transform(features)

        # Predict PCOS
        prediction_prob = pcos_model.predict_proba(scaled_features)[0][1] * 100
        prediction = pcos_model.predict(scaled_features)[0]

        # Prepare result and advice
        if prediction == 1:
            result = f"⚠️ PCOS Detected — Estimated likelihood: {prediction_prob:.2f}%"
            suggestions = [
                "Adopt a balanced, low-glycemic diet.",
                "Exercise regularly and maintain a healthy weight.",
                "Prioritize sleep and reduce stress.",
                "Consult a gynecologist or endocrinologist.",
                "Consider regular tracking of cycles and symptoms."
            ]
        else:
            result = f"✅ No PCOS Detected — Estimated likelihood: {prediction_prob:.2f}%"
            suggestions = [
                "Maintain your current healthy routine.",
                "Keep exercising regularly.",
                "Stay hydrated and eat balanced meals."
            ]

        return render_template(
            "result.html",
            prediction_text=result,
            bmi=bmi,
            suggestions=suggestions,
            age=age
        )

    except Exception as e:
        return render_template("result.html", prediction_text=f"⚠️ Error: {str(e)}")


# ==========================
# 🔹 PREDICT CYCLE LENGTH ROUTE
# ==========================
@app.route('/predict_cycle', methods=['POST'])
def predict_cycle():
    try:
        # Input from form
        age = float(request.form['age'])
        cycle_number = float(request.form['cycle_number'])
        conception_cycle = request.form['conception_cycle']
        conception_cycle = 1 if conception_cycle.lower() == "yes" else 0

        # ✅ Age validation
        if age < 9:
            return render_template(
                "result.html",
                age=age,
                prediction_text="⚠️ Invalid age: must be 9 or older for cycle prediction."
            )

        # Prepare input for model
        features = np.array([[age, cycle_number, conception_cycle]])
        features_imputed = cycle_imputer.transform(features)

        # Predict cycle length
        prediction = cycle_model.predict(features_imputed)[0]
        result = f"🩸 Predicted Menstrual Cycle Length: {prediction:.2f} days"

        return render_template(
            "result.html",
            prediction_text=result,
            cycle_result=f"{prediction:.2f}",
            suggestions=[],
            age=age
        )

    except Exception as e:
        return render_template("result.html", prediction_text=f"⚠️ Error: {str(e)}")


import webbrowser
import threading

if __name__ == "__main__":
    # Automatically open default browser
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")

    threading.Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
