from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("Rainfall.pkl", "rb"))
scaler = pickle.load(open("Scale.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = ""

    if request.method == "POST":
        try:
            # Get values from form (names must match HTML)
            min_temp = float(request.form["mintemp"])
            max_temp = float(request.form["maxtemp"])
            humidity = float(request.form["humidity"])
            rainfall = float(request.form["rainfall"])

            # Prepare input
            input_data = np.array([[min_temp, max_temp, humidity, rainfall]])
            input_scaled = scaler.transform(input_data)

            # Get probability
            prob = model.predict_proba(input_scaled)[0][1]

            # Lower threshold to handle imbalance
            if prob >= 0.3:
                prediction = f"🌧️ Rain Tomorrow (Probability: {prob*100:.2f}%)"
            else:
                prediction = f"☀️ No Rain Tomorrow (Probability: {prob*100:.2f}%)"

        except Exception as e:
            prediction = f"❌ Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
