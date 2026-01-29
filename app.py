from flask import Flask, render_template_string, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# ===============================
# Load Preprocessor & Model
# ===============================
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))


# ===============================
# HTML Template
# ===============================
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="UTF-8">
<title>Student Performance Predictor</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
    font-family:'Poppins',sans-serif;
}

body{
    min-height:100vh;
    background: linear-gradient(135deg,#667eea,#764ba2);
    display:flex;
    justify-content:center;
    align-items:center;
}

.container{
    background:#fff;
    width:500px;
    padding:35px;
    border-radius:20px;
    box-shadow:0 25px 60px rgba(0,0,0,0.3);
    text-align:center;
}

h1{
    margin-bottom:10px;
}

form{
    display:flex;
    flex-direction:column;
    gap:12px;
}

input, select{
    padding:10px;
    border-radius:6px;
    border:1px solid #ddd;
}

button{
    margin-top:10px;
    padding:12px;
    background:#667eea;
    color:white;
    border:none;
    border-radius:8px;
    font-size:16px;
    cursor:pointer;
}

button:hover{
    background:#5563d8;
}

.result{
    margin-top:20px;
    padding:15px;
    background:#f1f3ff;
    border-radius:10px;
    font-weight:600;
}

footer{
    margin-top:15px;
    font-size:12px;
    color:#777;
}

</style>
</head>


<body>

<div class="container">

<h1>📊 Student Performance Predictor</h1>

<form method="POST">

<select name="gender" required>
<option>male</option>
<option>female</option>
</select>

<select name="race" required>
<option>group A</option>
<option>group B</option>
<option>group C</option>
<option>group D</option>
<option>group E</option>
</select>

<select name="parent_edu" required>
<option>bachelor's degree</option>
<option>some college</option>
<option>master's degree</option>
<option>associate's degree</option>
<option>high school</option>
</select>

<select name="lunch" required>
<option>standard</option>
<option>free/reduced</option>
</select>

<select name="prep" required>
<option>none</option>
<option>completed</option>
</select>

<input type="number" name="reading" placeholder="Reading Score" required>
<input type="number" name="writing" placeholder="Writing Score" required>

<button type="submit">Predict 🚀</button>

</form>


{% if prediction %}
<div class="result">
🎯 Predicted Math Score: {{ prediction }}
</div>
{% endif %}

<footer>
Developed by Abdul Rehman | AI Project
</footer>

</div>

</body>
</html>
"""


# ===============================
# Flask Route
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        try:

            # Get Inputs
            gender = request.form.get("gender")
            race = request.form.get("race")
            parent = request.form.get("parent_edu")
            lunch = request.form.get("lunch")
            prep = request.form.get("prep")

            reading = float(request.form.get("reading"))
            writing = float(request.form.get("writing"))

            # Create DataFrame (IMPORTANT)
            input_data = pd.DataFrame({

                "gender": [gender],
                "race_ethnicity": [race],
                "parental level of education": [parent],
                "lunch": [lunch],
                "test preparation course": [prep],
                "reading score": [reading],
                "writing score": [writing]

            })

            # Preprocess
            transformed = preprocessor.transform(input_data)

            # Predict
            result = model.predict(transformed)[0]

            prediction = round(result, 2)

        except Exception as e:
            prediction = "Error: " + str(e)

    return render_template_string(TEMPLATE, prediction=prediction)


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
