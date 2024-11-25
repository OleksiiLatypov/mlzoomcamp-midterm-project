import pickle
from flask import Flask
from flask import request
from flask import jsonify



model_file = 'model.bin'



features_use = ['gender',
    'age',
    'working_professional_or_student',
    'profession',
    'work_pressure',
    'job_satisfaction',
    'sleep_duration',
    'dietary_habits',
    'degree',
    'have_you_ever_had_suicidal_thoughts',
    'work/study_hours',
    'financial_stress',
    'family_history_of_mental_illness']





model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print('Model loaded successfully')

app = Flask('depression')

@app.route('/predict', methods=['POST'])
def predict():
    person = request.get_json()

    X = dv.transform([person])
    y_pred = model.predict_proba(X)[0, 1]
    depression = y_pred >= 0.5

    result = {
        'depression_probability': float(y_pred),
        'depression': bool(depression)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)