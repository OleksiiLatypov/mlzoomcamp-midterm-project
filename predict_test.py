import requests


url = 'http://localhost:9696/predict'


person = {'gender': 'Female',
  'age': 40.0,
  'working_professional_or_student': 'Working Professional',
  'profession': 'Researcher',
  'work_pressure': '3.0',
  'job_satisfaction': '3.0',
  'sleep_duration': 'Less than 5 hours',
  'dietary_habits': 'Healthy',
  'degree': 'B.Tech',
  'have_you_ever_had_suicidal_thoughts': 'Yes',
  'work/study_hours': 3.0,
  'financial_stress': '3.0',
  'family_history_of_mental_illness': 'Yes'}


response = requests.post(url, json=person).json()
print(response)

if response['depression'] == True:
    print('Person has depression and need help.')
else:
    print('The person does not have depression.')


    