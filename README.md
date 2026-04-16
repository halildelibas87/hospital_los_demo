# Patient Length of Stay Prediction

This project predicts a patient's expected hospital length of stay using demographic, medical, and admission-related information.

## Business Problem
Hospitals need better visibility into expected patient stay duration for:
- bed planning
- staffing
- patient flow optimization
- operational efficiency

## Project Type
Machine Learning Regression

## Target Variable
`length_of_stay = Discharge Date - Date of Admission`

## Features Used
- Age
- Gender
- Blood Type
- Medical Condition
- Insurance Provider
- Admission Type
- Medication
- Test Results

## Project Structure
```bash
hospital_los_demo/
│
├── app.py
├── train.py
├── eda.py
├── requirements.txt
├── README.md
├── data/
│   └── healthcare_dataset.csv
├── model/
│   ├── los_model.pkl
│   ├── model_info.pkl
│   └── metrics.json
└── outputs/
