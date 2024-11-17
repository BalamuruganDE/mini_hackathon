
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib

app= FastAPI()

class Input(BaseModel):
    department:object
    region:object
    education:object
    gender:object
    recruitment_channel:object
    no_of_trainings:int
    age:int
    previous_year_rating:float
    length_of_service:int
    KPIs_met:int
    awards_won:int
    avg_training_score:int


class Output(BaseModel):
    target:int


@app.post("/predict")
def predict(data:Input) -> Output:
    X_input= pd.DataFrame([[
    data.department,
    data.region,
    data.education,
    data.gender,
    data.recruitment_channel,
    data.no_of_trainings,
    data.age,
    data.previous_year_rating,
    data.length_of_service,
    data.KPIs_met,
    data.awards_won,
    data.avg_training_score
    ]])
    X_input.columns=[
    'department',
    'region',
    'education',
    'gender',
    'recruitment_channel',
    'no_of_trainings',
    'age',
    'previous_year_rating',
    'length_of_service',
    'KPIs_met',
    'awards_won',
    'avg_training_score']

    #load model
    # model = joblib.load('jobchg_ppln_model.pkl')
    model = joblib.load('Stacking_XGB_model.pkl')
    prediction = model.predict(X_input)
    return Output(target = prediction)

'''
{
  "department": "string",
  "region": "string",
  "education": "string",
  "gender": "string",
  "recruitment_channel": "string",
  "no_of_trainings": 0,
  "age": 0,
  "previous_year_rating": 0,
  "length_of_service": 0,
  "KPIs_met": 0,
  "awards_won": 0,
  "avg_training_score": 0
}

'''

