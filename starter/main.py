# Put the code for your API here.
import pandas as pd
import pickle
from typing import Union, List
from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics, compute_slice_metric

class Adult(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }

app = FastAPI()

@app.get("/")
async def welcome(user: str = "User"):
    return {"greeting": f"Welcome {user}!"}

@app.post("/adult/")
async def inference_adult(person: Adult):
    cat_features = [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ]

    data = {
        "age": [person.age],
        "workclass": [person.workclass],
        "fnlgt": [person.fnlgt],
        "education": [person.education],
        "education-num": [person.education_num],
        "marital-status": [person.marital_status],
        "occupation": [person.occupation],
        "relationship": [person.relationship],
        "race": [person.race],
        "sex": [person.sex],
        "capital-gain": [person.capital_gain],
        "capital-loss": [person.capital_loss],
        "hours-per-week": [person.hours_per_week],
        "native-country": [person.native_country]
            }
    raw_input = pd.DataFrame(data)

    cur_path = str(Path(__file__).parent.absolute())

    # Load data parameters
    with open(cur_path + '/data/encoder.pkl','rb') as f:
        encoder = pickle.load(f)
    with open(cur_path + '/data/lb.pkl','rb') as f:
        lb = pickle.load(f)

    # Load model
    with open(cur_path + '/model/model.pkl','rb') as f:
        clf = pickle.load(f)

    # Process the raw input data
    input, _, _, _ = process_data(
            raw_input, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
            )
    # Model inference
    y_pred = inference(clf, input)
    label = y_pred.item()

    if label==0:
        output = "<=50K"
    else:
        output = ">50K" 

    return {"fetch": f"The estimated salary is {output}"}