import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()


class Data(BaseModel):
    model: str
    parameters: dict


def make_prediction(data: Data):
    with open(data.model, "rb") as f:
        model = joblib.load(f)

    data_df = pd.DataFrame.from_records([data.parameters])
    prediction = model.predict(data_df)

    return float(prediction[0])

@app.post("/predict")
def predict(data: Data):
    return {"prediction": make_prediction(data)}
