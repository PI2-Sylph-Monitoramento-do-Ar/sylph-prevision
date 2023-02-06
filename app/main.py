from typing import Union
from fastapi import FastAPI
import numpy as np
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel

class ForecastBody(BaseModel):
    hour_history: list[float]
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Access /docs to know more about this api"}

@app.post("/prediction")
async def calculate_forecast(forecast_body: ForecastBody) -> list[float]:
    #Recebendo array de 24 valores para 24 horas
    data = np.array(forecast_body.hour_history)

    #Reshape
    data = data.reshape(-1,1)

    #Treinando o modelo usando e preparando para extracao de 6 valores
    model = LinearRegression().fit(data[:-6], data[6:])

    #prevendo
    prediction = model.predict(data)
    next_six_hours_prediction = prediction[-6:]
    result = []
    for array in next_six_hours_prediction:
        for item in array:
            result.append(item)
    return result