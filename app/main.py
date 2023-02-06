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
async def calculate_forecast(forecast_body: ForecastBody) -> float:
    data = np.array(forecast_body.hour_history)
    data = data.reshape(-1,1)
    model = LinearRegression().fit(data[:-1], data[1:])
    prediction = model.predict(data)
    print(prediction)
    result = prediction[0][0]
    return result