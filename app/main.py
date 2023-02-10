from typing import Union
from fastapi import FastAPI
import numpy as np
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel
import numbers
class ForecastBody(BaseModel):
    hour_history: list
app = FastAPI()

def sanitize_array(array):
    new_array = []
    for item in array:
        if(isinstance(item, numbers.Number)):
            new_array.append(item)
    return new_array

@app.get("/")
def read_root():
    return {"Hello": "Access /docs to know more about this api"}

@app.post("/prediction")
async def calculate_forecast(forecast_body: ForecastBody) -> float:
    sanitized_array = sanitize_array(forecast_body.hour_history)
    print(sanitized_array)
    data = np.array(sanitized_array)
    data = data.reshape(-1,1)
    model = LinearRegression().fit(data[:-1], data[1:])
    prediction = model.predict(data)
    result = prediction[0][0]
    return result