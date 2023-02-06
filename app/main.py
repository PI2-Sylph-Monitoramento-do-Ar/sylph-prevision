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