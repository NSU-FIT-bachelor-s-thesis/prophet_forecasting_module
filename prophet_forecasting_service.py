import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import pandas as pd

import prophet_forecasting_core


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fastapi_service.log'),
        logging.StreamHandler()
    ]
)

DEFAULT_PREDICTOR_PORT = 8082

app = FastAPI()
class HistoricalData(BaseModel):
    prices: list[float]
    last_timestamp: int# Unix timestamp (Epoch time)
    measurement_period_in_hours: int


def calculate_forecast_length(history_prices_length):
    forecast_length = history_prices_length // 4
    if forecast_length < 1:
        raise HTTPException(status_code=400, detail=f"Not enough input data.")
    return forecast_length


@app.post("/predict")
async def predict(historical_data: HistoricalData):
    logging.info(historical_data)
    last_time = pd.to_datetime(historical_data.last_timestamp, unit='s')

    forecast_length = calculate_forecast_length(len(historical_data.prices))
    try:
        ds, yhat, yhat_lower, yhat_upper = prophet_forecasting_core.run_prophet_forecast(
            historical_data.prices,
            last_time,
            str(historical_data.measurement_period_in_hours) + "h",
            forecast_length
        )

        return {
            'ds': ds,
            'yhat': yhat,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper
        }
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        predictor_port = int(os.getenv('PREDICTOR_PORT'))
    except Exception as e:
        predictor_port = DEFAULT_PREDICTOR_PORT
        logging.error(f'Failed to get environment variable value, default value was taken: predictor_port = {DEFAULT_PREDICTOR_PORT}.')

    uvicorn.run(app, host="0.0.0.0", port=predictor_port)
