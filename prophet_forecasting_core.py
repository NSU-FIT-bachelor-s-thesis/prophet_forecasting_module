import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def run_prophet_forecast(
    price_sequence,
    last_timestamp,
    frequency="12h",
    forecast_length=14
):
    """
    price_sequence: список или массив цен
    last_timestamp: pd.Timestamp — время последнего замера
    frequency: частота замеров
    forecast_length: сколько шагов вперёд прогнозировать (по 12ч)
    """

    num_points = len(price_sequence)
    time_index = pd.date_range(
        end=last_timestamp,
        periods=num_points,
        freq=frequency
    )

    df = pd.DataFrame({
        "ds": time_index,
        "y": price_sequence
    })

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_length, freq=frequency)
    forecast = model.predict(future)

    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='Исторические данные', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Прогноз', color='green')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                     color='lightgreen', alpha=0.5, label='Доверительный интервал')
    plt.axvline(x=df['ds'].max(), color='gray', linestyle='--', label='Начало прогноза')
    plt.title("Прогноз цен")
    plt.xlabel("Дата")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PREDICTION_GRAPH.png")

    return forecast['ds'].tolist(), forecast['yhat'].tolist(), forecast['yhat_lower'].tolist(), forecast['yhat_upper'].tolist()