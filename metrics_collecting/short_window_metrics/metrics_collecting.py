import pandas as pd
from prophet import Prophet
import psycopg2
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import numpy as np
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fastapi_service.log'),
        logging.StreamHandler()
    ]
)


def normalize_time_series(df, time_column='measurement_time', value_column='product_price'):
    df_sorted = df.sort_values(by=time_column).reset_index(drop=True)
    start_time = pd.to_datetime(df_sorted[time_column].iloc[0], unit='s')
    time_index = [start_time + pd.Timedelta(hours=12) * i for i in range(len(df_sorted))]

    return pd.DataFrame({
        'ds': time_index,
        'y': df_sorted[value_column].values
    })


def median_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ape = np.abs((y_true - y_pred) / y_true) * 100
    return np.median(ape)


def run_for_product(product_id, cursor):
    cursor.execute("""
        SELECT measurement_time, product_price
        FROM product_statistics
        WHERE product_id = %s AND product_price > 0
        ORDER BY measurement_time ASC
    """, (product_id,))
    rows = cursor.fetchall()
    if len(rows) < 20:
        return None

    df = pd.DataFrame(rows, columns=['measurement_time', 'product_price'])
    df_prophet = normalize_time_series(df)

    window_size = 10
    forecast_horizon = 2
    step = 1

    maes, rmses, mapes, mdaes, mdapes = [], [], [], [], []
    successful_windows = 0
    failed_windows = 0

    for start in range(0, len(df_prophet) - window_size + 1, step):
        window = df_prophet.iloc[start:start + window_size].reset_index(drop=True)
        train = window.iloc[:window_size - forecast_horizon]
        test = window.iloc[window_size - forecast_horizon:]

        if len(train) < 2 or len(test) < forecast_horizon:
            # failed_windows += 1
            continue

        try:
            model = Prophet()
            model.fit(train)

            forecast = model.predict(test[['ds']])
            y_true = test['y'].values
            y_pred = forecast['yhat'].values

            maes.append(mean_absolute_error(y_true, y_pred))
            rmses.append(math.sqrt(mean_squared_error(y_true, y_pred)))
            mapes.append(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            mdaes.append(median_absolute_error(y_true, y_pred))
            mdapes.append(median_absolute_percentage_error(y_true, y_pred))

            successful_windows += 1

        except Exception as e:
            logging.error(f"Ошибка при обработке окна для продукта {product_id}: {e}")
            failed_windows += 1
            continue

    if successful_windows == 0:
        return None

    cursor.execute("""
        SELECT AVG(product_price)
        FROM product_statistics
        WHERE product_id = %s AND product_price > 0
    """, (product_id,))
    avg_price = cursor.fetchone()[0]

    return {
        'product_id': product_id,
        'MAE': np.mean(maes),
        'RMSE': np.mean(rmses),
        'MAPE': np.mean(mapes),
        'MDAE': np.mean(mdaes),
        'MDAPE': np.mean(mdapes),
        'avg_price': avg_price,
        'successful_windows': successful_windows,
        'failed_windows': failed_windows
    }


def main():
    conn = psycopg2.connect(
        dbname="db",
        user="user",
        password="password",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    output_file = "metrics.csv"

    if not os.path.exists(output_file):
        pd.DataFrame(columns=[
            "product_id", "MAE", "RMSE", "MAPE", "MDAE", "MDAPE",
            "avg_price", "successful_windows", "failed_windows"
        ]).to_csv(output_file, index=False)

    cursor.execute("""
        SELECT DISTINCT product_id
        FROM product_statistics
        WHERE product_price > 0
    """)
    product_ids = cursor.fetchall()

    counter = 0
    for product_id_tuple in product_ids:
        product_id = product_id_tuple[0]
        counter += 1
        logging.info(f"Анализ товара номер {counter} (ID: {product_id})...")

        metrics = run_for_product(product_id, cursor)
        if metrics:
            df = pd.DataFrame([metrics])
            df.to_csv(output_file, mode='a', index=False, header=False)

    logging.info(f"\nМетрики сохранены в {output_file}")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
