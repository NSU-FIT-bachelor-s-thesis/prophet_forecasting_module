import pandas as pd
from prophet import Prophet
import psycopg2
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import numpy as np
import os


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


def run_for_price_interval(min_price, max_price, cursor):
    cursor.execute("""
        SELECT product_id, AVG(product_price) as avg_price
        FROM product_statistics
        WHERE product_price > 0
        GROUP BY product_id
    """)
    product_price_info = cursor.fetchall()

    product_ids = [pid for pid, avg_price in product_price_info if min_price <= avg_price <= max_price]

    results = []

    for product_id in product_ids:
        cursor.execute("""
            SELECT measurement_time, product_price
            FROM product_statistics
            WHERE product_id = %s AND product_price > 0
            ORDER BY measurement_time ASC
        """, (product_id,))
        rows = cursor.fetchall()
        if len(rows) < 20:
            continue

        df = pd.DataFrame(rows, columns=['measurement_time', 'product_price'])
        df_prophet = normalize_time_series(df)

        split_idx = int(len(df_prophet) * 0.8)
        train_df = df_prophet.iloc[:split_idx]
        test_df = df_prophet.iloc[split_idx:]

        model = Prophet()
        model.fit(train_df)

        forecast = model.predict(test_df[['ds']])
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mdae = median_absolute_error(y_true, y_pred)
        mdape = median_absolute_percentage_error(y_true, y_pred)

        results.append((mae, rmse, mape, mdae, mdape))

    if results:
        df = pd.DataFrame(results, columns=['MAE', 'RMSE', 'MAPE', 'MDAE', 'MDAPE'])
        return {
            'min_price': min_price,
            'max_price': max_price,
            'count': len(results),
            'mean_MAE': df["MAE"].mean(),
            'mean_RMSE': df["RMSE"].mean(),
            'mean_MAPE': df["MAPE"].mean(),
            'mean_MDAE': df["MDAE"].mean(),
            'mean_MDAPE': df["MDAPE"].mean()
        }
    else:
        return {
            'min_price': min_price,
            'max_price': max_price,
            'count': 0,
            'mean_MAE': None,
            'mean_RMSE': None,
            'mean_MAPE': None,
            'mean_MDAE': None,
            'mean_MDAPE': None
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

    output_file = "interval_metrics.csv"

    # создаём файл с заголовком, если его нет
    if not os.path.exists(output_file):
        pd.DataFrame(columns=[
            "min_price", "max_price", "count",
            "mean_MAE", "mean_RMSE", "mean_MAPE",
            "mean_MDAE", "mean_MDAPE"
        ]).to_csv(output_file, index=False)

    for min_price in range(0, 100000, 100):
        max_price = min_price + 100
        print(f"Анализ товаров со средней ценой от {min_price} до {max_price} руб...")

        metrics = run_for_price_interval(min_price * 100, max_price * 100, cursor)

        df = pd.DataFrame([metrics])
        df.to_csv(output_file, mode='a', index=False, header=False)

    print(f"\nГотово! Метрики сохранены в {output_file}")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
