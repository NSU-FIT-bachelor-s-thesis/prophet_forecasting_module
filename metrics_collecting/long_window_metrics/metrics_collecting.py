import pandas as pd
from prophet import Prophet
import psycopg2
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import numpy as np
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

    # Получаем среднюю цену товара
    cursor.execute("""
        SELECT AVG(product_price)
        FROM product_statistics
        WHERE product_id = %s AND product_price > 0
    """, (product_id,))
    avg_price = cursor.fetchone()[0]

    return {
        'product_id': product_id,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MDAE': mdae,
        'MDAPE': mdape,
        'avg_price': avg_price  # Добавляем среднюю цену товара
    }


def main():
    conn = psycopg2.connect(
        dbname="db",
        user="user",
        password="password",
        host="db_service",
        port="5432"
    )
    cursor = conn.cursor()

    output_file = "metrics_for_every_product.csv"

    pd.DataFrame(columns=[
        "product_id", "MAE", "RMSE", "MAPE", "MDAE", "MDAPE", "avg_price"
    ]).to_csv(output_file, index=False)

    # if not os.path.exists(output_file):
    #     pd.DataFrame(columns=[
    #         "product_id", "MAE", "RMSE", "MAPE", "MDAE", "MDAPE", "avg_price"
    #     ]).to_csv(output_file, index=False)

    # Получаем список всех product_id
    cursor.execute("""
        SELECT DISTINCT product_id
        FROM product_statistics
        WHERE product_price > 0
    """)
    product_ids = cursor.fetchall()

    # Для каждого товара рассчитываем метрики и сохраняем в файл
    counter = 0
    for product_id_tuple in product_ids:
        product_id = product_id_tuple[0]
        counter = counter + 1
        logging.info(f"Анализ товара номер {counter}...")

        metrics = run_for_product(product_id, cursor)
        if metrics:
            df = pd.DataFrame([metrics])
            df.to_csv(output_file, mode='a', index=False, header=False)

    logging.info(f"\nМетрики сохранены в {output_file}")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
