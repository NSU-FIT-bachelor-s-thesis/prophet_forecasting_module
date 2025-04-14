import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("interval_metrics.csv")

df['price_center_rub'] = ((df['min_price'] + df['max_price']) / 2) / 100

os.makedirs("graphs", exist_ok=True)

metrics = {
    "mean_MAE": ("Средний MAE", "₽"),
    "mean_RMSE": ("Средний RMSE", "₽"),
    "mean_MAPE": ("Средний MAPE", "%"),
    "mean_MDAE": ("Медианная абсолютная ошибка", "₽"),
    "mean_MDAPE": ("Медианная абсолютная процентная ошибка", "%")
}

for column, (title, unit) in metrics.items():
    plt.figure(figsize=(10, 5))
    plt.plot(df['price_center_rub'], df[column], marker='o')
    plt.title(f"{title} по ценовым категориям")
    plt.xlabel("Ценовой диапазон (руб.)")
    plt.ylabel(f"{title} ({unit})")
    plt.grid(True)

    filename = f"graphs/{column}_by_price.png"
    plt.savefig(filename)
    plt.close()

print("Графики сохранены в папке graphs/")
