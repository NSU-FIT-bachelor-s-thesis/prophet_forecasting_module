import pandas as pd
import matplotlib.pyplot as plt
import os

output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("metrics.csv")

total_products = len(df)

# ценовые диапазоны в копейках
price_bins = [0, 100000, 200000, 300000, 400000, 500000, 1000000, 2000000, 3000000, 4000000, 5000000, float('inf')]

price_labels = [
    "<1k", "1k–2k", "2k–3k", "3k–4k", "4k–5k",
    "5k–10k", "10k–20k", "20k–30k", "30k–40k", "40k–50k", ">50k"
]

# Укажи правильное имя столбца, если отличается
price_column = 'avg_price'  # <-- замени на актуальное, если нужно

def plot_price_histogram():
    if price_column not in df.columns:
        raise ValueError(f"Столбец '{price_column}' не найден в данных!")

    df['price_range'] = pd.cut(df[price_column], bins=price_bins, labels=price_labels, right=False)
    counts = df['price_range'].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    ax = counts.plot(kind='bar', color='mediumseagreen', edgecolor='black')
    plt.title('Распределение продуктов по ценовым категориям')
    plt.xlabel('Диапазон цены (₽)')
    plt.ylabel('Количество продуктов')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    for i, count in enumerate(counts):
        percent = count / total_products * 100
        ax.text(i, count + max(counts) * 0.01, f"{percent:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.savefig(os.path.join(output_dir, "price_hist.svg"), format='svg')
    plt.close()

plot_price_histogram()
