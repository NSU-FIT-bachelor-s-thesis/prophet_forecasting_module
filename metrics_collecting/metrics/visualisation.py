import pandas as pd
import matplotlib.pyplot as plt
import os

output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("metrics_for_every_product.csv")

total_products = len(df)

percent_metrics = ["MAPE", "MDAPE"]
absolute_metrics = ["MAE", "RMSE", "MDAE"]

percent_bins = [0, 10, 20, 30, 40, float('inf')]
percent_labels = ["<10%", "10-20%", "20-30%", "30-40%", ">40%"]

absolute_bins = [0, 500, 1000, 5000, 10000, 50000, float('inf')]
absolute_labels = ["<500", "500–1000", "1k–5k", "5k–10k", "10k–50k", ">50k"]

def plot_histogram(metric, bins, labels, is_percent=False):
    df[f'{metric}_range'] = pd.cut(df[metric], bins=bins, labels=labels, right=False)
    counts = df[f'{metric}_range'].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    ax = counts.plot(kind='bar', color='cornflowerblue' if is_percent else 'salmon', edgecolor='black')
    plt.title(f'Распределение продуктов по {metric}')
    plt.xlabel('Интервал ошибки' + (' (%)' if is_percent else ' (₽)'))
    plt.ylabel('Количество продуктов')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    for i, count in enumerate(counts):
        percent = count / total_products * 100
        ax.text(i, count + max(counts)*0.01, f"{percent:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.savefig(os.path.join(output_dir, f"{metric}_hist.svg"), format='svg')
    plt.close()

for metric in percent_metrics:
    plot_histogram(metric, percent_bins, percent_labels, is_percent=True)

for metric in absolute_metrics:
    plot_histogram(metric, absolute_bins, absolute_labels, is_percent=False)
