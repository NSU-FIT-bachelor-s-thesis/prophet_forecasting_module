import pandas as pd
import matplotlib.pyplot as plt
import os

output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("metrics.csv")
total_products = len(df)

metric = "MAPE"
bins = [0, 10, 20, 30, 40, float('inf')]
labels = ["<10%", "10-20%", "20-30%", "30-40%", ">40%"]

df[f'{metric}_range'] = pd.cut(df[metric], bins=bins, labels=labels, right=False)
counts = df[f'{metric}_range'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
ax = counts.plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title(f'Распределение продуктов по {metric} (всего: {total_products})')
plt.xlabel('Интервал ошибки (%)')
plt.ylabel('Количество продуктов')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

for i, count in enumerate(counts):
    percent = count / total_products * 100
    ax.text(i, count + max(counts)*0.01, f"{percent:.1f}%", ha='center', va='bottom', fontsize=9)

plt.savefig(os.path.join(output_dir, f"{metric}.svg"), format='svg')
plt.close()
