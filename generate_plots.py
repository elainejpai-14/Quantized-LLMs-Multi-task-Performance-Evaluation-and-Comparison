import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Setup
# -----------------------------

sns.set(style="whitegrid")

data_path = "results/clean_results.csv"
df = pd.read_csv(data_path)

plots_dir = "results/plots"
os.makedirs(plots_dir, exist_ok=True)

# -----------------------------
# 1 Latency vs Quantization
# -----------------------------

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="quantization", y="latency", hue="model")
plt.title("Latency vs Quantization")
plt.savefig(f"{plots_dir}/1_latency_vs_quantization.png")
plt.close()

# -----------------------------
# 2 Throughput vs Quantization
# -----------------------------

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="quantization", y="tokens_per_sec", hue="model")
plt.title("Throughput vs Quantization")
plt.savefig(f"{plots_dir}/2_throughput_vs_quantization.png")
plt.close()

# -----------------------------
# 3 Quality vs Quantization
# -----------------------------

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="quantization", y="metric_value", hue="model")
plt.title("Quality vs Quantization")
plt.savefig(f"{plots_dir}/3_quality_vs_quantization.png")
plt.close()

# -----------------------------
# 4 Quality vs Speed Tradeoff
# -----------------------------

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="tokens_per_sec",
    y="metric_value",
    hue="model",
    style="quantization",
    s=120
)
plt.title("Quality vs Speed Tradeoff")
plt.savefig(f"{plots_dir}/4_quality_vs_speed_tradeoff.png")
plt.close()

# -----------------------------
# 5 Model Size vs Quantization
# -----------------------------

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="quantization", y="model_size_gb", hue="model")
plt.title("Model Size vs Quantization")
plt.savefig(f"{plots_dir}/5_model_size_vs_quantization.png")
plt.close()

# -----------------------------
# 6 Latency vs Model Size
# -----------------------------

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="model_size_gb",
    y="latency",
    hue="model",
    style="quantization",
    s=120
)
plt.title("Latency vs Model Size")
plt.savefig(f"{plots_dir}/6_latency_vs_model_size.png")
plt.close()

# -----------------------------
# 7 Quality vs Model Size
# -----------------------------

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="model_size_gb",
    y="metric_value",
    hue="model",
    style="quantization",
    s=120
)
plt.title("Quality vs Model Size")
plt.savefig(f"{plots_dir}/7_quality_vs_model_size.png")
plt.close()

# -----------------------------
# 8 Speed vs Model Size
# -----------------------------

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="model_size_gb",
    y="tokens_per_sec",
    hue="model",
    style="quantization",
    s=120
)
plt.title("Speed vs Model Size")
plt.savefig(f"{plots_dir}/8_speed_vs_model_size.png")
plt.close()

# -----------------------------
# 9 Dataset Specific Performance
# -----------------------------

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="dataset", y="metric_value", hue="model")
plt.title("Performance by Dataset")
plt.savefig(f"{plots_dir}/9_dataset_performance.png")
plt.close()

# -----------------------------
# 10 Quantization Sensitivity
# -----------------------------

plt.figure(figsize=(10,6))
sns.lineplot(
    data=df,
    x="quantization",
    y="metric_value",
    hue="model",
    marker="o"
)
plt.title("Quantization Sensitivity")
plt.savefig(f"{plots_dir}/10_quantization_sensitivity.png")
plt.close()

# -----------------------------
# 11 Pareto Frontier (Quality vs Latency)
# -----------------------------

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="latency",
    y="metric_value",
    hue="model",
    style="quantization",
    s=120
)
plt.title("Pareto Frontier: Quality vs Latency")
plt.savefig(f"{plots_dir}/11_pareto_quality_latency.png")
plt.close()

# -----------------------------
# 12 Radar Chart
# -----------------------------

import numpy as np

radar_df = df.groupby("model")[["latency","tokens_per_sec","metric_value","model_size_gb"]].mean()

labels = radar_df.columns
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, polar=True)

for model in radar_df.index:
    values = radar_df.loc[model].values
    values = np.concatenate((values,[values[0]]))
    angle = np.concatenate((angles,[angles[0]]))
    ax.plot(angle, values, label=model)

ax.set_xticks(angles)
ax.set_xticklabels(labels)
plt.legend(loc="upper right")
plt.title("Model Comparison Radar")
plt.savefig(f"{plots_dir}/12_radar_chart.png")
plt.close()

# -----------------------------
# 13 Heatmap of Quality
# -----------------------------

pivot = df.pivot_table(
    values="metric_value",
    index="model",
    columns="quantization",
    aggfunc="mean"
)

plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, cmap="viridis")
plt.title("Quality Heatmap")
plt.savefig(f"{plots_dir}/13_quality_heatmap.png")
plt.close()

# -----------------------------
# 14 Dataset Difficulty Comparison
# -----------------------------

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="dataset", y="metric_value")
plt.title("Dataset Difficulty Comparison")
plt.savefig(f"{plots_dir}/14_dataset_difficulty.png")
plt.close()

print("All 14 plots generated in results/plots/")
