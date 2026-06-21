# Empirical Evaluation of Quantized LLMs Across Diverse NLP Tasks Under Resource-Constrained Inference

## Overview

This project presents a comprehensive empirical evaluation of quantized Large Language Models (LLMs) under resource-constrained inference environments.

The study investigates how model quantization affects:

- Inference latency
- Throughput
- Model storage requirements
- Task-specific performance

Experiments were conducted using GGUF-quantized models executed through **llama.cpp** on consumer-grade hardware. The objective is to identify practical trade-offs between model efficiency and performance for local deployment scenarios.

---

## Motivation

Large Language Models (LLMs) have achieved remarkable performance across a wide range of Natural Language Processing (NLP) tasks. However, deploying these models often requires substantial computational resources, memory capacity, and storage.

Quantization has emerged as an effective technique for reducing the resource requirements of LLMs by representing model weights with lower numerical precision. While quantization significantly improves efficiency, it may also impact model quality.

This project systematically evaluates the trade-offs between:

- Model quality
- Inference speed
- Latency
- Storage footprint

across multiple model families, quantization levels, and NLP benchmarks.

---

## Research Objectives

The primary objectives of this work are:

1. Evaluate the impact of quantization on model performance.
2. Measure latency and throughput across different quantization levels.
3. Analyze storage savings achieved through quantization.
4. Compare the behavior of different LLM families under identical evaluation settings.
5. Identify optimal quantization strategies for resource-constrained deployment environments.

---

## Models Evaluated

### TinyLlama

**Model:** TinyLlama-1.1B-Chat-v1.0

Hugging Face:

https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

Approximate Parameters:

- 1.1 Billion

Quantizations Evaluated:

- Q2
- Q4
- Q6
- Q8

---

### Phi-2

**Model:** Phi-2

Hugging Face:

https://huggingface.co/microsoft/phi-2

Approximate Parameters:

- 2.7 Billion

Quantizations Evaluated:

- Q2
- Q4
- Q6
- Q8

---

### Phi-3 Mini

**Model:** Phi-3-mini-4k-instruct

GGUF Version:

https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF

Approximate Parameters:

- 3.8 Billion

Quantizations Evaluated:

- Q4
- Q6
- Q8

---

### Qwen2-7B-Instruct

**Model:** Qwen2-7B-Instruct

GGUF Version:

https://huggingface.co/bartowski/Qwen2-7B-Instruct-GGUF

Approximate Parameters:

- 7 Billion

Quantizations Evaluated:

- Q2
- Q4

---

## Datasets

### GSM8K

Task:

Mathematical Reasoning

Dataset:

[https://huggingface.co/datasets/gsm8k](https://www.kaggle.com/datasets/johnsonhk88/gsm8k-grade-school-math-8k-dataset-for-llm)

Paper:

https://arxiv.org/abs/2110.14168

---

### MBPP

Task:

Code Generation

Dataset:

[https://huggingface.co/datasets/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)

Paper:

https://arxiv.org/abs/2108.07732

---

### CNN/DailyMail

Task:

Abstractive Summarization

Dataset:

[https://huggingface.co/datasets/cnn_dailymail](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

Paper:

https://arxiv.org/abs/1506.03340

---

### TriviaQA

Task:

Question Answering

Dataset:

[https://huggingface.co/datasets/trivia_qa
](https://nlp.cs.washington.edu/triviaqa/)

Paper:

https://arxiv.org/abs/1705.03551

---

## Experimental Setup

### Hardware

- Apple Silicon MacBook
- 16 GB Unified Memory

### Software

- Python 3.11
- llama-cpp-python
- Hugging Face Datasets
- Evaluate
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Inference Configuration

```python
n_ctx = 2048
temperature = 0
max_tokens = 120
n_threads = 8
n_gpu_layers = -1
```

### Evaluation Protocol

- 30 samples per dataset
- 2 independent experimental runs
- Consistent prompt templates across all models
- Identical inference settings for all evaluations

---

## Evaluation Metrics

### GSM8K

Metric:

- Accuracy

### MBPP

Metric:

- Code Validity Score

### CNN/DailyMail

Metric:

- ROUGE-L

### TriviaQA

Metric:

- Answer Accuracy

### System-Level Metrics

- Average Latency
- Tokens per Second
- Model Size (GB)

---

## Experimental Scope

Models:

- TinyLlama
- Phi-2
- Phi-3 Mini
- Qwen2-7B

Datasets:

- GSM8K
- MBPP
- CNN/DailyMail
- TriviaQA

Quantization Levels:

- Q2
- Q4
- Q6
- Q8

Runs:

- 2

### Total Evaluations

13 model-quantization configurations × 4 datasets × 2 runs

= **104 individual experiments**

---

## Repository Structure

```text
llm_quantization_project/
│
├── models/
│   ├── *.gguf
│
├── results/
│   ├── quantization_results.csv
│   ├── clean_results.json
│   └── full_results.json
│
├── plots/
│   ├── latency_vs_quantization.png
│   ├── throughput_vs_quantization.png
│   ├── quality_vs_quantization.png
│   ├── quality_vs_speed_tradeoff.png
│   ├── model_size_vs_quantization.png
│   ├── latency_vs_model_size.png
│   ├── quality_vs_model_size.png
│   ├── speed_vs_model_size.png
│   ├── dataset_performance.png
│   ├── quantization_sensitivity.png
│   ├── pareto_quality_latency.png
│   ├── radar_chart.png
│   ├── quality_heatmap.png
│   └── dataset_difficulty.png
│
├── run_experiments.py
├── generate_plots.py
├── requirements.txt
└── README.md
```

---

## Visualizations Generated

The project includes the following analyses:

1. Latency vs Quantization
2. Throughput vs Quantization
3. Quality vs Quantization
4. Quality vs Speed Trade-off
5. Model Size vs Quantization
6. Latency vs Model Size
7. Quality vs Model Size
8. Speed vs Model Size
9. Dataset Performance Comparison
10. Quantization Sensitivity Analysis
11. Pareto Frontier Analysis
12. Radar Chart Comparison
13. Quality Heatmap
14. Dataset Difficulty Analysis

---

## Key Findings

### Quantization Effectiveness

- Q4 generally provided the best balance between quality and efficiency.
- Q2 achieved the smallest model sizes but occasionally reduced task performance.

### Throughput

- TinyLlama consistently achieved the highest throughput.
- Larger models showed reduced inference speed but improved task quality.

### Model Quality

- Qwen2-7B achieved the strongest overall quality scores.
- Phi-3 Mini demonstrated strong performance while requiring significantly fewer resources than Qwen2-7B.
- Phi-2 provided a balanced compromise between efficiency and quality.

### Dataset Difficulty

- GSM8K was the most challenging benchmark.
- MBPP achieved the highest performance scores.
- CNN/DailyMail showed relatively stable behavior across quantization levels.
- TriviaQA benefited from larger model sizes.

### Resource-Constrained Deployment

Results indicate that quantized LLMs can provide useful performance on consumer-grade hardware without requiring dedicated GPUs.

---

## Running the Experiments

### Clone Repository

```bash
git clone <repository_url>
cd llm_quantization_project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Experiments

```bash
python run_experiments.py
```

### Generate Plots

```bash
python generate_plots.py
```

---

## Results Files

### quantization_results.csv

Contains:

- Model
- Quantization Level
- Dataset
- Latency
- Throughput
- Model Size
- Quality Metrics

### clean_results.json

Processed experimental results for analysis.

### full_results.json

Complete experimental outputs including generated model responses.

---

## Future Work

Potential extensions include:

- Evaluation of larger models (13B+)
- Additional quantization methods (GPTQ, AWQ, SmoothQuant)
- Energy consumption measurements
- Memory utilization analysis
- Benchmarking on edge devices and mobile platforms
- Human evaluation of generated outputs
