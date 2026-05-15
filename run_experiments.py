import time
import os
import re
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import evaluate
from llama_cpp import Llama

# --------------------------------
# Model paths
# --------------------------------

models = {
    "tinyllama": {
        "q8": "models/tinyllama-q8.gguf",
        "q6": "models/tinyllama-q6.gguf",
        "q4": "models/tinyllama-q4.gguf",
        "q2": "models/tinyllama-q2.gguf"
    },
    "phi2": {
        "q8": "models/phi2-q8.gguf",
        "q6": "models/phi2-q6.gguf",
        "q4": "models/phi2-q4.gguf",
        "q2": "models/phi2-q2.gguf"
    },
    "phi3mini": {
        "q8": "models/phi3mini-q8.gguf",
        "q6": "models/phi3mini-q6.gguf",
        "q4": "models/phi3mini-q4.gguf"
    },
    "qwen2_7b": {
        "q4": "models/qwen2-7b-q4.gguf",
        "q2": "models/qwen2-7b-q2.gguf"
    }
}

# --------------------------------
# Load datasets
# --------------------------------

gsm8k = load_dataset("gsm8k","main",split="test").select(range(30))
mbpp = load_dataset("mbpp",split="test").select(range(30))
cnn = load_dataset("cnn_dailymail","3.0.0",split="test").select(range(30))
trivia = load_dataset("trivia_qa","rc",split="validation").select(range(30))

datasets_dict = {
    "gsm8k": gsm8k,
    "mbpp": mbpp,
    "cnn_dm": cnn,
    "triviaqa": trivia
}

# --------------------------------
# Evaluation tools
# --------------------------------

rouge = evaluate.load("rouge")

# --------------------------------
# Load existing results
# --------------------------------

results = []
existing_experiments = set()

csv_path = "results/quantization_results.csv"
json_path = "results/full_results.json"

if os.path.exists(csv_path):

    old_df = pd.read_csv(csv_path)

    for _, row in old_df.iterrows():

        key = (
            row["model"],
            row["quantization"],
            row["dataset"],
            row["run"]
        )

        existing_experiments.add(key)

    results = old_df.to_dict("records")

    print(f"Loaded {len(results)} previous experiments.")

# --------------------------------
# Prompt builder
# --------------------------------

def build_prompt(dataset_name, sample):

    if dataset_name == "gsm8k":
        return f"Question: {sample['question']}\nAnswer:"

    elif dataset_name == "mbpp":
        return f"Write Python code for the following task:\n{sample['text']}\nCode:"

    elif dataset_name == "cnn_dm":
        article = sample["article"][:1500]
        return f"Summarize the following article:\n{article}\nSummary:"

    elif dataset_name == "triviaqa":
        return f"Question: {sample['question']}\nAnswer:"

# --------------------------------
# Run model inference
# --------------------------------

def run_prompt(llm, prompt):

    start = time.time()

    output = llm(
        prompt,
        max_tokens=120,
        temperature=0
    )

    latency = time.time() - start

    generated = output["choices"][0]["text"].strip()

    return generated, latency

# --------------------------------
# Metric helpers
# --------------------------------

def gsm8k_accuracy(outputs, refs):

    correct = 0

    for o, r in zip(outputs, refs):

        true = re.findall(r'\d+', r["answer"])[-1]
        pred_nums = re.findall(r'\d+', o)

        if len(pred_nums) > 0 and pred_nums[-1] == true:
            correct += 1

    return correct / len(outputs)

def trivia_accuracy(outputs, refs):

    correct = 0

    for o, r in zip(outputs, refs):

        answer = r["answer"]["value"].lower()
        pred = o.lower()

        if answer in pred or answer.split()[0] in pred:
            correct += 1

    return correct / len(outputs)

def mbpp_score(outputs):

    valid = 0

    for o in outputs:

        code = o.lower()

        if "def " in code or "return " in code:
            valid += 1

    return valid / len(outputs)

def rouge_score(outputs, refs):

    preds = [o if len(o.strip()) > 0 else "empty" for o in outputs]
    references = [r["highlights"] for r in refs]

    scores = rouge.compute(predictions=preds, references=references)

    return scores["rougeL"]

# --------------------------------
# Experiment loop
# --------------------------------

runs = 2

for run in range(runs):

    print(f"\nRUN {run}\n")

    for model_name, variants in models.items():

        for quant_name, model_path in variants.items():

            model_size = os.path.getsize(model_path) / (1024**3)

            print(f"Preparing {model_name} {quant_name}")

            llm = None

            for dataset_name, dataset in datasets_dict.items():

                key = (model_name, quant_name, dataset_name, run)

                if key in existing_experiments:
                    print(f"Skipping completed experiment: {key}")
                    continue

                if llm is None:
                    print(f"Loading {model_name} {quant_name}")
                    llm = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_threads=8,
                        n_gpu_layers=-1,
                        verbose=False
                    )

                latencies = []
                outputs = []
                refs = []

                print(f"Dataset: {dataset_name}")

                for sample in tqdm(dataset):

                    prompt = build_prompt(dataset_name, sample)

                    output, latency = run_prompt(llm, prompt)

                    if len(output.strip()) == 0:
                        output = "EMPTY"

                    latencies.append(latency)
                    outputs.append(output)
                    refs.append(sample)

                avg_latency = np.mean(latencies)

                if dataset_name == "gsm8k":
                    metric_val = gsm8k_accuracy(outputs, refs)

                elif dataset_name == "cnn_dm":
                    metric_val = rouge_score(outputs, refs)

                elif dataset_name == "mbpp":
                    metric_val = mbpp_score(outputs)

                elif dataset_name == "triviaqa":
                    metric_val = trivia_accuracy(outputs, refs)
                
                avg_tokens = np.mean([len(o.split()) for o in outputs])
                tokens_per_sec = avg_tokens / avg_latency

                results.append({
                    "run": run,
                    "model": model_name,
                    "quantization": quant_name,
                    "dataset": dataset_name,
                    "latency": avg_latency,
                    "tokens_per_sec": tokens_per_sec,
                    "model_size_gb": model_size,
                    "quality_metric": metric_val,
                    "outputs": outputs,
                    "references": refs
                })

                existing_experiments.add(key)

            if llm is not None:
                del llm

# --------------------------------
# Save results
# --------------------------------

df = pd.DataFrame(results)

df.to_csv(csv_path, index=False)

with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("\nExperiment complete.\n")
print(df)
