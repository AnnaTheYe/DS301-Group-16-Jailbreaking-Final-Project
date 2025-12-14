# ReadMe
# NYU DS-UA 301 — Group 16 Final Project  
## GuardLLM: Evaluating LLM Robustness to Jailbreak Attacks

This repository contains the final project for **NYU DS-UA 301 (Advanced Topics in Data Science: LLMs and Deep Learning)** by **Group 16**. Our project, **GuardLLM**, is a unified evaluation framework designed to test the robustness of large language models (LLMs) against jailbreak attacks. GuardLLM supports both single-turn and multi-turn adversarial prompts and enables systematic analysis of how different pipeline components—such as smoothing, templating, transcript memory, and judge-based evaluation—affect model safety behavior. In addition to offline notebook-based analysis, this repository includes a **Streamlit application** that allows interactive testing of jailbreak prompts and inspection of model responses.

---
## Group Members 
Diya Agrawal, Anna Ye, Sophia Ugaz 


## Motivation

Large language models are increasingly deployed in real-world systems, yet they remain vulnerable to adversarial prompting techniques that bypass safety constraints. Many existing evaluations focus on isolated, single-turn prompts, while real jailbreak attacks often unfold gradually across multiple conversational turns. This project investigates both explicit and incremental jailbreak strategies, compares safety behavior across different pipeline configurations, and studies how multi-turn context accumulation influences alignment outcomes.

---

## Key Features

- Single-turn and multi-turn jailbreak evaluation  
- Modular GuardLLM pipeline architecture  
- Prompt smoothing and safety-aware templating  
- Transcript-level memory for multi-turn attacks  
- Judge-based output evaluation  
- Ablation study support  
- Interactive Streamlit app for testing and demonstration  

---

## Project Structure

- `guard_llm_full_pipeline_with_cycles.ipynb:` Main notebook including dataset handling, LangGraph pipeline, and code for running experiments.
- `guard_llm.py`: Python file containing only LangGraph pipeline.
- `streamlit_app.py`: Python file for running Streamlit app.
- `FINAL_GuardLLM_Ablation_Analysis.ipynb`: Notebook for result analyses used for Milestones 2+3 and final presentaiton.
- `Dataset Visualization.ipynb`: Exploratory dataset analysis for Milestone 1.


---

## Running the Streamlit Demo

1. Clone repository
```
git clone https://github.com/AnnaTheYe/DS301-Group-16-Jailbreaking-Final-Project.git
cd DS301-Group-16-Jailbreaking-Final-Project
```
2. Install requirements
```
pip install -r requirements.txt
```
3. Run via Streamlit
```
streamlit run streamlit_app.py
```


---

## Running Experiments

1. Clone repository
```
git clone https://github.com/AnnaTheYe/DS301-Group-16-Jailbreaking-Final-Project.git
cd DS301-Group-16-Jailbreaking-Final-Project
```
2. In `guard_llm_full_pipeline_with_cycles.ipynb`, navigate to the bottommost cell and choose configurations
```
# Run ablation experiments
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import numpy as np

token_max = # ENTER YOUR CONFIGURATION
test_size = # ENTER YOUR CONFIGURATION
N = # ENTER YOUR CONFIGURATION
cycles = # ENTER YOUR CONFIGURATION

kf = KFold(n_splits=cycles, shuffle=True, random_state=42)
splits = list(kf.split(df)) # split dataframe for processing

os.makedirs("results", exist_ok=True)

for i, (train_idx, test_idx) in enumerate(splits):
    df_cycle = df.iloc[test_idx]   # unique chunk for this cycle
    for exp_name, recipe in experiment_configurations.items():
        for model in providers:
            print(f"\n===== RUNNING EXPERIMENT: {exp_name} | MODEL: {model} =====")

            state_config = {
                "name": f"{exp_name}__{model.replace(':','')}",
                "model": model,
                "token_max": token_max,
                "smooth": recipe["smooth"],
                "judges": recipe["judges"],
                "judge_threshold": recipe["judge_threshold"],
                "template": recipe["template"]
            }

            config_str = f"experiment-{state_config['name']}_model-{state_config['model']}_{i+1}-of-{cycles}"
            output_file = f'results/{config_str}.csv'

            # updated call — takes config_str, not output_file
            try:
                # run experiment
                run_single_experiment(state_config, df_cycle, config_str, test_size/cycles, int(N/cycles))
            except Exception as e:
                print(f"⚠️ Experiment {config_str} failed: {e}")
                continue  # continue to the next experiment
            
            
```
3. Run entire notebook
