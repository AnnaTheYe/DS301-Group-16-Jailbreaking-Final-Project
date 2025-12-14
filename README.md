# ReadMe
# NYU DS-UA 301 — Group 16 Final Project  
## GuardLLM: Evaluating LLM Robustness to Jailbreak Attacks

This repository contains the final project for **NYU DS-UA 301 (Social Networking)** by **Group 16**. Our project, **GuardLLM**, is a unified evaluation framework designed to test the robustness of large language models (LLMs) against jailbreak attacks. GuardLLM supports both single-turn and multi-turn adversarial prompts and enables systematic analysis of how different pipeline components—such as smoothing, templating, transcript memory, and judge-based evaluation—affect model safety behavior. In addition to offline notebook-based analysis, this repository includes a **Streamlit application** that allows interactive testing of jailbreak prompts and inspection of model responses.

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

- guard_llm_full_pipeline_with_cycles.ipynb: Main notebook including dataset handling, LangGraph pipeline, and code for running experiments.
- guard_llm.py: Python file containing only LangGraph pipeline.
- streamlit_app.py: Python file for running Streamlit app.
- FINAL_GuardLLM_Ablation_Analysis.ipynb: Notebook for result analyses used for Milestones 2+3 and final presentaiton.
- Dataset Visualization.ipynb: Exploratory dataset analysis for Milestone 1.


---

## Installation
