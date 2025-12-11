import streamlit as st
import pandas as pd
import json
import streamlit as st
from guard_llm import pipeline_graph
from guard_llm import experiment_configurations, providers

import __main__

pipeline_graph = __main__.pipeline_graph
experiment_configurations = __main__.experiment_configurations
providers = __main__.providers
model_titles = {
    "Gemini Flash 2.5": "google:",
    "ChatGPT 4o Mini": "openai:",
    "Claude Haiku 3.0": "anthropic:"
}

# Title
st.title("🔓 GuardLLM Testing Interface")

# Sidebar Controls
st.sidebar.header("Configuration")

# Choose provider
model_choice = st.sidebar.selectbox(
    "Choose Model",
    list(model_titles.keys())
)

# Choose experiment
experiment_choice = st.sidebar.selectbox(
    "Choose Experiment Configuration",
    list(experiment_configurations.keys())
)

recipe = experiment_configurations[experiment_choice]

# Prompt input
user_prompt = st.text_area(
    "Enter your prompt to test jailbreak:",
    height=200,
    placeholder="Type a jailbreak attempt..."
)

run_button = st.button("Run Attack")

# -------------------------------
# Run Pipeline
# -------------------------------
if run_button:
    if not user_prompt.strip():
        st.error("Please enter a prompt.")
    else:
        st.write("### Running Pipeline…")

        state_config = {
            "name": f"interactive_{experiment_choice}",
            "model": model_titles[model_choice],
            "token_max": 4000,
            "smooth": recipe["smooth"],
            "judges": recipe["judges"],
            "judge_threshold": recipe["judge_threshold"],
            "template": recipe["template"],
        }

        # Build an initial state for the graph
        state = {
            "user_input": user_prompt, 
            "messages": [{"role": "user", "content": user_prompt}],
            "config": state_config,
        }
        try:
            result = pipeline_graph.invoke(state)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            raise e

        # -------------------------------
        # Display results
        # -------------------------------
        st.success("Pipeline complete!")

        st.subheader("Model Final Output")
        st.write(result.get("final_response", "No final response."))

        flagged = result.get("jailbreak", False)
        if flagged:
            st.error("🚨 **Jailbreak detected!**")
        else:
            st.success("🟢 No jailbreak detected.")
