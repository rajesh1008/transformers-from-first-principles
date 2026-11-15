"""Shared set-up for Chapter 16's scripts (run from 04-code/ch16).  NumPy only.
Reference machine: the book's build machine (CPU, 8 cores), NumPy 2.5.4.  Timings: one thread,
fastest of five, dated.  The configuration files in configs/ were downloaded by fetch.py."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
SEED = 1865
CONFIGS = os.path.join(HERE, "configs")
MODELS = {  # short name -> file stem in configs/
    "GPT-2 small": "openai-community_gpt2",
    "SmolLM2-135M": "HuggingFaceTB_SmolLM2-135M",
    "Qwen3-0.6B": "Qwen_Qwen3-0.6B",
}

def load_config(name):
    return json.load(open(os.path.join(CONFIGS, MODELS[name] + ".config.json")))

def load_header(name):
    return json.load(open(os.path.join(CONFIGS, MODELS[name] + ".header.json")))

def save_json(name, obj):
    json.dump(obj, open(os.path.join(HERE, "outputs", name + ".json"), "w"), indent=1)

def c(n):
    return f"{n:,}"
