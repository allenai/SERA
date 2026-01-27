import anthropic
import argparse
import copy
import json
import networkx as nx
import os
import random
import re
import shutil
import subprocess
import time
import yaml

from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from jinja2 import Template
from litellm import completion, APIError
from openai import OpenAI, APIConnectionError, RateLimitError
from pathlib import Path
from tqdm import tqdm

from sera.datagen.train.filter_dataset_hf import filter_dataset

##############################
# misc

def dump_json(fp, data, overwrite=False):
    if not overwrite:
        assert not os.path.exists(fp), f"{fp} already exists"
    with open(fp, "w") as f:
        json.dump(data, f, indent=4)

def dump_jsonl(fp, data, overwrite=False):
    if not overwrite:
        assert not os.path.exists(fp), f"{fp} already exists"
    with open(fp, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def save_yaml(fp, data, overwrite=False):
    if not overwrite:
        assert not os.path.exists(fp), f"{fp} already exists"
    with open(fp, "w") as f:
        yaml.safe_dump(data, f, indent=2, sort_keys=False)

def load_yaml(fp):
    with open(fp, "r") as f:
        return yaml.safe_load(f)

def filter_messages(dataset, truncate=False, return_token_to_data_tuples=False):
    with open("./train/train_config/qwen3_32b.yml", "r") as f:
        cfg = yaml.safe_load(f)
    return filter_dataset("Qwen/Qwen3-8B", dataset, truncate=truncate, return_token_to_data_tuples=return_token_to_data_tuples, conversation_style="openai", conversation_column="messages", custom_limit=32768)

##############################
# postprocess
def pp_regex(text, re_string=r"<output>(.*?)</output>"):
    matches = re.findall(re_string, text, re.DOTALL)
    if len(matches) == 0:
        return None
    return matches

def pp_query(system, prompt, model, base_url="", api_key="", max_tokens=4096, retries=0, args={}):
    # Create OpenAI-compatible client
    # Slice openai if it starts with
    if model.startswith("anthropic/"):
        model = model[len("anthropic/"):]
        base_url = "https://api.anthropic.com/v1/"
        api_key = os.getenv("ANTHROPIC_API_KEY")
    if base_url != "":
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        max_tokens = max_tokens
    else:
        client = OpenAI()
        max_tokens = max_tokens
    if len(args) > 0:
        task_prompt = Template(prompt).render(**args)
        # print(task_prompt)
    else:
        task_prompt = prompt
    # print("Prompt:", task_prompt)
    # Make a request
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.6,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": task_prompt}
                ]
            )
            break
        except Exception as e:
            # print("=== ERROR ===")
            if retries == 0:
                raise
            time.sleep(30)
            retries -= 1
    return completion.choices[0].message.content

##############################
# Experiment Folder

class ExperimentFolder:
    """
    Wrapper around experiment folder and absolute paths to directories within.
    """
    def __init__(self, root_dir: Path, config_dir: Path, data_dir: Path, traj_dir: Path):
        self.root_dir = root_dir
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.traj_dir = traj_dir

    @classmethod
    def create(
        cls,
        base_dir: str,
        name: str,
        exist_ok: bool = True,
    ):
        """
        Creates:
          base_dir/name/
            <configs>/
            <data>/
            <trajs>/
        """
        base_dir = Path(base_dir).expanduser().resolve()
        base_dir.mkdir(exist_ok=True)

        if name is None:
            name = "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        root_dir = base_dir / name
        root_dir.mkdir(exist_ok=exist_ok)

        subfolder_names = ('configs', 'data', 'trajs')
        subpaths = tuple(root_dir / s for s in subfolder_names)
        for p in subpaths:
            p.mkdir(exist_ok=exist_ok)

        return cls(root_dir=root_dir, config_dir=subpaths[0], data_dir=subpaths[1], traj_dir=subpaths[2])

    def add_config(self, path: Path):
        shutil.copy(src=path, dst=self.config_dir)