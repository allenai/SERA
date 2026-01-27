import argparse
import copy
import json
import networkx as nx
import os
import random
import re
import subprocess
import time
import yaml
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import List

import anthropic
from jinja2 import Template
from litellm import completion, APIError
from openai import OpenAI, APIConnectionError, RateLimitError

from sera.config_schema import GenerateConfig

from sera.utils import (
    pp_query, 
    pp_regex, 
    ExperimentFolder
)
from sera.datagen.data.generate.classes import (
    SyntheticDataset, 
    SyntheticInstance,
    LocalRepository,
    RepositoryInstance
)

##############################
# utils
def create_instance(synthetic: SyntheticInstance, 
                    id_number):
    synthetic = copy.deepcopy(synthetic)
    repo = synthetic.repo
    synthetic.repo = None
    # Keep real values
    sd = asdict(synthetic)
    synthetic_dict = {}
    for key in sd:
        if sd[key]:
            synthetic_dict[key] = sd[key]
    
    # JSON instance
    json_instance = {
        "extra_fields": synthetic_dict,
        "image_name": repo.image_name,
        "id": f"{repo.get_full_name()}_{id_number}",
        "problem_statement": "n/a",
        "repo_name": "testbed" # So swe_env will set ROOT env var, needed for submit tool call
    }
    if "repo" in json_instance["extra_fields"]:
        json_instance["extra_fields"].pop("repo")
    return json_instance

class NoBugDataset(SyntheticDataset):
    def __init__(self, config: GenerateConfig, repositories: list[LocalRepository], metadata_dir: str, folder: ExperimentFolder):
        super().__init__()
        
        self.generate_cfg = config
        self.metadata_dir = metadata_dir
        self.folder = folder
        self.repos = repositories

        self.instance_path = folder.data_dir / "stage_one_instances.yaml"
        self.fns_per_repo = config.fns_per_repo
        self.insts_per_fn = config.insts_per_fn

    def build_dataset(self):
        if not os.path.exists(self.instance_path):
            instances = self.process_dataset()
            with open(self.instance_path, "w") as f:
                yaml.safe_dump(instances, f, indent=2)
        else:
            print(f"Instance file {self.instance_path} already exists")

    def process_dataset(self):
        # Generate instances
        output_instances = []
        
        instances_to_process: List[RepositoryInstance] = []
        for repo in self.repos:
            instances_to_process += repo.instances

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for repo_instance_object in instances_to_process:
                futures.append(executor.submit(self.process_repo, repo_instance_object))

            for future in as_completed(futures):
                repo_instances = future.result()
                if repo_instances:  # handle None returns
                    output_instances.extend(repo_instances)
        # Save results
        json_output_instances = []
        repo_idxs = {}
        for synth_instance in output_instances:
            idx = repo_idxs.get(synth_instance.repo.get_full_name(), 10000)
            instance = create_instance(synthetic=synth_instance,
                                       id_number=idx)
            json_output_instances.append(instance)
            repo_idxs[synth_instance.repo.get_full_name()] = idx + 1
            
            # json_output_instances.append(synth_instance)
        print(f"Generated {len(output_instances)} instances")
        return json_output_instances

    def process_instance(self, fn_path, replicas: int, repo: RepositoryInstance):
        new_instances = []
        # Create instances
        for _ in range(replicas):
            # Get a random start function in case for the prompt
            synthetic_instance = SyntheticInstance(start_fn=fn_path.split("::")[1],
                                                   start_fn_file=fn_path.split("::")[0],
                                                   repo=repo)
            new_instances.append(synthetic_instance)
        return new_instances        

    def process_repo(self, repo: RepositoryInstance):
        repo_instances = []
        n_fn_processed = 0
        call_graph_nodes = list(set(repo.call_graph)) # Shuffle nodes since its not actually unordered
        random.shuffle(call_graph_nodes)
        print("Total Functions:", len(call_graph_nodes))
        for fn_path in tqdm(call_graph_nodes, desc=repo.get_full_name()):
            if "tests" in fn_path:
                continue
            result = self.process_instance(fn_path=fn_path,
                                           replicas=self.insts_per_fn,
                                           repo=repo,)
            if result is None:
                continue
            repo_instances.extend(result)
            n_fn_processed += 1
            if n_fn_processed >= self.fns_per_repo:
                break
        return repo_instances