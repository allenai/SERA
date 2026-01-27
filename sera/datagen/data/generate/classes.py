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
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

import anthropic
from jinja2 import Template
from litellm import completion, APIError
from openai import OpenAI, APIConnectionError, RateLimitError

from sera.constants import SWEBENCH_IMAGES, SWESMITH_IMAGES
from sera.datagen.data.generate.docker import build_container
from sera.datagen.data.generate.codebase_parsing import get_adj_list, find_code_folders

##############################
# repository classes for generate.py

DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}

@dataclass
class RepositoryInstance:
    """
    Represents a single instance of a repository thats ready to be processed for data generation.
    """
    parent: None
    base_commit: str
    """Set later via mutator"""
    image_name: str = ""
    call_graph: Optional[nx.DiGraph] = None
    folders: List[str] = field(default_factory=list)

    def setup(self, docker_org: str, gh_mirror_org: str, metadata_dir: str, max_folder_depth: int):
        if not self.image_name:
            print("Creating docker image...")
            self.image_name = self.create_container(docker_org=docker_org, gh_mirror_org=gh_mirror_org)
        if not self.image_name:
            print("Failed to create container")
            return None
        try:
            print("Setting code folders...")
            self.set_code_folders(depth=max_folder_depth)
            print(self.folders)
        except RuntimeError as e:
            print(e)
            return None
        self.create_call_graph(metadata_dir=metadata_dir)
        print(self.call_graph, self.call_graph.number_of_nodes())

    def create_container(self, docker_org: str, gh_mirror_org: str):
        output = build_container(org_dh=docker_org,
                                            org_gh=gh_mirror_org,
                                            gh_owner=self.parent.org_name,
                                            repo_name=self.parent.last_name,
                                            commit=self.base_commit,
                                            install_cmds=self.parent.install_cmds,
                                            test_cmd=self.parent.test_cmd,
                                            language=self.parent.language,
                                            python_version=self.parent.python_version,
                                            package_name=self.parent.skip_package_name
                                         )
        return output

    def set_code_folders(self, depth: int): # Set way to pass these in
        wildcards = find_code_folders(repo_path=self.parent.repo_path, 
                                        repo_last_name=self.parent.last_name,
                                        base_commit=self.base_commit,
                                        top_level_folder=self.parent.top_level_folder)
        if not wildcards:
            raise RuntimeError("Could not automatically find wildcards. Please pass in a top level code folder.")
        for wildcard in wildcards:
            if not depth or wildcard.count("*") <= depth:
                self.folders.append(wildcard)

    def create_call_graph(self, metadata_dir):
        adj_list = get_adj_list(repo_path=self.parent.repo_path, 
                                repo_last_name=self.parent.last_name,
                                base_commit=self.base_commit,
                                relevant_folders=self.folders,
                                metadata_dir=metadata_dir, overwrite=self.parent.overwrite_cg)
        if adj_list is None:
            return None
        self.call_graph = nx.DiGraph(adj_list)

    def get_full_name(self):
        return f"{self.parent.org_name}_{self.parent.last_name}_{self.base_commit[:5]}"

@dataclass(kw_only=True)
class Repository:
    org_name: str
    last_name: str
    top_level_folder: List[str]
    overwrite_cg: bool
    """Set later via mutator"""
    repo_path: Optional[Path] = None
    instances: List[RepositoryInstance] = field(default_factory=list)

    def _clone_repo(self, repo_parent_dir: str | Path):
        """
        Clone the encapsulated repository into `repo_parent_dir`.
        """
        if isinstance(repo_parent_dir, str):
            repo_parent_dir = Path(repo_parent_dir)
        repo_parent_dir.mkdir(exist_ok=True)
        repo_url = f"https://github.com/{self.org_name}/{self.last_name}.git"
        clone_path = repo_parent_dir / self.last_name
        self.repo_path = clone_path
        if clone_path.exists():
            return clone_path
        else:
            subprocess.run(["git", "clone", repo_url], cwd=str(repo_parent_dir), check=True)

    def _set_repo_state(self, commit):
        subprocess.run(["git", "reset", "--hard"], check=True, **DEVNULL, cwd=self.repo_path)
        subprocess.run(["git", "clean", "-fdx"], check=True, **DEVNULL, cwd=self.repo_path)
        subprocess.run(["git", "checkout", commit], cwd=self.repo_path)
        if os.path.isfile(self.repo_path / ".git/index.lock"):
            subprocess.run(["rm", ".git/index.lock"], cwd=self.repo_path)

@dataclass
class ExistingRepository(Repository):
    """
    Represents repositories that already have a docker image from swebench, swesmith, or somewhere else.
    """
    source: str
    base_commit: Optional[str]
    instance_id: Optional[str]
    image_name: Optional[str]

    def setup(self, repo_parent_dir: str | Path, metadata_dir: str, max_folder_depth: int):
        if self.source and self.source == "swebench":
            repo_info = SWEBENCH_IMAGES.get(self.instance_id)
            self.image_name = repo_info["image_name"]
            self.base_commit = repo_info["base_commit"]
        elif self.source and self.source == "swesmith":
            # pull from constants
            repo_info = SWESMITH_IMAGES.get("/".join([self.org_name, self.last_name]))
            self.image_name = repo_info["image_name"]
            self.base_commit = repo_info["base_commit"]
        else:
            if not self.image_name or not self.base_commit:
                raise RuntimeError("If you are not using an existing swesmith or swebench repository, then you must provide an image to use")
        
        self._clone_repo(repo_parent_dir=repo_parent_dir)
        self._create_instances(metadata_dir=metadata_dir, max_folder_depth=max_folder_depth)
    
    def _create_instances(self, metadata_dir: str, max_folder_depth: int):
        self._set_repo_state(self.base_commit)
        repo_instance = RepositoryInstance(image_name=self.image_name, base_commit=self.base_commit, parent=self)
        try:
            repo_instance.setup(docker_org="", gh_mirror_org="", metadata_dir=metadata_dir, max_folder_depth=max_folder_depth)
            self.instances.append(repo_instance)
        except Exception as e:
            print(e)

@dataclass
class LocalRepository(Repository):
    """
    Represents new codebases that we want to create an image for and generate data from.
    """
    python_version: str
    install_cmds: List[str]
    test_cmd: str
    skip_package_name: List[str]
    language: str
    """Set later via mutator"""
    commits: Optional[list[str]] = None

    def setup(self, repo_parent_dir: str | Path, n_commits: int, lookback: int, docker_org: str, gh_mirror_org: str, metadata_dir: str, max_folder_depth: int):
        self._clone_repo(repo_parent_dir=repo_parent_dir)
        if not self.commits:
            self._set_spaced_commits(n_commits=n_commits,
                                    lookback=lookback)
        self._create_instances(docker_org=docker_org, gh_mirror_org=gh_mirror_org, metadata_dir=metadata_dir, max_folder_depth=max_folder_depth)

    def _create_instances(self, docker_org: str, gh_mirror_org: str, metadata_dir: str, max_folder_depth: int):
        for commit in self.commits:
            self._set_repo_state(commit)
            repo_instance = RepositoryInstance(base_commit=commit, parent=self)
            try:
                repo_instance.setup(docker_org=docker_org, gh_mirror_org=gh_mirror_org, metadata_dir=metadata_dir, max_folder_depth=max_folder_depth)
                self.instances.append(repo_instance)
            except Exception as e:
                print(e)
                continue
    
    def _set_spaced_commits(self, n_commits, lookback):
        """
        Get `n_commits` temporally spaced from `lookback` days until now.
        """
        # List commits in window, oldest -> newest. %ct is committer seconds since creation.
        res = subprocess.run(
            ["git", "log", "HEAD", "--since", f"{lookback} days ago", "--reverse", "--format=%H %ct"],
            cwd=str(self.repo_path),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        text = res.stdout.strip()
        if not text:
            print("No commits found when running `git log`.")
            return []
        commits: List[Tuple[str, datetime]] = []
        for line in text.splitlines():
            sha, ct = line.split()
            commits.append(sha)
        m = len(commits)

        # Choose indices including endpoints, approximately evenly spaced.
        if n_commits == 1:
            idxs = [m - 1]
        elif n_commits >= m:
            idxs = list(range(m))
        else:
            # Commit indices we want
            raw = [int(round(i * (m - 1) / (n_commits - 1))) for i in range(n_commits)]
            seen = set()
            idxs = []
            for idx in raw:
                if idx not in seen:
                    idxs.append(idx)
                    seen.add(idx)
            # Fill in commits with newest commits first if we fall short
            if len(idxs) < n_commits:
                for idx in range(m-1, 0, -1):
                    if idx not in seen:
                        idxs.append(idx)
                        seen.add(idx)
                        if len(idxs) == n_commits:
                            break
                idxs.sort()
        self.commits = [commits[i] for i in idxs]


##############################
# classes representing synthetic data for no_bug_dataset.py

@dataclass
class SyntheticInstance:
    repo: RepositoryInstance # SET TO NONE WHEN CONVERTING TO JSON INSTANCE
    start_fn: str
    start_fn_file: str

# Abstract class for synthetic datasets
class SyntheticDataset(ABC):

    @abstractmethod
    def build_dataset(self):
        pass

    @abstractmethod
    def process_instance(self):
        pass

    @abstractmethod
    def process_repo(self):
        pass

    @abstractmethod
    def process_dataset(self):
        pass