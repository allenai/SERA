"""
Configuration schema using dataclasses for type safety and validation.
This module defines the structure of all configuration objects used in the system.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from omegaconf import MISSING

@dataclass
class SWEAgentWrapperConfig:
    num_workers: int = 32
    per_instance_call_limit: int = 115
    per_instance_cost_limit: float = 0.0
    total_cost_limit: float = 0.0
    temperature: float = 0.6

@dataclass
class ModelConfig:
    """Configuration for a single model endpoint."""
    model: str = ""
    url: Optional[str] = ""

@dataclass
class PromptConfig:
    """Configuration for model prompts"""
    synthetic_pr_prompt: Optional[str] = None
    check_rollout_prompt: Optional[str] = None

@dataclass
class DockerConfig:
    """Configuration for model prompts"""
    docker_org: Optional[str] = None
    gh_mirror_org: Optional[str] = None

@dataclass
class PersonalRepoConfig:
    """Metadata"""
    org_name: str = MISSING # Github org for repository
    last_name: str = MISSING # Repository name
    commits: Optional[list[str]] = None # List of commits to train on
    n_commits: int = 5 # If `commits` not specified, how many commits to automatically train on
    lookback: int = 365 # How far to look back for `n_commits`
    language: str = "python"
    """Docker setup""" # TODO: Set up to use existing containers if passed in as well
    install_cmds: list[str] = field(default_factory=lambda: ["python -m pip install -e ."])
    test_cmd: Optional[str] = None
    python_version: str = "3.10"
    skip_package_name: List[str] = field(default_factory=list) # Skip installing these packages, sidesteps rare dependency errors
    """Codebase parsing"""
    top_level_folder: List[str] = field(default_factory=list) # Top level code folder, will be automatically found if not specified
    overwrite_cg: bool = False # Whether to overwrite the parsed codebase graph (recommend False)
    max_folder_depth: int = 3 # How deep to parse into the codebase

@dataclass
class ExistingRepoConfig:
    org_name: str = MISSING
    last_name: str = MISSING
    base_commit: Optional[str] = None
    instance_id: Optional[str] = None
    source: Optional[str] = None
    image_name: Optional[str] = None
    """Codebase parsing"""
    top_level_folder: List[str] = field(default_factory=list) # Top level code folder, will be automatically found if not specified
    overwrite_cg: bool = False # Whether to overwrite the parsed codebase graph (recommend False)
    max_folder_depth: int = 3 # How deep to parse into the codebase

#############

@dataclass
class GenerateConfig:
    """Configuration for data generation."""
    default: Dict[str, str] = field(default_factory=lambda: {"repo_domain": "github.com"})
    fns_per_repo: int = 5000
    insts_per_fn: int = 1
    overwrite_call_graph: bool = False
    personal_repos: List[PersonalRepoConfig] = field(default_factory=list)
    existing_repos: List[ExistingRepoConfig] = field(default_factory=list)
    repo_parent_dir: str = "./repos"
    docker: DockerConfig = field(default_factory=DockerConfig)

@dataclass
class DistillConfig:
    """Configuration for distillation process."""
    models: List[ModelConfig] = field(default_factory=list)
    config_name: str = "e2e"
    sweagent_wrapper_config: SWEAgentWrapperConfig = field(default_factory=SWEAgentWrapperConfig)
    args: Dict[str, Any] = field(default_factory=dict)
    shard: int = -1
    total_shards: int = -1
    stage_one_config_name: str = MISSING
    stage_two_config_name: str = MISSING

@dataclass
class EvalConfig: # Handle only resolved here
    compare_patch_threshold: float = 1

@dataclass
class PostprocessConfig: # Postprocessing
    """Configuration for data formatting."""
    tool_call_format: str = "hermes"
    add_think: bool = False # Add <think> tags around any non-toolcall content
    add_train_key: bool = True # Add train key (for axolotl)
    reformat_assistant_message: Optional[str] = "keep_only_think" # empty | keep_only_think | keep_only_non_think
    enforce_submit: bool = True

@dataclass
class SeraConfig:
    """Main configuration object for SERA datagen system."""
    stage: str = "pipeline"
    name: Optional[str] = None
    experiment_dir: str = "./experiments"
    metadata_dir: str = "./metadata"
    sweagent_cfg_dir: str = "./sera/configs/sweagent/"
    sweagent_cfgs: List[str] = field(default_factory=list)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)