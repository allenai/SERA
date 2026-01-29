"""
Configuration schema using dataclasses for type safety and validation.
This module defines the structure of all configuration objects used in the system.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from omegaconf import MISSING

@dataclass
class SWEAgentWrapperConfig:
    """
    Wrapper for adjusting SWE agent settings. Suggest directly modifying in sera/configs/sweagent for more options.
    """
    # Number of concurrent rollouts to spin up
    num_workers: int = 32
    # Max number of rollout steps
    per_instance_call_limit: int = 115
    # Max cost per rollout. Set to 0.0 for local model, or > 0.0 for an API
    per_instance_cost_limit: float = 0.0
    # Max cost for entire run across all rollouts
    total_cost_limit: float = 0.0
    # Model temperature
    temperature: float = 0.6

@dataclass
class ModelConfig:
    """Configuration for a single model endpoint."""
    # Model name. Use openai/ as a prefix for local and openai models. anthropic/ prefix for anthropic models.
    name: str = ""
    # Model URL. Leave empty for openai or anthropic API models.
    url: Optional[str] = ""

@dataclass
class DockerConfig:
    """Configuration for container creation for personal repositories"""
    # Docker org to push created images to
    docker_org: Optional[str] = None
    # Mirror organization for Github. 
    gh_mirror_org: Optional[str] = None

@dataclass
class PersonalRepoConfig:
    """Args for generating data from personal repositories"""
    # Github repo org
    org_name: str = MISSING
    # Github repo name
    last_name: str = MISSING
    # List of commits to create containers for
    commits: Optional[list[str]] = None
    # If `commits` not specified, how many commits to automatically scrape
    n_commits: int = 5
    # How many days to look back for `n_commits`
    lookback: int = 365
    # Repository language. Right now only Python supported
    language: str = "python"
    """Container setup"""
    # Installation commands for repository
    install_cmds: list[str] = field(default_factory=lambda: ["python -m pip install -e ."])
    # Commands to test repository installation, e.g. run test scripts
    test_cmd: Optional[str] = None
    # Python version to install into container
    python_version: str = "3.10"
    # Skip installing these packages into container, sidesteps rare dependency errors
    skip_package_name: List[str] = field(default_factory=list)
    """Codebase function parsing"""
    # Top level code folder to search under (e.g. src), will be automatically found if not specified
    top_level_folder: List[str] = field(default_factory=list)
    # Creating codebase graphs takes a few minutes so we cache the created graph. Set to True to turn off caching.
    overwrite_cg: bool = False 
    # How deep to parse into the codebase from `top_level_folder`. Higher number = more functions extracted.
    max_folder_depth: int = 3

@dataclass
class ExistingRepoConfig:
    """Args for generating data from repositories with Docker containers already (swesmith, swebench, etc.)"""
    # Github repo org
    org_name: str = MISSING
    # Github repo name
    last_name: str = MISSING
    # Commit that the container is based on. Needed for function scraping. We auto set this if swesmith or swebench.
    base_commit: Optional[str] = None
    # Pass in this if using swebench container so we can automatically set the `base_commit`
    instance_id: Optional[str] = None
    # swebench | swesmith, or leave empty
    source: Optional[str] = None
    # If the target codebase has a container but is not swebench or swesmith, then set this
    image_name: Optional[str] = None
    """Codebase function parsing"""
    # Top level code folder to search under (e.g. src), will be automatically found if not specified
    top_level_folder: List[str] = field(default_factory=list)
    # Creating codebase graphs takes a few minutes so we cache the created graph. Set to True to turn off caceing.
    overwrite_cg: bool = False 
    # How deep to parse into the codebase from `top_level_folder`. Higher number = more functions extracted.
    max_folder_depth: int = 3

#############

@dataclass
class GenerateConfig:
    """Configuration for data generation."""
    default: Dict[str, str] = field(default_factory=lambda: {"repo_domain": "github.com"})
    # Max number of fns to extract from repositories
    fns_per_repo: int = 5000
    # Number of times to process each function through the pipeline. This can be safely increased to increase sample size.
    insts_per_fn: int = 1
    # Repositories to generate from
    personal_repos: List[PersonalRepoConfig] = field(default_factory=list)
    existing_repos: List[ExistingRepoConfig] = field(default_factory=list)
    # Where to store cloned repositories
    repo_parent_dir: str = "./repos"
    # Args for container creation
    docker: DockerConfig = field(default_factory=DockerConfig)

@dataclass
class DistillConfig:
    """Configuration for distillation process."""
    # Model config
    model: ModelConfig = field(default_factory=ModelConfig)
    # Sweagent config
    sweagent_wrapper_config: SWEAgentWrapperConfig = field(default_factory=SWEAgentWrapperConfig)
    # Extra args to pass into sweagent
    args: Dict[str, Any] = field(default_factory=dict)
    # Shard idx if sharding the data
    shard: int = 0
    # Number of total shards to shard data into
    total_shards: int = 1
    # Sweagent config for rollout one to use. Should be included in SeraConfig.sweagent_cfgs
    stage_one_config_name: str = "e2e"
    # Sweagent config for rollout two to use. Should be included in SeraConfig.sweagent_cfgs
    stage_two_config_name: str = "qwen"

@dataclass
class EvalConfig:
    # Soft verification threshold. 1 for hard-verify. 1 > r > 0 for soft-verify. 0 for no verify.
    compare_patch_threshold: float = 1

@dataclass
class PostprocessConfig: # Postprocessing
    """Configuration for data formatting."""
    # Tool call format. Choose hermes or xml.
    tool_call_format: str = "hermes" # hermes | xml
    # Whether to add <think> tags. Good for training Qwen3 models if the teacher does not use these tags (e.g. Claude).
    add_think: bool = False
    # Add train key to assistant messages (for axolotl), to make sure we only train on assistant messages
    add_train_key: bool = True
    # Some teacher models like GLM produce <think>TEXT</think>MORE TEXT. Change this to choose what part of this output is kept.
    reformat_assistant_message: Optional[str] = "keep_only_think" # empty | keep_only_think | keep_only_non_think
    # Only process trajectories that submit, ignoring ones that hit cost limits, context limits, etc.
    enforce_submit: bool = True

@dataclass
class SeraConfig:
    """Main configuration object for SERA datagen system."""
    # What stage to run
    stage: str = "pipeline"
    # Name of the run. Will create a folder in experiment_dir/name. This is automatically set if not specified.
    name: Optional[str] = None
    # Where to save experiment data
    experiment_dir: str = "./experiments"
    # Where to save parsed codebases and other metadata
    metadata_dir: str = "./metadata"
    # Directory storing full sweagent configs
    sweagent_cfg_dir: str = "./sera/configs/sweagent/"
    # Which sweagent configs to load into this experiment
    sweagent_cfgs: List[str] = field(default_factory=lambda: ["e2e", "qwen"])
    # Stage specific configs
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)