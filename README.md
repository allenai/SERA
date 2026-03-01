![SERA](assets/SERA.jpg)

Test SERA in Claude Code for Free: https://github.com/allenai/sera-cli.

Technical Report: https://allenai.org/papers/opencodingagents

# Installation

Clone the repository locally, and then set up the environment.

With pip:
```
git clone --recurse-submodules https://github.com/allenai/SERA.git
cd SERA
conda create -n sera python=3.12
conda activate sera
pip install -e . -e modules/code2flow -e modules/SERA-SWE-agent 
```

# Generation

[The full configuration reference](#full-configuration-reference) expands on the examples below.

## Inference Servers

We support generation from open source and close source models.

We provide launch scripts for [GLM-4.5-Air](sera/datagen/inference/launch_glm45.sh), [GLM-4.6](sera/datagen/inference/launch_glm46.sh), and [Qwen models](sera/datagen/inference/launch_qwen3_models.sh).

Here is example usage:
```
# This sets TP to 8, launches a server on port 24444, and sets a seed of 42.
bash launch_glm45.sh 8 24444 42
```

The resulting server is usually `http://HOSTNAME:PORT/v1` if its an openai server (which our servers are).

## Single Server

Every experiment can be run either with one inference server, or multiple for higher efficiency.
We release several examples showing how to reproduce our experiments or run your own.
[sera/config_schema.py](sera/config_schema.py) contains a full list of configuration settings, enabling even more control over experiments.

### 1. Specialization to Django/Sympy from SWE-Bench

```
python sera/main.py \
    --config-name=specialization_django \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=URL
```

### 2. Specialization to Personal Repositories

[sera/configs/specialization_personal.yaml](sera/configs/specialization_personal.yaml) defines a set of arbitrary codebases to generate data from.

```
python sera/main.py \
    --config-name=specialization_personal \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=URL \
    generate.docker.gh_mirror_org=oca-repos
```

Personal repositories require a little more involvement to generate data because we need to identify the main code folder, installation commands, etc. We suggest modifying the yaml file directly for this, instead of through CLI.

```
    - org_name: OpenHands
      last_name: OpenHands
      commits: # Provide exact commits to make containers on OR we automatically scrape some if not provided
        - 29b77be807e0e6aab380d953c0d79a166df4f0cc
        - cc8b677f3ec324fb7b9de86229f727b25741a66c
      install_cmds: # This is the default but sometimes personal repositories have their own installations
        - "pip install -e ."
      test_cmd: null
      python_version: 3.12 # For the docker container
      top_level_folder: # The main folder to look for to parse out functions.
        - openhands
    - org_name: R2E-Gym
      last_name: R2E-Gym
      install_cmds: 
        - "pip install -e ."
```

Right now, specialization requires a **Github organization** to store repository mirrors. We provide oca-repos as a public Github organization for mirrors,  but users should create their own if they want to generate data from private codebases. See [Creating GitHub Mirrors and Pushing Containers](#creating-github-mirrors).

You can also set a **Docker organization** to push images to using `generate.docker.docker_org=DOCKER_ORG`. This makes it so created images are persistent. Make sure you have push permissions for the organization you choose. Otherwise, created images will be rebuilt every time the pipeline is rerun, taking a few extra minutes.

The default synthetic PRs created in the second rollout use SWE-Bench as demonstrations. In [Personal PR Issues](#personal-pr-issues), we explain how you can set the demonstrations to be your own PR issues.

### 3. Using Closed-Source Models

If you want to use closed-source models, then the step of creating inference servers can be skipped.

```
python sera/main.py \
    --config-name=specialization_anthropic \
    generate.docker.gh_mirror_org=oca-repos
```

## Multiple Servers

Multiple generation runs can be launched in parallel for the same experiment for large data generation runs using sharding. This is when the user defines multiple servers, and then shards the dataset to each server.

We use this for scaling swesmith to generate our largest datasets.

Replica 1:
```
python sera/main.py \
    --config-name=swesmith_scaling \
    distill.shard=0 \
    distill.total_shards=4 \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=URL_1
```

Replica 2:
```
python sera/main.py \
    --config-name=swesmith_scaling \
    distill.shard=1 \
    distill.total_shards=4 \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=URL_2
```
etc.

## Creating Github Mirrors

This is a required step right now to create a docker container for your personal repository in the second example.
- [Github Org Creation](https://docs.github.com/en/organizations/collaborating-with-groups-in-organizations/creating-a-new-organization-from-scratch)

We have created a mirror org on Github for everyone to use called oca-repos. However, any repositories mirrored here *will* be publicly viewable, so we recommend creating your own organization if you want to generate data from your private repository.

## Continuing an Interrupted Run

At large scales, some generations (< 1%) will stall the teacher model, but this is enough to prevent the pipeline from completing a distillation step. To handle this, we support restarting any run from an arbitrary stage if the user chooses to kill a stalled run.
```
    stage_map = {
        "pipeline": -1,
        "generate": 0,
        "distill_stage_one": 1,
        "distill_stage_two": 2,
        "eval": 3,
        "postprocess": 4
    }
```
To continue a generation run simply make sure the name of the run (can be set via `name=`) matches the run to resume. Next, the argument `stage=SOME_STAGE_MAP_KEY` will continue the generation from whatever stage is chosen.

For example, if a few trajectories hang in `distill_stage_one`, you can run:
```
python sera/main.py \
    --config-name=specialization_django \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=URL \
    stage=distill_stage_two
```
And then the pipeline will skip the hanging trajectories and proceed to the second stage using only the successful rollouts from the first stage.

Alternatively, you can just rerun the initial command and as long as the experiment name matches the original run, it will pick up exactly where it left off instead of skipping to the next step in the pipeline.

## Personal PR Issues

We write a script to scrape previous issue texts from any repository.
```
python scrape_github.py -o ORG_NAME -n REPO_NAME -c N_ISSUES
```

This saves a JSON file containing a list of issues to a `pr_issues` directory.

This list can then be passed into a run as:
```
distill.args.pipeline_repo=GENERATED_PATH
```

# Full Configuration Reference

All settings are defined as dataclasses in [sera/config_schema.py](sera/config_schema.py) and can be overridden via the command line using OmegaConf dot notation. For example:
```
python sera/main.py --config-name=specialization_django \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=http://HOST:PORT/v1 \
    distill.sweagent_wrapper_config.num_workers=16 \
    eval.compare_patch_threshold=0.5
```

## Top-Level Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `stage` | str | `"pipeline"` | Which stage to run. Options: `pipeline` (all stages), `generate`, `distill_stage_one`, `distill_stage_two`, `eval`, `postprocess` |
| `name` | str \| None | `None` | Name of the run. Creates a folder at `experiment_dir/name`. Auto-generated if not set. Used to resume interrupted runs. |
| `experiment_dir` | str | `"./experiments"` | Where to save experiment data (trajectories, rollouts, outputs) |
| `metadata_dir` | str | `"./metadata"` | Where to save parsed codebase graphs and other metadata |
| `sweagent_cfg_dir` | str | `"./sera/configs/sweagent/"` | Directory containing full SWE-agent config YAML files |
| `sweagent_cfgs` | list[str] | `["e2e", "qwen"]` | Which SWE-agent configs to load into the experiment |

## Generate Settings (`generate.*`)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `generate.fns_per_repo` | int | `5000` | Max number of functions to extract per repository |
| `generate.insts_per_fn` | int | `1` | Number of times to process each function through the pipeline. Increase to generate more samples. |
| `generate.repo_parent_dir` | str | `"./repos"` | Where to store cloned repositories |
| `generate.docker.docker_org` | str \| None | `None` | Docker organization to push created images to. Makes containers persistent across reruns. |
| `generate.docker.gh_mirror_org` | str \| None | `None` | GitHub mirror organization for personal repos (required for personal repo containerization) |

### Personal Repos (`generate.personal_repos`)

Defined as a list in YAML config files. Each entry supports:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `org_name` | str | *required* | GitHub organization name |
| `last_name` | str | *required* | GitHub repository name |
| `commits` | list[str] \| None | `None` | Exact commits to create containers for. If not set, auto-scrapes `n_commits` commits. |
| `n_commits` | int | `5` | Number of commits to auto-scrape if `commits` is not specified |
| `lookback` | int | `365` | How many days to look back when auto-scraping commits |
| `language` | str | `"python"` | Repository language. Only Python is supported currently. |
| `install_cmds` | list[str] | `["python -m pip install -e ."]` | Commands to install the repository inside the container |
| `test_cmd` | str \| None | `None` | Optional test command to verify installation |
| `python_version` | str | `"3.10"` | Python version for the Docker container |
| `skip_package_name` | list[str] | `[]` | Packages to skip installing (sidesteps rare dependency errors) |
| `top_level_folder` | list[str] | `[]` | Main code folder(s) to parse (e.g. `src`). Auto-detected if empty. |
| `overwrite_cg` | bool | `False` | Set `True` to regenerate codebase graphs instead of using cache |
| `max_folder_depth` | int | `3` | How deep to parse into the codebase. Higher = more functions extracted. |

### Existing Repos (`generate.existing_repos`)

For repositories with pre-built Docker containers (SWE-Bench, SWE-Smith, etc.). Defined as a list in YAML config files.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `org_name` | str | *required* | GitHub organization name |
| `last_name` | str | *required* | GitHub repository name |
| `base_commit` | str \| None | `None` | Commit the container is based on. Auto-set for `swebench`/`swesmith` sources. |
| `instance_id` | str \| None | `None` | SWE-Bench instance ID (e.g. `django__django-7530`). Used to auto-set `base_commit`. |
| `source` | str \| None | `None` | Container source: `"swebench"`, `"swesmith"`, or leave empty for custom |
| `image_name` | str \| None | `None` | Custom Docker image name (for repos that are not from SWE-Bench or SWE-Smith) |
| `top_level_folder` | list[str] | `[]` | Main code folder(s) to parse. Auto-detected if empty. |
| `overwrite_cg` | bool | `False` | Set `True` to regenerate codebase graphs instead of using cache |
| `max_folder_depth` | int | `3` | How deep to parse into the codebase. Higher = more functions extracted. |

## Distill Settings (`distill.*`)

### Model (`distill.model.*`)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `distill.model.name` | str | `""` | Model identifier. Use `openai/` prefix for OpenAI-compatible servers (e.g. `openai/GLM-4.5-Air`), `anthropic/` prefix for Anthropic API (e.g. `anthropic/claude-sonnet-4-20250514`). |
| `distill.model.url` | str \| None | `""` | API endpoint URL (e.g. `http://HOST:PORT/v1`). Leave empty/null for official OpenAI or Anthropic APIs. |

### SWE-Agent Wrapper (`distill.sweagent_wrapper_config.*`)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `distill.sweagent_wrapper_config.num_workers` | int | `32` | Number of concurrent rollouts |
| `distill.sweagent_wrapper_config.per_instance_call_limit` | int | `115` | Max number of rollout steps per instance |
| `distill.sweagent_wrapper_config.per_instance_cost_limit` | float | `0.0` | Max cost per rollout. Set to `0.0` for local models, `> 0.0` for API models. |
| `distill.sweagent_wrapper_config.total_cost_limit` | float | `0.0` | Max total cost across all rollouts |
| `distill.sweagent_wrapper_config.temperature` | float | `0.6` | Model sampling temperature |

### Sharding & Stage Configs

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `distill.shard` | int | `0` | Current shard index (0-indexed) for parallel multi-server runs |
| `distill.total_shards` | int | `1` | Total number of shards to split the data into |
| `distill.stage_one_config_name` | str | `"e2e"` | SWE-agent config name for stage 1 rollouts. Must be in `sweagent_cfgs`. |
| `distill.stage_two_config_name` | str | `"qwen"` | SWE-agent config name for stage 2 rollouts. Must be in `sweagent_cfgs`. |
| `distill.args` | dict | `{"pipeline": True, "pipeline_yaml": "sera/configs/pipeline/default_pipeline.yaml"}` | Extra args passed to SWE-agent. Use `distill.args.pipeline_repo=PATH` to provide custom PR issues. |

## Eval Settings (`eval.*`)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `eval.compare_patch_threshold` | float | `1` | Verification threshold. `1` = hard verification (exact patch match), `0 < r < 1` = soft verification, `0` = no verification. |

## Postprocess Settings (`postprocess.*`)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `postprocess.tool_call_format` | str | `"hermes"` | Tool call format in output data. Options: `hermes`, `xml`, `raw` |
| `postprocess.add_think` | bool | `False` | Add `<think>` tags to output. Useful for training Qwen3 models when the teacher doesn't produce think tokens (e.g. Claude). |
| `postprocess.add_train_key` | bool | `True` | Add train key to assistant messages for axolotl, ensuring only assistant messages are trained on |
| `postprocess.include_tool_json` | bool | `True` | Include OpenAI-formatted tool JSON as a field in each sample (helps debugging) |
| `postprocess.reformat_assistant_message` | str \| None | `"keep_only_think"` | How to handle multi-part assistant messages (e.g. `<think>TEXT</think>MORE TEXT`). Options: `""` (keep all), `"keep_only_think"`, `"keep_only_non_think"` |
| `postprocess.enforce_submit` | bool | `True` | Only process trajectories that successfully submitted, filtering out ones that hit cost/context limits |

## Example: All Settings on CLI

```
python sera/main.py \
    --config-name=specialization_django \
    stage=pipeline \
    name=my_experiment \
    experiment_dir=./experiments \
    metadata_dir=./metadata \
    generate.fns_per_repo=5000 \
    generate.insts_per_fn=1 \
    generate.docker.docker_org=my-docker-org \
    generate.docker.gh_mirror_org=my-gh-mirrors \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=http://localhost:24444/v1 \
    distill.sweagent_wrapper_config.num_workers=24 \
    distill.sweagent_wrapper_config.per_instance_cost_limit=5.0 \
    distill.sweagent_wrapper_config.temperature=0.6 \
    distill.shard=0 \
    distill.total_shards=1 \
    eval.compare_patch_threshold=0.5 \
    postprocess.tool_call_format=hermes \
    postprocess.add_think=false \
    postprocess.enforce_submit=true
```

# Training

See the README.md in [sera/datagen/train](sera/datagen/train).

# Citation
```
@misc{shen2026sera,
      title={SERA: Soft-Verified Efficient Repository Agents}, 
      author={Ethan Shen and Danny Tormoen and Saurabh Shah and Ali Farhadi and Tim Dettmers},
      year={2026},
      eprint={2601.20789},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.20789}, 
}
```
