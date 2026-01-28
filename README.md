![SERA](assets/SERA.jpg)

Test SERA in Claude Code for Free: https://github.com/allenai/sera-cli.

Technical Report: https://allenai.org/papers/opencodingagents

# Installation

Clone the repository locally, and then set up the environment.

With pip:
```
conda create -n sera python=3.12
pip install -e . -e modules/code2flow -e modules/SWE-agent 
```

# Generation

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

### 1. Specialization to Django/Sympy from SWE-Bench

```
python sera/main.py \
    --config-name=specialization_django \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=HOSTNAME
```

### 2. Specialization to Personal Repositories

[/sera/configs/specialiation_personal.yaml](/sera/configs/specialiation_personal.yaml) defines a set of arbitrary codebases to generate data from.

```
python sera/main.py \
    --config-name=specialization_personal \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=HOSTNAME \
    generate.docker.docker_org=alleninstituteofai \
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
      top_level_folder:
      # The main folder to look for to parse out functions.
      # Multiple folders can be provided if the main folder changes through commits
      # If this isnt provided, we try to automatically identify it by checking for folders like src/
        - openhands
    - org_name: R2E-Gym
      last_name: R2E-Gym
      install_cmds: 
        - "pip install -e ."
```

Right now, specialization requires a **Github organization** to store repository mirrors. We provide oca-repos as a public Github organization for mirrors,  but users should create their own if they want to generate data from private codebases. See [Creating GitHub Mirrors and Pushing Containers](#creating-github-mirrors).
We also set a placeholder **Docker organization** to push images to. This *will fail* to push, but that does not affect the ability to run the pipeline. However, created images will need to be rebuilt every time the pipeline is rerun. If you want to avoid this, set a new Docker organization.

The default synthetic PRs created in the second rollout use SWE-Bench as demonstrations. In [Personal PR Issues](#personal-pr-issues), we explain how you can set the demonstrations to be your own PR issues.

### 3. Using Closed-Source Models

If you want to use closed-source models, then the step of creating inference servers can be skipped.

```
python sera/main.py \
    --config-name=specialization_anthropic \
    generate.docker.docker_org=alleninstituteofai \
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
    distill.model.url=HOSTNAME_1
```

Replica 2:
```
python sera/main.py \
    --config-name=swesmith_scaling \
    distill.shard=1 \
    distill.total_shards=4 \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=HOSTNAME_2
```
etc.

## Creating Github Mirrors

This is a required step right now to create a docker container for your personal repository in the second example.
- [Github Org Creation](https://docs.github.com/en/organizations/collaborating-with-groups-in-organizations/creating-a-new-organization-from-scratch)

We have created a mirror org on Github for everyone to use called oca-repos. However, any repositories mirrored here *will* be publicly viewable, so we recommend creating your own organization if you want to generate data from your private repository.

## Continuing an Interrupted Run

At large scales, generation can hang at a particular stage due to long trajectories that can take hours to complete. To handle this, we support restarting any run from an arbitrary stage.
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
    distill.shard=0 \
    distill.total_shards=1 \
    "distill.models.0.model=openai/GLM-4.5-Air" \
    "distill.models.0.url=HOSTNAME" \
    stage=distill_stage_two
```
And then the pipeline will skip the hanging trajectories and proceed to the second rollout using the successful rollouts.

Alternatively, you can just rerun the initial command and as long as the experiment name matches the original run, it will pick up exactly where it left off. However, during trajectory rollouts, the final 3-5% often take significantly longer to complete, so we suggest just starting from the next stage.

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

# Training

See the README.md in sera/datagen/train.

# Citation
```
@article{sera2026,
  title={SERA: Soft-Verified Efficient Repository Agents},
  author={Shen, Ethan and Tormoen, Daniel and Shah, Saurabh and Farhadi, Ali and Dettmers, Tim},
  year={2026},
  institution={Allen Institute for AI},
  url={https://allenai.org/papers/opencodingagents}
}
```
