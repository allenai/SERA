
Note: Expect many helpful updates in the next few days as we improve the codebase and clean up the pipeline!

Use SERA models directly in Claude Code: https://github.com/allenai/sera-cli.

# Installation

Clone the repository locally, and then set up the environment.

With pip:
```
conda create -n sera python=3.12
pip install -e . -e modules/code2flow -e modules/SWE-agent 
```

Show images

# Data Generation

## Example Commands for Anthropic

These examples can be run as long as ANTHROPIC_API_KEY is set.

1. Generate data from a repository that has a docker image already (from SWE-bench, SWE-smith, SWE-gym, etc.).

```
python sera/main.py name=test --config-name=config_specialize_existing_anthropic.yaml
```

2. Generate data from a personal repository. 

```
python sera/main.py --config-name=config_specialize_personal_anthropic.yaml \
                    --generate.docker.docker_org=YOUR_DOCKER_ORG \
                    --generate.docker.gh_mirror_org=YOUR_GITHUB_MIRROR_ORG
```

## Example Commands for Local Models

These examples require a hosted model that exposes hostname and port.

1. Generate data from a repository that has a docker image already.

```
python sera/main.py --config-name=config_specialize_existing_localmodel.yaml \
                    --distill.models[0].model=openai/GLM-4.5-Air \
                    --distill.models[0].url=YOUR_URL
```

2. Generate data from a personal repository. 

```
python sera/main.py --config-name=config_specialize_existing_localmodel.yaml \
                    --generate.docker.docker_org=YOUR_DOCKER_ORG \
                    --generate.docker.gh_mirror_org=YOUR_GITHUB_MIRROR_ORG \
                    --distill.models[0].model=openai/GLM-4.5-Air \
                    --distill.models[0].url=YOUR_URL
```

## Running at Scale
`config_fullscale.yaml` reflects a configuration for larger scale experiments.
For these larger runs, we recommend multiple teacher models be run concurrently. By using the `shard` argument, these models can each be given a unique partition of the input data to process, significantly improving efficiency. For example, when multiple servers are being hosted, as in the configs above, we can run 
```
python sera/main.py --config-name=config_fullscale.yaml \
                    --generate.docker.docker_org=YOUR_DOCKER_ORG \
                    --generate.docker.gh_mirror_org=YOUR_GITHUB_MIRROR_ORG \
                    --distill.shard 0

python sera/main.py --config-name=config_fullscale.yaml \
                    --generate.docker.docker_org=YOUR_DOCKER_ORG \
                    --generate.docker.gh_mirror_org=YOUR_GITHUB_MIRROR_ORG \
                    --distill.shard 1

python sera/main.py --config-name=config_fullscale.yaml \
                    --generate.docker.docker_org=YOUR_DOCKER_ORG \
                    --generate.docker.gh_mirror_org=YOUR_GITHUB_MIRROR_ORG \
                    --distill.shard 2
...
```
In addition, `sera/config_schema.py` contains the full set of configuration settings that can be used.

## Creating Github Mirrors and Pushing Containers

This is a required step right now to create a docker container for your personal repository in the second example.
- [Docker Org Creation](https://docs.docker.com/admin/organization/orgs/)
- [Github Org Creation](https://docs.github.com/en/organizations/collaborating-with-groups-in-organizations/creating-a-new-organization-from-scratch)

We have created a mirror org on Github for everyone to use called oca-repos. However, any repositories mirrored here *will* be publicly viewable, so we recommend creating your own organization if you want to generate data from your private repository.

Unfortunately, Docker Org creation requires a subscription. Usually, companies and research groups have one, but don't worry if you do not. We are actively looking into a solution that works for everyone!

## Staging

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
python sera/main.py name=test stage=distill_stage_two --config-name=config_specialize_existing_anthropic.yaml
```
And then the pipeline will skip the hanging trajectories and proceed to the second rollout using the successful rollouts.


# Training

See the README.md in sera/datagen/train!