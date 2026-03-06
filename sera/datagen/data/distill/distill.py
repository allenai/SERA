import json
import os
import subprocess
import yaml

from filelock import FileLock
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from typing import List

from sera.config_schema import DistillConfig, ModelConfig, SWEAgentWrapperConfig
from sera.utils import ExperimentFolder, load_yaml

def get_dataset_shard(instances_fp, shard, total_shards):
    if shard > total_shards - 1 or shard < 0:
        raise RuntimeError("Invalid shard")
    instance_file, instance_ext = os.path.splitext(instances_fp)
    with open(instances_fp, "r") as f:
        instances = yaml.safe_load(f)
    sharded_data = [[] for _ in range(total_shards)]
    for i, inst in enumerate(instances):
        if "repo" in inst["extra_fields"]:
            inst["extra_fields"].pop("repo")
        sharded_data[i % total_shards].append(inst)
    # Shard dataset into `total_shards`
    for i in range(total_shards):
        saving_fp = f"{instance_file}_shard{i}_{len(sharded_data[i])}-{len(instances)}{instance_ext}"
        if shard == i:
            return_fp = saving_fp
            if not os.path.exists(saving_fp):
                print(f"Saving {saving_fp}...")
                with open(saving_fp, "w") as f:
                    yaml.safe_dump(sharded_data[i], f, indent=2)
    return return_fp

class DistillRunner:

    def __init__(self, config: DistillConfig, folder: ExperimentFolder, instances_fp: Path, cfg_fp: Path, args={}):
        print(instances_fp, cfg_fp)
        assert instances_fp.exists() and cfg_fp.exists()
        self.config = config
        self.cfg_fp = str(cfg_fp)
        self.cfg_name = cfg_fp.stem
        self.run_title = instances_fp.stem
        self.shard = None
        self.total_shards = None
        self.folder = folder
        self.args = args
        self.model = config.model
        
        if config.shard >= config.total_shards:
            raise RuntimeError("Cannot have a shard index more than what the total shards allows")
        if config.shard >= 0 and config.total_shards > 1:
            print(f"Dataset shard: {config.shard}")
            total_shards = config.total_shards
            self.shard = config.shard
            self.total_shards = total_shards
            self.instances_fp = get_dataset_shard(str(instances_fp), shard=self.shard, total_shards=self.total_shards)
        else:
            self.instances_fp = str(instances_fp)

    @property
    def name(self):
        if self.shard:
            name = "_".join([self.run_title,
                            self.config.agent_harness,
                            self.cfg_name,
                            self.model.name,
                            f"{self.shard}-{self.total_shards}",
                            f"t{self.config.sweagent_wrapper_config.temperature}",
                            f"msteps{self.config.sweagent_wrapper_config.per_instance_call_limit}",
                            f"mcost{self.config.sweagent_wrapper_config.per_instance_cost_limit}",
                            f"mtcost{self.config.sweagent_wrapper_config.total_cost_limit}",])
        else:
            name = "_".join([self.run_title,
                            self.config.agent_harness,
                            self.model.name,
                            self.cfg_name,
                            f"t{self.config.sweagent_wrapper_config.temperature}",
                            f"maxsteps{self.config.sweagent_wrapper_config.per_instance_call_limit}",
                            f"maxcost{self.config.sweagent_wrapper_config.per_instance_cost_limit}",
                            f"maxtotalcost{self.config.sweagent_wrapper_config.total_cost_limit}",])
        name = name.replace("/", "-")
        return name

    @property
    def output_dir(self):
        return self.folder.traj_dir / self.name

    def _build_sweagent_cmd(self, output_dir, num_workers, model_name, model_api_base,
                            per_instance_cost_limit, total_cost_limit, temperature, per_instance_call_limit):
        """Build the sweagent run-batch CLI command."""
        if not model_name:
            print(f"Multiturn with default in {self.cfg_fp}...")
            cmd = f"sweagent run-batch --config {os.path.join('/', self.cfg_fp)} \
            --num_workers {num_workers} \
            --instances.type file \
            --instances.path {self.instances_fp} \
            --instances.shuffle True \
            --output_dir {output_dir} \
            --agent.model.per_instance_cost_limit {per_instance_cost_limit} \
            --agent.model.total_cost_limit {total_cost_limit} \
            --agent.model.temperature {temperature} \
            --agent.model.per_instance_call_limit {per_instance_call_limit}"
        elif model_name and not model_api_base:
            print(f"Multiturn with default in {self.cfg_fp} with {model_name}...")
            cmd = f"sweagent run-batch --config {os.path.join('/', self.cfg_fp)} \
            --num_workers {num_workers} \
            --instances.type file \
            --instances.path {self.instances_fp} \
            --agent.model.name {model_name} \
            --instances.shuffle True \
            --output_dir {output_dir} \
            --agent.model.per_instance_cost_limit {per_instance_cost_limit} \
            --agent.model.total_cost_limit {total_cost_limit} \
            --agent.model.temperature {temperature} \
            --agent.model.per_instance_call_limit {per_instance_call_limit} \
            --agent.model.api_key {os.getenv('ANTHROPIC_API_KEY')}"
        else:
            print(f"Multiturn with {model_name} and {model_api_base}...")
            if total_cost_limit != 0 or per_instance_cost_limit != 0:
                print("If your model is not mapped in litellm, then set the costs to 0.0 or else there will be an error")
            cmd = f"sweagent run-batch --config {os.path.join('/', self.cfg_fp)} \
            --num_workers {num_workers} \
            --instances.type file \
            --instances.path {self.instances_fp} \
            --instances.shuffle True \
            --output_dir {output_dir} \
            --agent.model.api_base {model_api_base} \
            --agent.model.name {model_name} \
            --agent.model.per_instance_cost_limit {per_instance_cost_limit} \
            --agent.model.total_cost_limit {total_cost_limit} \
            --agent.model.temperature {temperature} \
            --agent.model.per_instance_call_limit {per_instance_call_limit}"
        return cmd

    def _build_mini_swe_agent_cmd(self, output_dir, num_workers, model_name, model_api_base,
                                   per_instance_cost_limit, total_cost_limit, temperature, per_instance_call_limit):
        """Build the mini-swe-agent swebench CLI command."""
        print(f"mini-swe-agent with {model_name or 'default'} and config {self.cfg_fp}...")
        cmd = f"python -m minisweagent.run.benchmarks.swebench \
        -c {self.cfg_fp} \
        --instances-file {self.instances_fp} \
        --shuffle \
        -o {output_dir} \
        -w {num_workers}"
        if model_name:
            cmd += f" -m {model_name}"
        # mini-swe-agent uses config-level cost/step limits, so pass via -c overrides
        cmd += f" -c agent.step_limit={per_instance_call_limit}"
        cmd += f" -c agent.cost_limit={per_instance_cost_limit}"
        cmd += f" -c model.model_kwargs.temperature={temperature}"
        return cmd

    def run(self):
        """
        instances_fp: Name of instances yaml
        cfg_fp: Name of sweagent config
        """
        try:
            output_dir = str(self.output_dir)
            num_workers = self.config.sweagent_wrapper_config.num_workers
            per_instance_cost_limit = self.config.sweagent_wrapper_config.per_instance_cost_limit
            total_cost_limit = self.config.sweagent_wrapper_config.total_cost_limit
            temperature = self.config.sweagent_wrapper_config.temperature
            per_instance_call_limit = self.config.sweagent_wrapper_config.per_instance_call_limit
            model_name, model_api_base = self.model.name, self.model.url
            print(model_name, model_api_base)

            agent_harness = getattr(self.config, "agent_harness", "sweagent")
            if agent_harness == "mini-swe-agent":
                cmd = self._build_mini_swe_agent_cmd(
                    output_dir, num_workers, model_name, model_api_base,
                    per_instance_cost_limit, total_cost_limit, temperature, per_instance_call_limit,
                )
            else:
                cmd = self._build_sweagent_cmd(
                    output_dir, num_workers, model_name, model_api_base,
                    per_instance_cost_limit, total_cost_limit, temperature, per_instance_call_limit,
                )

            for key, value in self.args.items():
                if isinstance(value, bool):
                    if value:
                        cmd = cmd + f" --{key}"
                else:
                    cmd = cmd + f" --{key} {value}"
            print(f"Agent Launch: {cmd}")
            subprocess.run(cmd.split())
        except Exception as e:
            print(e)
            raise
        return Path(output_dir)

# Maybe make this just run with the current run, process just this shard or wtv based on the class
def scrape_synthetic_prs(instance_fp: Path, traj_dir: Path, remove_duplicates: bool = True):
    instance_dict = {}
    seen_patches = set()
    with open(instance_fp, "r") as f:
        instances = yaml.safe_load(f)
        for inst in tqdm(instances):
            inst_pred = os.path.join(traj_dir, inst["id"], f"{inst['id']}.pred")
            # Get pred patches
            if os.path.exists(inst_pred):
                with open(inst_pred, "r") as f:
                    try:
                        pred_json = json.load(f)
                    except json.decoder.JSONDecodeError as e:
                        continue
                    model_patch = pred_json["model_patch"]
                if model_patch:
                    inst["extra_fields"]["pred_patch"] = model_patch
                    if remove_duplicates:
                        if model_patch in seen_patches:
                            continue
                        else:
                            seen_patches.add(model_patch)
                else:
                    print(f"did not find pred patch for {inst['id']}")
                    continue
            else:
                continue
            synth_path = os.path.join(traj_dir, inst["id"], f"{inst['id']}.synth")
            with open(synth_path, "r") as f:
                try:
                    synth_metadata = json.load(f)
                except Exception as e:
                    continue
                if synth_metadata["is_good_patch"] and synth_metadata["synth_pr"]:
                    inst["problem_statement"] = synth_metadata["synth_pr"]
                else:
                    continue
            instance_dict[inst["id"]] = inst

    instances_with_prs = []
    for inst_id, inst in instance_dict.items():
        instances_with_prs.append(inst)
    return instances_with_prs

def main(config: DistillConfig, folder: ExperimentFolder, stage: str, metadata_only: bool = False): # TODO: Add save configs to experiment fodler
    assert stage in ["stage_one", "stage_two"]
    args = OmegaConf.to_container(config.args, resolve=True)
    if stage == "stage_one":
        assert args.get("pipeline", False), "To run stage one of SVG, distill.args.pipeline should be True"
        instances_fp = folder.data_dir / "stage_one_instances.yaml" # TODO: Replace with a constant
        config_fp = folder.config_dir / f"{config.stage_one_config_name}.yaml"
    else:
        instances_fp = folder.data_dir / "stage_two_instances.yaml" # TODO: Replace with a constant
        config_fp = folder.config_dir / f"{config.stage_two_config_name}.yaml"
        # Set pipeline to False if starting from a later stage because the instance yamls are configured for normal SWE-agent
        args["pipeline"] = False
    distiller = DistillRunner(config=config, folder=folder, instances_fp=instances_fp, cfg_fp=config_fp, args=args)
    if metadata_only:
        output_dir = distiller.output_dir
    else:
        output_dir = distiller.run() # TODO: Test with small repo/dataset so it actually finishes
    return output_dir, distiller.instances_fp # distiller may modify the instance file (if it shards)
