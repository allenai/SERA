import hydra
import os

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict

from sera.config_schema import SeraConfig  # Default config structure
from sera.utils import ExperimentFolder, save_yaml, dump_json
from sera.datagen.data.generate.generate import main as generate_method
from sera.datagen.data.distill.distill import scrape_synthetic_prs, main as distill_method
from sera.datagen.data.eval.eval import eval_loop
from sera.datagen.data.postprocess.postprocess import format_and_save

##############################
# Definitions and helpers

cs = ConfigStore.instance()
cs.store(name="sera", node=SeraConfig)  # Register structured defaults

##############################
# Experiment class

class Experiment:
    stage_map = {
        "pipeline": -1,
        "generate": 0,
        "distill_stage_one": 1,
        "distill_stage_two": 2,
        "eval": 3,
        "postprocess": 4
    }
    def __init__(self, cfg: SeraConfig, folder: ExperimentFolder):
        self.folder = folder
        self.general_cfg = cfg

    def run(self, stage: str):
        # translate to a number
        assert stage in self.stage_map
        OmegaConf.to_container(self.general_cfg, resolve=True) # Check all stages
        self._run_pipeline(self.general_cfg, stage_idx=self.stage_map.get(stage))

    def _run_generate(self, cfg: DictConfig, skip: bool = False) -> None:
        print("GENERATE")
        print(OmegaConf.to_yaml(cfg))
        if not skip:
            generate_method(config=cfg,
                            metadata_dir=self.general_cfg.metadata_dir, 
                            folder=self.folder)

    def _run_distill_one(self, cfg: DictConfig, skip: bool = False) -> None:
        print("DISTILL")
        print(OmegaConf.to_yaml(cfg))
        stage_one_output_dir, stage_one_instances_fp = distill_method(config=cfg, folder=self.folder, stage="stage_one", metadata_only=skip)
        self.stage_one_output_dir = stage_one_output_dir
        self.stage_one_instances_fp = stage_one_instances_fp

    def _run_distill_two(self, cfg: DictConfig, skip: bool = False) -> None:
        print("DISTILL")
        print(OmegaConf.to_yaml(cfg))
        stage_two_instances_fp = self.folder.data_dir / "stage_two_instances.yaml"
        if not skip and not stage_two_instances_fp.exists():
            synthetic_pr_instances = scrape_synthetic_prs(self.stage_one_instances_fp, self.stage_one_output_dir)
            save_yaml(stage_two_instances_fp, synthetic_pr_instances)
        stage_two_output_dir, stage_two_instances_fp = distill_method(config=cfg, folder=self.folder, stage="stage_two", metadata_only=skip)
        self.stage_two_output_dir = stage_two_output_dir
        self.stage_two_instances_fp = stage_two_instances_fp

    def _run_eval(self, cfg: DictConfig, skip: bool = False) -> None:
        print("EVAL")
        print(OmegaConf.to_yaml(cfg))
        report_fp = self.stage_two_output_dir / f"report_t{cfg.compare_patch_threshold}.json"
        if skip:
            self.report_fp = report_fp
            return
        resolved_instances = eval_loop(cfg, instances_fp=self.stage_two_instances_fp, second_stage_dir=self.stage_two_output_dir)
        dump_json(report_fp, {"resolved_ids": resolved_instances})
        self.report_fp = report_fp

    def _run_postprocess(self, cfg: DictConfig, skip: bool = False) -> None:
        print("POSTPROCESS")
        print(OmegaConf.to_yaml(cfg))
        stage_one_fp = format_and_save(config=cfg, traj_dir=self.stage_one_output_dir, report_path=None, out_dir=self.folder.data_dir)
        stage_two_fp = format_and_save(config=cfg, traj_dir=self.stage_two_output_dir, report_path=self.report_fp, out_dir=self.folder.data_dir)

    def _run_pipeline(self, cfg: DictConfig, stage_idx: int = -1) -> None:
        self._run_generate(self.general_cfg.generate, skip=stage_idx > 0)
        self._run_distill_one(self.general_cfg.distill, skip=stage_idx > 1)
        self._run_distill_two(self.general_cfg.distill, skip=stage_idx > 2)
        self._run_eval(self.general_cfg.eval, skip=stage_idx > 3)
        self._run_postprocess(self.general_cfg.postprocess, skip=stage_idx > 4)

##############################
# Runner
# ----- (ask danny?)
# TODO: fix sweagent and add way to get personal prs
# TODO: configs for swebench, example usecases
# TODO: uv install
# TODO: submit open source request
# -----
# TODO: Cleanup
# TODO: Run on UW cluster
# TODO: document everything
# TODO: See if possible to use container thats only local

@hydra.main(version_base=None, config_path="configs", config_name="config") # TODO: Add example of running on a standard cluster with shards
def main(cfg: DictConfig) -> None:

    # Folder setup
    expt_folder = ExperimentFolder.create(base_dir=cfg.experiment_dir, name=cfg.name)
    p = Path(cfg.metadata_dir).expanduser().resolve()
    p.mkdir(exist_ok=True)
    cfg.metadata_dir = str(p)
    for cfg_name in cfg.sweagent_cfgs:
        expt_folder.add_config(path=os.path.join(cfg.sweagent_cfg_dir, f"{cfg_name}.yaml"))

    # Run
    stage = cfg.get("stage", "pipeline")
    expt = Experiment(cfg=cfg, folder=expt_folder)
    expt.run(stage)

if __name__ == "__main__":
    main()
