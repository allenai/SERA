"""
Filter datasets by trajectory characteristics.

Removes trajectories from a JSONL dataset based on quality heuristics:
  - long_edit:    Remove trajectories with patches > 40 lines changed
  - user_length:  Remove trajectories with high avg tool call response tokens

Usage:
    python filter.py -d <data_file> -fm <filter_mode> [-f <traj_folder>]

Arguments:
    -d, --dataset       Input JSONL file(s), space-separated
    -f, --folder        Trajectory folder(s) containing .pred files (required for long_edit)
    -fm, --filter-mode  Filter mode: long_edit | user_length

Example:
    python filter.py -d stage_two_instances_openai-GLM-4.5-Air_qwen_t0.6_maxsteps115_maxcost0.0_maxtotalcost0.0_noreport_addthink-False_atk-True_rft-True_format-hermes.jsonl \
                     -f stage_two_instances_openai-GLM-4.5-Air_qwen_t0.6_maxsteps115_maxcost0.0_maxtotalcost0.0 \
                     -fm long_edit
    python filter.py -d data.jsonl -fm user_length

Output:
    Saved as <original_name>_filter_<mode>.jsonl in the same directory
"""

import argparse
import json
import logging
import os

from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PATCH_EDIT_MAX = 40
TOOL_CALL_TOKEN_MAX = 600

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', nargs="+")
    parser.add_argument('-f', '--folder', nargs="+")
    parser.add_argument('-fm', '--filter-mode')
    return parser.parse_args()

def analyze_diff(patch_text: str):
    added = 0
    deleted = 0
    new_files = 0

    current_file_is_new = False

    for line in patch_text.splitlines():

        # Detect entering a new file diff block
        if line.startswith("diff --git"):
            # Reset for new file
            current_file_is_new = False

        # Detect new files (git diff marks them like this)
        if line.startswith("new file mode"):
            current_file_is_new = True
            new_files += 1

        # Count added / removed lines
        # Skip diff metadata lines
        if line.startswith('+++') or line.startswith('---') or line.startswith('diff --git') or line.startswith('@@'):
            continue

        # Added lines (but not "+++" metadata)
        if line.startswith('+') and not line.startswith('+++'):
            added += 1

        # Removed lines (but not "---" metadata)
        if line.startswith('-') and not line.startswith('---'):
            deleted += 1

    return {
        "added_lines": added,
        "deleted_lines": deleted,
        "new_files": new_files
    }

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def main():
    args = get_args()
    remove_ids = set()

    logger.info("Loading datasets...")
    loaded_dataset = []
    for data_fp in args.dataset:
        logger.info(f"Loading: {data_fp}")
        with open(data_fp, "r") as f:
            loaded_dataset += [json.loads(line) for line in f.readlines()]
    dataset_ids = set([d["instance_id"] for d in loaded_dataset])
    logger.info(f"Loaded {len(loaded_dataset)} instances from {len(args.dataset)} file(s)")
    front, tail = os.path.splitext(args.dataset[0])

    if args.filter_mode == "long_edit":
        if not args.folder:
            raise ValueError(
                "Filter mode 'long_edit' requires trajectory folder(s) via -f/--folder. "
                "These should be directories containing .pred files (e.g., experiments/traj/)."
            )
        for folder in args.folder:
            logger.info(f"Processing trajectory folder: {folder}")
            subdirs = os.listdir(folder)
            for inst_id in tqdm(subdirs):
                if inst_id not in dataset_ids:
                    continue
                patch = None
                pred_path = os.path.join(folder, inst_id, f"{inst_id}.pred")
                if not os.path.exists(pred_path):
                    remove_ids.add(inst_id)
                else:
                    try:
                        with open(pred_path, "r") as f:
                            patch = json.load(f)["model_patch"]
                    except Exception as e:
                        remove_ids.add(inst_id)
                if patch:
                    diff_stats = analyze_diff(patch)
                    if args.filter_mode == "long_edit":
                        # Mark trajectories with long patches for removal
                        if diff_stats["added_lines"] + diff_stats["deleted_lines"] > PATCH_EDIT_MAX:
                            remove_ids.add(inst_id)
    elif args.filter_mode == "user_length":
        logger.info("Loading tokenizer (Qwen/Qwen3-8B)...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        logger.info("Tokenizer loaded")
        # This is the penultimate message in all SWE-agent trajectories. We do not count this in tool call outputs.
        TARGET_PREFIX = "OBSERVATION:\nThank you for your work on this issue."
        response_lengths = []
        for data in tqdm(loaded_dataset):
            cur_lengths = 0
            for msg in data["messages"]:
                if msg["role"] == "user":
                    if TARGET_PREFIX in msg["content"]:
                        break
                    cur_lengths += count_tokens(msg["content"])
            # Mark trajectories with average tool call response > TOOL_CALL_TOKEN_MAX for removal
            if cur_lengths / (len(data["messages"]) // 2) > TOOL_CALL_TOKEN_MAX:
                remove_ids.add(data["instance_id"])
    else:
        raise RuntimeError(
            f"Invalid filter mode: '{args.filter_mode}'. "
            f"Must be one of: 'long_edit' (filter by patch size), 'user_length' (filter by avg tool response tokens)."
        )

    logger.info(f"Flagged {len(remove_ids)} instances for removal ({len(remove_ids)/len(loaded_dataset)*100:.1f}% of dataset)")

    new_fp = f"{front}_filter_{args.filter_mode}{tail}"
    kept_data = 0
    if os.path.exists(new_fp):
        raise FileExistsError(f"Output file already exists: {new_fp}. Remove it before running again.")
    with open(new_fp, "w") as f:
        for d in loaded_dataset:
            if d["instance_id"] not in remove_ids:
                f.write(json.dumps(d) + "\n")
                kept_data += 1
    logger.info(f"Filtered dataset: {kept_data} instances ({kept_data/len(loaded_dataset)*100:.1f}% retained)")
    logger.info(f"Saved to {new_fp}")

main()