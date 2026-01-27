"""
Scale down datasets using various selection strategies.

Reduces a large JSONL dataset to a target size using one of three methods:
  - tokens: Select by truncation ratio (prioritizes complete trajectories)
  - repo:   Select by repository source (adds all instances from entire repos until target reached)
  - random: Random sampling

Usage:
    python scale.py -d <data_file> -t <type> -n <number> [-o <output>] [-th <threshold>] [-nf]

Arguments:
    -d, --dataset       Input JSONL file(s), space-separated
    -t, --type          Scaling strategy: tokens | repo | random
    -n, --number        Target dataset size (or proportion if 0 < n < 1)
    -th, --threshold    Min truncation ratio for token scaling (optional)
    -o, --output-file   Output filename (default: <type>_<number>.jsonl)
    -nf, --no-filter    Skip automatic filtering of data > 32768 tokens

Example:
    python scale.py -d data.jsonl -t tokens -n 3000 -th 0.8

Notes:
    - Output is saved to the same directory as the input file
    - Automatic token filtering is applied unless using -t tokens or -nf
"""

import argparse
import json
import logging
import os
import random

from sera.utils import filter_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', nargs="+")
    parser.add_argument('-t', '--type')
    parser.add_argument('-n', '--number', type=float)
    parser.add_argument('-th', '--threshold', type=float)
    parser.add_argument('-o', '--output-file')
    parser.add_argument('-nf', '--no-filter', action="store_true")
    return parser.parse_args()

def scale_repos(dataset, number):
    """
    This scales down by sorting the dataset into each constituent repository and then
    adding all data from one repository at a time until the dataset reaches `number`.
    """
    new_dataset = []
    repo_to_data = {}
    for data in dataset:
        repo_name = "_".join(data["instance_id"].split("_"))[:-1]
        if repo_name not in repo_to_data:
            repo_to_data[repo_name] = []
        repo_to_data[repo_name].append(data)
    for repo_name in repo_to_data:
        logger.info(f"Repository '{repo_name}': {len(repo_to_data[repo_name])} instances")
    repo_order = list(repo_to_data.keys())
    random.shuffle(repo_order)
    for repo_name in repo_order:
        logger.info(f"Adding {len(repo_to_data[repo_name])} instances from '{repo_name}'")
        if len(new_dataset) < number:
            new_dataset += repo_to_data[repo_name]
        else:
            break
    return new_dataset

def scale_tokens(dataset, number, threshold=None):
    """
    This scales down by sorting trajectories by truncation ratio and taking the first `number`
    trajectories with the highest truncation ratios.

    Alternatively, `threshold` can be passed that will cut truncation short if a certain truncation
    ratio is reached.
    """
    logger.info(f"Truncation ratio threshold: {threshold}")
    _, token_to_data_tuples = filter_messages(dataset, truncate=True, return_token_to_data_tuples=True)
    one_count = 0
    random.shuffle(token_to_data_tuples)
    for tup in token_to_data_tuples:
        if tup[0] == 1:
            one_count += 1
    number = min(number, len(token_to_data_tuples))
    logger.info(f"Fully included trajectories (ratio=1.0): {one_count}")
    sorted_token_to_tuples = sorted(token_to_data_tuples, key=lambda x: x[0], reverse=True)
    logger.info(f"Truncation ratio range: {sorted_token_to_tuples[0][0]:.3f} to {sorted_token_to_tuples[number-1][0]:.3f}")
    if not threshold:
        new_dataset = [seq for _, seq in sorted_token_to_tuples][:number]
    else:
        logger.info(f"Applying threshold filter: keeping only trajectories with ratio >= {threshold}")
        new_dataset = []
        for ratio, seq in sorted_token_to_tuples:
            if ratio >= threshold:
                new_dataset.append(seq)
        new_dataset = new_dataset[:number]
    return new_dataset

def main():
    args = get_args()
    total_ds = []
    for dataset_fp in args.dataset:
        with open(dataset_fp, "r") as f:
            ds = [json.loads(line) for line in f.readlines()]
            if not args.no_filter and not args.type == "tokens": 
                ds = filter_messages(ds)
            total_ds += ds
    if args.number > len(total_ds):
        args.number = len(total_ds)
    elif args.number > 0 and args.number < 1:
        args.number = args.number * len(total_ds)
        logger.info(f"Interpreting number as proportion: {args.number / len(total_ds):.1%} of dataset")
    number = int(args.number)
    logger.info(f"Target selection: {number} instances from {len(total_ds)} total")
    scaled_ds = None
    random.shuffle(total_ds)
    if args.type == "repo":
        scaled_ds = scale_repos(dataset=total_ds, number=number)
        number = len(scaled_ds)
    elif args.type == "tokens":
        scaled_ds = scale_tokens(dataset=total_ds, number=number, threshold=args.threshold)
        number = len(scaled_ds)
    elif args.type == "random":
        scaled_ds = random.sample(total_ds, k=number)
    else:
        raise RuntimeError(
            f"Invalid scaling strategy: '{args.type}'. "
            f"Must be one of: 'random' (random sampling), 'tokens' (by truncation ratio), 'repo' (by repository source)."
        )
    if args.output_file:
        fp = f"{args.output_file}.jsonl"
    else:
        fp = f"{args.type}_{number}.jsonl"
    save_dir = os.path.dirname(args.dataset[0])
    fp = os.path.join(save_dir, fp)
    if scaled_ds:
        logger.info(f"Scaled dataset size: {len(scaled_ds)} instances")
        if os.path.exists(fp):
            raise FileExistsError(f"Output file already exists: {fp}. Remove it or specify a different output name with -o.")
        with open(fp, "w") as f:
            for data in scaled_ds:
                f.write(json.dumps(data) + "\n")
        logger.info(f"Saved to {fp}")

main()