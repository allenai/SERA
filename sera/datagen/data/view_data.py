"""
View generated trajectories step-by-step.

Displays randomly sampled instances from a JSONL dataset, showing each message
role, content, and training flags in a readable format.

Usage:
    python view_data.py -d <data_file> [-n <num_samples>]

Arguments:
    -d, --data      Path to JSONL data file
    -n, --number    Number of random instances to view (default: 3)

Example:
    python view_data.py -d trajectories.jsonl -n 5
"""

import argparse
import json
import os
import random
import re
import statistics
import yaml

from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Data file path')
    parser.add_argument('-n', '--number', default=3, type=int)
    return parser.parse_args()

def examine_examples(instances, n_to_view):
    random_instances = random.sample(instances, n_to_view)
    for instance in random_instances:
        print("===============================================")
        if "instance_id" in instance.keys():
            print(f"== INSTANCE {instance['instance_id']} ==")
        print("===============================================")
        if "masking_indices" in instance:
            print(instance["masking_indices"])
        content_list = []
        assistant_step = 0
        msg_field = "messages" if "messages" in instance.keys() else "conversations"
        for message in instance[msg_field]:
            print(f"### Step {assistant_step}, Role: {message['role']}")
            if "train" in message:
                print(f"Train: {message['train']}")
            print(message["content"])
            if message["role"] == "assistant":
                content_list.append(message["content"])
            assistant_step += 1
    print(len(instances))

def main():
    args = get_args()
    with open(args.data, "r") as f:
        instances = [json.loads(line) for line in f.readlines()]
    examine_examples(instances, n_to_view=args.number)
main()