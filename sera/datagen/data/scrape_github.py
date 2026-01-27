"""
Scrape GitHub issues for synthetic data generation.

Fetches issue texts (title + body) from a GitHub repository and saves them as
JSON files to ROOT/pr_issues/. These can be used as demonstrations for synthetic
PR generation in the distillation pipeline.

Usage:
    python scrape_github.py -o <org> -n <repo_name> [-c <count>]

Arguments:
    -o, --org       GitHub organization or user name
    -n, --name      Repository name
    -c, --count     Number of issues to scrape (default: 100)

Example:
    python scrape_github.py -o anthropics -n claude-code -c 50

Environment:
    Requires GITHUB_TOKEN or GH_TOKEN to be set for API authentication.

Output:
    Saved to ROOT/pr_issues/<org>_<name>_c<count>.json

Config Example (sera/configs/config_specialize_personal.yaml):
    distill:
      shard: 0
      models:
      - model: openai/GLM-4.5-Air
        url: YOUR_URL
      stage_one_config_name: e2e
      stage_two_config_name: qwen
      sweagent_wrapper_config:
        num_workers: 24
      args:
        pipeline_repo: GENERATED_PATH  # <-- set to output path

CLI Example:
    python sera/main.py distill.args.pipeline_repo=GENERATED_PATH
"""

import argparse
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from sera.constants import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--org')
    parser.add_argument('-n', '--name')
    parser.add_argument('-c', '--count', type=int, default=100)
    return parser.parse_args()

def scrape_issue_texts(
    org: str,
    repo: str,
    n: int,
    out_path: str = "issues.json",
    state: str = "all",
    skip_pr: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch up to N issue texts (title + body) from a GitHub repo and save to JSON.

    Notes:
      - Uses GitHub REST API v3.
      - Filters out pull requests (PRs are returned by /issues unless excluded).
      - token via env var GITHUB_TOKEN / GH_TOKEN.
    """
    if n <= 0:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError(
            "GitHub API authentication required. Set GITHUB_TOKEN or GH_TOKEN environment variable. "
            "Create a token at https://github.com/settings/tokens with 'repo' scope."
        )

    def request_json(url: str):
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", "minimal-issue-scraper")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)

    per_page = 100  # max for GitHub REST list endpoints
    page = 1
    results: List[Dict[str, Any]] = []

    while len(results) < n:
        params = {
            "state": state,          # "open" | "closed" | "all"
            "per_page": per_page,
            "page": page,
            "sort": "created",
            "direction": "desc",
        }
        url = (
            f"https://api.github.com/repos/{urllib.parse.quote(org)}/"
            f"{urllib.parse.quote(repo)}/issues?"
            + urllib.parse.urlencode(params)
        )
        items = request_json(url)
        if not items:
            break

        for it in items:
            # Exclude PRs (they appear in /issues unless filtered)
            if skip_pr and "pull_request" in it:
                continue
            body = it.get("body") or ""
            if body:
                results.append(body)
            if len(results) >= n:
                break

        page += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

def main():
    args = get_args()
    file_dir = ROOT / "pr_issues"
    file_dir.mkdir(exist_ok=True)
    file_name = file_dir / f"{args.org}_{args.name}_c{args.count}"
    results = scrape_issue_texts(org=args.org, repo=args.name, n=args.count, out_path=file_name)
    logger.info(f"Saved {len(results)} issues to {file_name}")

main()