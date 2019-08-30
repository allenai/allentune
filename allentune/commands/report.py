import argparse
import glob
import json
import logging
import os
import re
import sys
from collections import ChainMap
from typing import Optional

import pandas as pd

from allentune.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)
class Report(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(
                name, description="generate report from experiment", help='Generate a report from hyperparameter search experiments.')
        subparser.add_argument(
            "--log-dir",
            required=True,
        )
        subparser.add_argument(
            '--performance-metric',
            required=False,
            type=str
        )
        subparser.add_argument(
            '--model',
            required=False,
            type=str
        )
        subparser.set_defaults(func=generate_report)

        return subparser

def generate_report(args: argparse.Namespace):
    experiment_dir = os.path.abspath(args.log_dir)
    dirs = glob.glob(experiment_dir + '/run_*/trial/')

    master = []
    for dir in dirs:
        try:
            with open(os.path.join(dir, "metrics.json"), 'r') as metrics_file:
                metric = json.load(metrics_file)
            with open(os.path.join(dir, "config.json"), 'r') as config_file:
                config = json.load(config_file)
            with open(os.path.join(dir, "stdout.log"), 'r') as stdout_file:
                stdout = stdout_file.read()
                random_seed = re.search("random_seed = (\d+)", stdout) 
                pytorch_seed = re.search("pytorch_seed = (\d+)", stdout)
                numpy_seed = re.search("numpy_seed = (\d+)", stdout)
            if random_seed:
                seeds = {"random_seed": random_seed.group(1), "pytorch_seed": pytorch_seed.group(1), "numpy_seed": numpy_seed.group(1)}
            else:
                seeds = {"random_seed": None, "pytorch_seed": None, "numpy_seed": None}
            directory = {"directory": dir}
            master.append((metric, config, seeds, directory))
        except:
            continue


    master_dicts = [dict(ChainMap(*item)) for item in master]

    df = pd.io.json.json_normalize(master_dicts)
    try:
        df['training_duration'] = pd.to_timedelta(df['training_duration']).dt.total_seconds()
    except KeyError:
        logger.error(f"No finished experiments found in {args.log_dir}")
        sys.exit(0)
    if args.model:
        df['model'] = args.model
    output_file = os.path.join(experiment_dir, "results.jsonl")
    df.to_json(output_file, lines=True, orient='records')
    print("results written to {}".format(output_file))
    print(f"total experiments: {df.shape[0]}")

    try:
        best_experiment = df.iloc[df[args.performance_metric].idxmax()]
    except KeyError:
        logger.error(f"No performance metric {args.performance_metric} found in results of {args.log_dir}")
        sys.exit(0)
    print(f"best model performance: {best_experiment[args.performance_metric]}")
    print(f"best model directory path: {best_experiment['directory']}")
