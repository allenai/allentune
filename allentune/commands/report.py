import glob
import os
import argparse
from typing import Optional
from collections import ChainMap
import pandas as pd
import json
from allentune.commands.subcommand import Subcommand

class Report(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(
                name, description="generate report from experiment", help='Run the configuration wizard.')
        subparser.add_argument(
            "--logdir",
            required=True,
        )
        subparser.add_argument(
            '--throttle',
            required=False,
            type=int
        )
        subparser.add_argument(
            '--performance_metric',
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
    experiment_dir = os.path.abspath(args.logdir)
    dirs = glob.glob(experiment_dir + '/run_*/trial/')

    master = []
    for dir in dirs:
        try:
            metric = json.load(open(os.path.join(dir, "metrics.json"), 'r'))
            config = json.load(open(os.path.join(dir, "config.json"), 'r'))
            master.append((metric, config))
        except:
            continue


    master_dicts = [dict(ChainMap(*item)) for item in master]

    df = pd.io.json.json_normalize(master_dicts)
    df['training_duration'] = pd.to_timedelta(df['training_duration']).dt.total_seconds()
    if args.throttle:
        df['dataset_reader.sample'] = args.throttle
    if args.model:
        df['model'] = args.model
    output_file = os.path.join(experiment_dir, "results.jsonl")
    df.to_json(output_file, lines=True, orient='records')
    print("results written to {}".format(output_file))
    print(f"total experiments: {df.shape[0]}")
    print(f"best perf: {df[args.performance_metric].max()}")
