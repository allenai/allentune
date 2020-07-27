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

class Merge(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(
                name, description="generate report from experiment", help='Generate a report from hyperparameter search experiments.')
        subparser.add_argument(
            "--input-files",
            nargs="+",
            type=str,
            required=True,
        )
        subparser.add_argument(
            '--output-file',
            type=str,
            required=True,
        )
        subparser.set_defaults(func=merge_reports)
        return subparser

def merge_reports(args: argparse.Namespace):
    dfs = []
    for file in args.input_files:
        dfs.append(pd.read_json(file, lines=True))
    master = pd.concat(dfs, 0)
    
    try:
        os.makedirs(os.path.dirname(args.output_file))
    except FileExistsError:
        logger.error(f"{args.output_file} already exists, aborting.")

    master.to_json(args.output_file, lines=True, orient='records')

    logger.info(f"Merged files in {args.output_file}.")