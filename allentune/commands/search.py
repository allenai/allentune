#!/usr/bin/env python
import sys
import logging
import os
import argparse

from allentune.modules import AllenNlpRunner
from allentune.modules import RayExecutor
from allentune.commands.subcommand import Subcommand

if os.environ.get("ALLENTUNE_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL
)


class Search(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(
                name, description="search with RayTune", help='Perform hyperparameter search')

        subparser.add_argument(
            "--experiment-name",
            type=str,
            required=True,
            help="a name for the experiment",
        )
        subparser.add_argument(
            "--num-cpus",
            type=int,
            default=1,
            help="number of CPUs available to the experiment",
        )
        subparser.add_argument(
            "--num-gpus",
            type=int,
            default=1,
            help="number of GPUs available to the experiment",
        )
        subparser.add_argument(
            "--cpus-per-trial",
            type=int,
            default=1,
            help="number of CPUs dedicated to a single trial",
        )
        subparser.add_argument(
            "--gpus-per-trial",
            type=int,
            default=1,
            help="number of GPUs dedicated to a single trial",
        )
        subparser.add_argument(
            "--log-dir",
            type=str,
            default="./logs",
            help="directory in which to store trial logs and results",
        )
        subparser.add_argument(
            "--with-server",
            action="store_true",
            default=False,
            help="start the Ray server",
        )
        subparser.add_argument(
            "--server-port",
            type=int,
            default=10000,
            help="port for Ray server to listens on",
        )
        subparser.add_argument(
            "--search-strategy",
            type=str,
            default="variant-generation",
            help="hyperparameter search strategy used by Ray-Tune",
        )
        subparser.add_argument(
            "--search-space",
            "-e",
            type=os.path.abspath,
            required=True,
            help="name of dict describing the hyperparameter search space",
        )
        subparser.add_argument(
            "--num-samples",
            type=int,
            default=1,
            help="Number of times to sample from the hyperparameter space. "
            + "If grid_search is provided as an argument, the grid will be "
            + "repeated num_samples of times.",
        )
        subparser.add_argument(
            "--base-config",
            dest='base_config',
            required=True,
            type=os.path.abspath,
            help="path to parameter file describing the model to be trained",
        )
        
        subparser.add_argument(
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )
        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )

        subparser.set_defaults(func=search_from_args)

        return subparser


def search_from_args(args: argparse.Namespace):
    runner = AllenNlpRunner()
    executor = RayExecutor(runner)
    executor.run(args)
