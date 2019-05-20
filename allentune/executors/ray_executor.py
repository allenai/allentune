import os
import argparse
import logging
import json
import ray
import numpy as np
import random

from ray.tune import register_trainable, run_experiments, sample_from, function
from ray.tune.function_runner import StatusReporter
from ray.tune.schedulers import MedianStoppingRule
from allentune.executors import Executor
from allentune.runners import Runner
from typing import Dict, Optional, Any, Callable
from allentune.util.random_search import RandomSearch


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RayExecutor(Executor):
    name = "Ray"

    def __init__(self, runner: Runner) -> None:
        super(RayExecutor, self).__init__(runner)

    def default_argument_parser(self):
        parser = argparse.ArgumentParser(description="Run tuna")

        parser.add_argument(
            "--experiment-name",
            type=str,
            required=True,
            help="a name for the experiment",
        )
        parser.add_argument(
            "--num-cpus",
            type=int,
            default=1,
            help="number of CPUs available to the experiment",
        )
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=1,
            help="number of GPUs available to the experiment",
        )
        parser.add_argument(
            "--cpus-per-trial",
            type=int,
            default=1,
            help="number of CPUs dedicated to a single trial",
        )
        parser.add_argument(
            "--gpus-per-trial",
            type=int,
            default=1,
            help="number of GPUs dedicated to a single trial",
        )
        parser.add_argument(
            "--log-dir",
            type=str,
            default="./logs",
            help="directory in which to store trial logs and results",
        )
        parser.add_argument(
            "--with-server",
            action="store_true",
            default=False,
            help="start the Ray server",
        )
        parser.add_argument(
            "--server-port",
            type=int,
            default=10000,
            help="port for Ray server to listens on",
        )
        parser.add_argument(
            "--search-strategy",
            type=str,
            default="variant-generation",
            help="hyperparameter search strategy used by Ray-Tune",
        )
        parser.add_argument(
            "--search_space",
            "-e",
            type=os.path.abspath,
            required=True,
            help="name of dict describing the hyperparameter search space",
        )
        parser.add_argument(
            "--num-samples",
            type=int,
            default=1,
            help="Number of times to sample from the hyperparameter space. "
            + "If grid_search is provided as an argument, the grid will be "
            + "repeated num_samples of times.",
        )

        return parser

    def parse_search_config(self, search_config: Dict) -> Dict:
        for hyperparameter, val in search_config.items():
            if not isinstance(val, dict):
                ray_sampler = val
            elif val['sampling strategy'] == 'loguniform':
                low, high = val['bounds'][0], val['bounds'][1]
                ray_sampler = function(RandomSearch.random_loguniform(low, high))
            elif val['sampling strategy'] == 'integer':
                low, high = val['bounds'][0], val['bounds'][1]
                ray_sampler = function(RandomSearch.random_integer(low, high))
            elif val['sampling strategy'] == 'choice':
                ray_sampler = function(RandomSearch.random_choice(*val['choices']))
            elif val['sampling strategy'] == 'subset':
                ray_sampler = function(RandomSearch.random_subset(*val['choices']))
            elif val['sampling strategy'] == 'pair':
                ray_sampler = function(RandomSearch.random_pair(*val['choices']))
            elif val['sampling strategy'] == 'uniform':
                low, high = val['bounds'][0], val['bounds'][1]
                ray_sampler = function(RandomSearch.random_uniform(low, high))
            else:
                raise KeyError(f"sampling strategy {val['sampling strategy']} does not exist")
            search_config[hyperparameter] = ray_sampler
        return search_config

    def run_distributed(
        self,
        run_func: Callable[[Dict[str, Any], StatusReporter], None],
        default_args: argparse.Namespace,
        run_args: Optional[argparse.Namespace] = None,
    ) -> None:
        
        logger.info(
            f"Init Ray with {default_args.num_cpus} CPUs "
            + f"and {default_args.num_gpus} GPUs."
        )
        ray.init(num_cpus=default_args.num_cpus, num_gpus=default_args.num_gpus)

        run_func = self._runner.get_run_func(default_args, run_args)
        register_trainable("run", run_func)

        with open(default_args.search_space) as f:
            search_config = json.load(f)

        search_config = self.parse_search_config(search_config)
        experiments_config = {
            default_args.experiment_name: {
                "run": "run",
                "resources_per_trial": {
                    "cpu": default_args.cpus_per_trial,
                    "gpu": default_args.gpus_per_trial,
                },
                "config": search_config,
                "local_dir": default_args.log_dir,
                "num_samples": default_args.num_samples,
            }
        }

        logger.info(f"Run Configuration: {experiments_config}")
        try:
            run_experiments(
                experiments=experiments_config,
                scheduler=None,
                with_server=default_args.with_server,
                server_port=default_args.server_port,
            )

        except ray.tune.TuneError as e:
            logger.error(
                f"Error during run of experiment '{default_args.experiment_name}': {e}"
            )
