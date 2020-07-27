import argparse
import json
import logging
import os
import random
from typing import Any, Callable, Dict, Optional

import numpy as np
import ray
from ray.tune import function, register_trainable, run_experiments, sample_from
from ray.tune.function_runner import StatusReporter

from allentune.modules.allennlp_runner import AllenNlpRunner
from allentune.util.random_search import RandomSearch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RayExecutor(object):
    name = "Ray"

    def __init__(self, runner: AllenNlpRunner) -> None:
        self._runner = runner

    def parse_search_config(self, search_config: Dict) -> Dict:
        for hyperparameter, val in search_config.items():
            if not isinstance(val, dict):
                ray_sampler = val
            elif val['sampling strategy'] == 'loguniform':
                low, high = val['bounds'][0], val['bounds'][1]
                ray_sampler = RandomSearch.random_loguniform(low, high)
            elif val['sampling strategy'] == 'integer':
                low, high = val['bounds'][0], val['bounds'][1]
                ray_sampler = RandomSearch.random_integer(low, high)
            elif val['sampling strategy'] == 'choice':
                ray_sampler = RandomSearch.random_choice(val['choices'])
            elif val['sampling strategy'] == 'uniform':
                low, high = val['bounds'][0], val['bounds'][1]
                ray_sampler = RandomSearch.random_uniform(low, high)
            else:
                raise KeyError(f"sampling strategy {val['sampling strategy']} does not exist")
            search_config[hyperparameter] = ray_sampler
        return search_config

    def run_distributed(
        self,
        run_func: Callable[[Dict[str, Any], StatusReporter], None],
        args: argparse.Namespace,
    ) -> None:
        
        logger.info(
            f"Init Ray with {args.num_cpus} CPUs "
            + f"and {args.num_gpus} GPUs."
        )
        ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

        run_func = self._runner.get_run_func(args)
        register_trainable("run", run_func)

        with open(args.search_space) as f:
            search_config = json.load(f)

        search_config = self.parse_search_config(search_config)
        experiments_config = {
            args.experiment_name: {
                "run": "run",
                "resources_per_trial": {
                    "cpu": args.cpus_per_trial,
                    "gpu": args.gpus_per_trial,
                },
                "config": search_config,
                "local_dir": args.log_dir,
                "num_samples": args.num_samples,
            }
        }

        logger.info(f"Run Configuration: {experiments_config}")
        try:
            run_experiments(
                experiments=experiments_config,
                scheduler=None,
                with_server=args.with_server,
                server_port=args.server_port,
            )

        except ray.tune.TuneError as e:
            logger.error(
                f"Error during run of experiment '{args.experiment_name}': {e}"
            )

    def run(self, args: argparse.Namespace) -> None:
        setattr(args, "cwd", os.getcwd())
        run_func = self._runner.get_run_func(args)
        self.run_distributed(run_func, args)