import argparse
import glob
import json
import logging
import os
from collections import ChainMap
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
from allennlp.commands.train import train_model
from allennlp.common.params import Params, parse_overrides, with_fallback
from allennlp.common.util import import_submodules

import _jsonnet
from allentune.util.random_search import HyperparameterSearch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class AllenNlpRunner(object):
    name = "AllenNLP"

    def get_run_func(
        self,
        args: argparse.Namespace,
    ):
        if args is None:
            raise ValueError("No run arguments found for AllenNLP runner.")

        with open(args.base_config, "r") as parameter_f:
            parameter_file_snippet = parameter_f.read()

        def train_func(config, reporter):
            logger.debug(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            
            for package_name in getattr(args, "include_package", ()):
                import_submodules(package_name)

            search_space = HyperparameterSearch(**config)
            sample = search_space.sample()
            for k, v in sample.items():
                config[k] = str(v)
            
            params_dict = json.loads(
                _jsonnet.evaluate_snippet(
                    "config", parameter_file_snippet, tla_codes={}, ext_vars=config
                )
            )
            if args.num_gpus == 0:
                logger.warning(f"No GPU specified, using CPU.")
                params_dict["trainer"]["cuda_device"] = -1

            if args.cpus_per_trial > 0:
                torch.set_num_threads(args.cpus_per_trial)

            params = Params(params_dict)

            logger.debug(f"AllenNLP Configuration: {params.as_dict()}")

            train_model(params=params, serialization_dir="trial")

            reporter(done=True)
            
        return train_func
