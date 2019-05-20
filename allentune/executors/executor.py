import os
import argparse
import logging

from abc import ABC, abstractmethod
from ray.tune.function_runner import StatusReporter
from allentune.runners import Runner

from typing import List, Dict, Tuple, Optional, Any, Callable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Executor(ABC):
    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    @abstractmethod
    def default_argument_parser(self):
        pass

    @abstractmethod
    def run_distributed(
        self,
        run_func: Callable[[Dict[str, Any], StatusReporter], None],
        default_args: argparse.Namespace,
        run_args: Optional[argparse.Namespace] = None,
    ) -> None:
        pass

    def parse_args(
        self, args: List[str]
    ) -> Tuple[argparse.Namespace, Optional[argparse.Namespace]]:
        default_parser = self.default_argument_parser()
        run_parser = self._runner.get_argument_parser()

        default_args, remaining_args = default_parser.parse_known_args(args)
        run_args = run_parser.parse_args(remaining_args)

        logger.info(f"Runner: {self._runner.name}")
        logger.info(f"Default Arguments: {vars(default_args)}")
        logger.info(f"Run Arguments: {vars(run_args) if run_args else None}")

        return default_args, run_args

    def run(self, args: List[str]) -> None:
        default_args, run_args = self.parse_args(args)
        setattr(default_args, "cwd", os.getcwd())
        run_func = self._runner.get_run_func(default_args, run_args)
        self.run_distributed(run_func, default_args, run_args)
