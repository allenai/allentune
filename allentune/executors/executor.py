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
    def run_distributed(
        self,
        run_func: Callable[[Dict[str, Any], StatusReporter], None],
        default_args: argparse.Namespace,
        run_args: Optional[argparse.Namespace] = None,
    ) -> None:
        pass

    def run(self, args: argparse.Namespace) -> None:
        setattr(args, "cwd", os.getcwd())
        run_func = self._runner.get_run_func(args)
        self.run_distributed(run_func, args)
