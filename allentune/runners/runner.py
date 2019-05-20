import argparse

from abc import ABC, abstractmethod

from typing import Optional


class Runner(ABC):
    name = None

    def get_argument_parser(self) -> Optional[argparse.ArgumentParser]:
        return None

    @abstractmethod
    def get_run_func(
        self,
        default_args: argparse.Namespace,
        run_args: Optional[argparse.Namespace] = None,
    ):
        pass
