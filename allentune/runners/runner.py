import argparse

from abc import ABC, abstractmethod

from typing import Optional


class Runner(ABC):
    name = None

    @abstractmethod
    def get_run_func(
        self,
        args: argparse.Namespace
    ):
        pass
