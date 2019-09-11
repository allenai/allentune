from typing import Dict
import argparse
import logging
from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allentune.commands.report import Report
from allentune.commands.search import Search
from allentune.commands.plot import Plot
from allentune.commands.merge import Merge

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ArgumentParserWithDefaults(argparse.ArgumentParser):
    """
    Custom argument parser that will display the default value for an argument
    in the help message.
    """

    _action_defaults_to_ignore = {"help", "store_true", "store_false", "store_const"}

    @staticmethod
    def _is_empty_default(default):
        if default is None:
            return True
        if isinstance(default, (str, list, tuple, set)):
            return not bool(default)
        return False

    @overrides
    def add_argument(self, *args, **kwargs):
        # pylint: disable=arguments-differ
        # Add default value to the help message when the default is meaningful.
        default = kwargs.get("default")
        if kwargs.get("action") not in self._action_defaults_to_ignore and not self._is_empty_default(default):
            description = kwargs.get("help") or ""
            kwargs["help"] = f"{description} (default = {default})"
        super().add_argument(*args, **kwargs)


def main(prog: str = None) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag.
    """
    # pylint: disable=dangerous-default-value
    parser = ArgumentParserWithDefaults(description="Run AllenTune", usage='%(prog)s', prog=prog)

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "search": Search(),
            "report": Report(),
            "plot": Plot(),
            "merge": Merge()
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)

    args = parser.parse_args()
    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()