from allentune.modules import AllenNlpRunner, RayExecutor
import pytest
import argparse
import os
import shutil
import pathlib
class TestExampleRun(object):
    
    def test_run(self):
        runner = AllenNlpRunner()
        executor = RayExecutor(runner)
        args = argparse.Namespace()
        PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()  # pylint: disable=no-member
        MODULE_ROOT = PROJECT_ROOT / "allentune"
        TESTS_ROOT = MODULE_ROOT / "tests"
        FIXTURES_ROOT = TESTS_ROOT / "fixtures"
        args.experiment_name = "test"
        args.num_cpus = 1
        args.num_gpus = 0
        args.cpus_per_trial = 1
        args.gpus_per_trial = 0
        args.base_config = FIXTURES_ROOT / "classifier.jsonnet"
        args.search_space = FIXTURES_ROOT / "search_space.json"
        args.log_dir = TESTS_ROOT / "logs"
        args.num_samples = 1
        args.with_server = False
        args.server_port = 1000
        args.search_strategy = "variant-generation"
        executor.run(args)
        assert os.path.isdir(TESTS_ROOT / "logs")
        shutil.rmtree(TESTS_ROOT / "logs/")