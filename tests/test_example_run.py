from allentune.modules import AllenNlpRunner, RayExecutor
import pytest
import argparse
import os
import shutil
class TestExampleRun(object):
    
    def test_run(self):
        runner = AllenNlpRunner()
        executor = RayExecutor(runner)
        args = argparse.Namespace()
        args.experiment_name = "test"
        args.num_cpus = 1
        args.num_gpus = 0
        args.cpus_per_trial = 1
        args.gpus_per_trial = 0
        args.base_config = "/Users/suching/Github/allentune/tests/fixtures/classifier.jsonnet"
        args.search_space = "/Users/suching/Github/allentune/tests/fixtures/search_space.jsonnet"
        args.log_dir = "/Users/suching/Github/allentune/tests/logs"
        args.num_samples = 1
        args.with_server = False
        args.server_port = 1000
        args.search_strategy = "variant-generation"
        executor.run(args)
        assert os.path.isdir("/Users/suching/Github/allentune/tests/logs/")
        shutil.rmtree("/Users/suching/Github/allentune/tests/logs/")