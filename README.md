# Allentune

Hyperparameter Search for AllenNLP, powered by RayTune.

Run distributed, parallel hyperparameter search on GPUs or CPUs. Compatibility with all Raytune search algorithms (e.g. Grid, Random, etc.) and search schedulers (e.g. Hyperband, Median Stopping Rule, Population Based Training)

To get started, clone the repository and run `pip install --editable .`

Then, run `allentune -h`.


Example command for random search:

```
$ allentune --experiment-name my_random_search --num-cpus 56 --num-gpus 4 --cpus-per-trial 1 --gpus-per-trial 1 -e ./examples/search_space.json --num-samples 2 --base-config ./examples/model.jsonnet --include-package mylib
```