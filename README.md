# Allentune

*Hyperparameter Search for AllenNLP, powered by RayTune.*

Run distributed, parallel hyperparameter search on GPUs or CPUs. See the associated paper [here](http://arxiv.org).

This library was inspired by https://github.com/ChristophAlt/tuna, thanks to that author for their work!

To get started, clone the repository and run `pip install --editable .`

Then, run `allentune -h`.

## What does Allentune support?

This library is compatible with random and grid search algorithms via Raytune. Support for complex search schedulers (e.g. Hyperband, Median Stopping Rule, Population Based Training) is on the roadmap.

## Setup base training config

See `examples/classifier.jsonnet` as an example of a CNN-based classifier on the IMDB dataset.

## Setup the Search space

See `examples/search_space.jsonnet` as an example of search bounds applied to each hyperparameter of the CNN classifier.

## Run Hyperparameter Search

Example command for 10 samples of random search with the classifier, on 4 GPUs:

```
$ allentune search \
    --experiment-name classifier_search \
    --num-cpus 56 \
    --num-gpus 4 \
    --cpus-per-trial 1 \
    --gpus-per-trial 1 \
    --search_space ./examples/search_space.jsonnet \
    --num-samples 30 \
    --base-config ./examples/classifier.jsonnet
```

To restrict the GPUs you run on, run the above command with `CUDA_VISIBLE_DEVICES=xxx`.

When using allentune with your own allennlp modules, run it with the `--include-package xxx` flag, just like you would when running the `allennlp` command.

The `search` command will output all results of experiments in the specified `--logdir`, default output directory is `$(pwd)/logs/`.

## Generate a report from the search

Generate a dataset of resulting hyperparameter assignments and training metrics, for further analysis:

```
$ allentune report \
    --logdir ./logs/classifier_search/ \
    --performance_metric best_validation_accuracy
```

This will create a file `results.jsonl` in `logs/classifier_search`. Each line has the hyperparameter assignments and resulting training metrics from each experiment of your search.


## Plot expected performance

Plot expected performance as a function of hyperparameter assignments or training duration. For more information on how this plot is generated, check the associated paper [here](http://arxiv.org).

```
allentune plot \
    --data_name IMDB \
    --subplot 1 1 \
    --figsize 10 10 \
    --result-file ./logs/classifier_search/results.jsonl \
    --output-file ./classifier_performance.pdf
```

<div style="text-align:center"> <img src="figs/classifier_performance.png" width="500"></div>

Sample more hyperparameters until this curve converges to some expected validation performance!
