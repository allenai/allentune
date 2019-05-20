# Allentune

*Hyperparameter Search for AllenNLP, powered by RayTune.*

This projects allows you to run distributed, parallel hyperparameter search with AllenNLP on GPUs or CPUs. 

Supports all Raytune search algorithms (e.g. Grid, Random, etc.) and search schedulers (e.g. Hyperband, Median Stopping Rule, Population Based Training).

Additionally, plot expected performance of models as a function of computational budget (e.g. number of hyperparameter assignments or training duration).



To get started, clone the repository and run `pip install --editable .`

Then, run `allentune -h`.


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
    --num-samples 50 \
    --base-config ./examples/classifier.jsonnet
```

## Generate a report from the search

Generate a dataset of resulting hyperparameter assignments and training metrics, for further analysis:

```
$ allentune report \
    --logdir ./logs/classifier_search/ \
    --performance_metric best_validation_accuracy
```

This will create a file `results.jsonl` in `logs/classifier_search`. Each line has the hyperparameter assignments and resulting training metrics from each experiment of your search.


## Plot expected performance

Plot expected performance as a function of hyperparameter assignments or training duration.

```
allentune plot \
    --data_name IMDB \
    --subplot 1 1 \
    --figsize 10 10 \
    --result-file ./logs/classifier_search/results.jsonl \
    --output-file ./classifier_performance.pdf
```

![classifier performance](http://github.com/allenai/allentune/figs/classifier_performance.pdf)
