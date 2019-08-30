import argparse
import datetime
import glob
import json
import os
from collections import ChainMap
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from allentune.commands.subcommand import Subcommand

sns.set_style("white")

class Plot(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(
                name, description="generate report from experiment", help='Plot expected validation accuracy curves.')
        subparser.add_argument(
            "--result-file",
            type=str,
            required=True
        )
        subparser.add_argument(
            "--output-file",
            type=str,
            required=True
        )
        subparser.add_argument(
            "--linestyle",
            type=str,
            required=False,
            default="-"
        )
        subparser.add_argument(
            "--logx",
            action="store_true"
        )
        subparser.add_argument(
            "--duration-field",
            type=str,
            required=False,
            default="training_duration"
        )
        subparser.add_argument(
            "--performance-metric-field",
            type=str,
            required=False,
            default="best_validation_accuracy"
        )
        subparser.add_argument(
            "--model-field",
            type=str,
            required=False,
            default="model"
        )
        subparser.add_argument(
            "--plot-errorbar",
            action="store_true"
        )
        subparser.add_argument(
            "--show-xticks",
            action="store_true"
        )
        subparser.add_argument(
            "--legend-location",
            type=str,
            required=False,
            default="lower right"
        )
        subparser.add_argument(
            "--x-axis-time",
            action="store_true"
        )
        subparser.add_argument(
            "--linewidth",
            type=int,
            required=False,
            default=3
        )
        subparser.add_argument(
            "--relabel-logx-scalar",
            type=list,
            required=False,
            default=None
        )
        subparser.add_argument(
            "--x-axis-rot",
            type=float,
            required=False,
            default=0.0
        )
        subparser.add_argument(
            "--data-name",
            type=str,
            required=True,
        )
        subparser.add_argument(
            '--performance-metric',
            required=False,
            type=str,
            default="accuracy"
        )
        subparser.add_argument(
            "--fontsize",
            type=int,
            required=False,
            default=24
        )
        subparser.add_argument(
            "--subplots",
            nargs=2,
            type=int,
            required=True
        )
        subparser.add_argument(
            "--figsize",
            nargs=2,
            type=int,
            required=True
        )
        subparser.set_defaults(func=plotter)
        return subparser



def _cdf_with_replacement(i,n,N):
    return (i/N)**n

def _cdf_without_replacement(i,n,N):
    return scipy.special.comb(i,n) / scipy.special.comb(N,n)

def _compute_variance(N, cur_data, expected_max_cond_n, pdfs):
    """
    this computes the standard error of the max.
    this is what the std dev of the bootstrap estimates of the mean of the max converges to, as
    is stated in the last sentence of the summary on page 10 of http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf
    uses equation 
    """
    variance_of_max_cond_n = []
    for n in range(N):
        # for a given n, estimate variance with \sum(p(x) * (x-mu)^2), where mu is \sum(p(x) * x).
        cur_var = 0
        for i in range(N):
            cur_var += (cur_data[i] - expected_max_cond_n[n])**2 * pdfs[n][i]
        cur_var = np.sqrt(cur_var)
        variance_of_max_cond_n.append(cur_var)
    return variance_of_max_cond_n
    

# this implementation assumes sampling with replacement for computing the empirical cdf
def samplemax(validation_performance, with_replacement=True):
    validation_performance = list(validation_performance)
    validation_performance.sort()
    N = len(validation_performance)
    pdfs = []
    for n in range(1,N+1):
        # the CDF of the max
        F_Y_of_y = []
        for i in range(1,N+1):
            if with_replacement:
                F_Y_of_y.append(_cdf_with_replacement(i,n,N))
            else:
                F_Y_of_y.append(_cdf_without_replacement(i,n,N))

        f_Y_of_y = []
        cur_cdf_val = 0
        for i in range(len(F_Y_of_y)):
            f_Y_of_y.append(F_Y_of_y[i] - cur_cdf_val)
            cur_cdf_val = F_Y_of_y[i]
        
        pdfs.append(f_Y_of_y)

    expected_max_cond_n = []
    for n in range(N):
        # for a given n, estimate expected value with \sum(x * p(x)), where p(x) is prob x is max.
        cur_expected = 0
        for i in range(N):
            cur_expected += validation_performance[i] * pdfs[n][i]
        expected_max_cond_n.append(cur_expected)


    var_of_max_cond_n = _compute_variance(N, validation_performance, expected_max_cond_n, pdfs)

    return {"mean":expected_max_cond_n, "var":var_of_max_cond_n, "max": np.max(validation_performance)}

def td_format(td_object):
    seconds = int(td_object.total_seconds())
    periods = [
        ('yr',        60*60*24*365),
        ('mo',       60*60*24*30),
        ('d',         60*60*24),
        ('h',        60*60),
        ('min',      60),
        ('sec',      1)
    ]
    strings=[]
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value , seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 and period_name not in ['min', 'sec', 'd', 'h'] else ''
            strings.append("%s%s%s" % (period_value, period_name, has_s))
    res = ", ".join(strings)
    if res == '60min':
        res = '1h'
    elif res == '24h':
        res = '1d'
    elif res == '30d':
        res = '1mo'
    return res

def _one_plot(
            data: pd.DataFrame,
            avg_time: pd.DataFrame,
            data_size: int,
            cur_ax: matplotlib.axis,
            data_name: str = "SST5",
            linestyle: str = "-",
            linewidth: int = 3,
            logx: bool = False,
            plot_errorbar: bool = False,
            errorbar_kind: str = 'shade',
            errorbar_alpha: float = 0.1,
            x_axis_time: bool = False,
            legend_location: str = 'lower right',
            relabel_logx_scalar: List[int] = None,
            rename_labels: Dict[str, str] = None,
            reported_accuracy: List[float] = None,
            encoder_name: str = None,
            show_xticks: bool = False,
            fontsize: int = 16,
            xlim: List[int] = None,
            model_order: List[str] = None,
            performance_metric: str = "accuracy",
            x_axis_rot: int = 0,
            line_colors: List[str] = ["#8c564b", '#1f77b4', '#ff7f0e', '#17becf'],
            errorbar_colors: List[str] = ['#B22222', "#089FFF", "#228B22"]):

    cur_ax.set_title(data_name, fontsize=fontsize)
    if model_order:
        models = model_order
    else:
        models = data.index.levels[0].tolist()
        models.sort()
    max_first_point = 0
    cur_ax.set_ylabel("Expected validation " + performance_metric, fontsize=fontsize)
    
    if x_axis_time:
        cur_ax.set_xlabel("Training duration",fontsize=fontsize)
    else:
        cur_ax.set_xlabel("Hyperparameter assignments",fontsize=fontsize)
    
    if logx:
        cur_ax.set_xscale('log')
    
    for ix, model in enumerate(models):
        means = data[model]['mean']
        vars = data[model]['var']
        max_acc = data[model]['max']
        
        if x_axis_time:
            x_axis = [avg_time[model] * (i+1) for i in range(len(means))]
        else:
            x_axis = [i+1 for i in range(len(means))]

        if rename_labels:
            model_name = rename_labels.get(model, model)
        else:
            model_name = model
        if reported_accuracy:
            cur_ax.plot([0, 6.912e+6],
                        [reported_accuracy[model],
                        reported_accuracy[model]],
                        linestyle='--',
                        linewidth=linewidth,
                        color=line_colors[ix])
            plt.text(6.912e+6-3600000,
                    reported_accuracy[model] + 0.01,
                    f'reported {model_name} {performance_metric}',
                    ha='right',
                    style='italic',
                    fontsize=fontsize-5,
                    color=line_colors[ix])

        if encoder_name:
            model_name = encoder_name + " " + model_name

        if plot_errorbar:
            if errorbar_kind == 'shade':
                minus_vars = np.array(means)-np.array(vars)
                plus_vars = [x + y if (x + y) <= max_acc else max_acc for x,y in zip(means, vars)]
                plt.fill_between(x_axis,
                                    minus_vars,
                                    plus_vars,
                                    alpha=errorbar_alpha,
                                    facecolor=errorbar_colors[ix])
            else:
                line = cur_ax.errorbar(x_axis,
                                means,
                                yerr=vars,
                                label=model_name,
                                linestyle=linestyle,
                                linewidth=linewidth,
                                color=line_colors[ix])
        line = cur_ax.plot(x_axis,
                            means,
                            label=model_name,
                            linestyle=linestyle,
                            linewidth=linewidth,
                            color=line_colors[ix])
    
    left, right = cur_ax.get_xlim()
    if xlim:
        cur_ax.set_xlim(xlim)
        # cur_ax.xaxis.set_ticks(np.arange(xlim[0], xlim[1]+5, 10))

    for tick in cur_ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in cur_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    plt.locator_params(axis='y', nbins=10)
    if relabel_logx_scalar:
        for axis in [cur_ax.xaxis]:
            axis.set_ticks(relabel_logx_scalar)
            axis.set_major_formatter(ScalarFormatter())
    plt.xticks(rotation=x_axis_rot)
    
    if show_xticks:
        cur_ax.tick_params(which="both", bottom=True)
    if x_axis_time:
        def timeTicks(x, pos):                                                                                                                                                                                                                                                         
            d = datetime.timedelta(seconds=float(x))
            d = td_format(d)
            return str(d)                                                                                                                                                                                                                                                          
        formatter = matplotlib.ticker.FuncFormatter(timeTicks)                                                                                                                                                                                                                         
        cur_ax.xaxis.set_major_formatter(formatter)
    cur_ax.legend(loc=legend_location, fontsize=fontsize)
    
    plt.tight_layout()

def plotter(args: argparse.Namespace):

    config = vars(args)
    subplots = tuple(config.pop("subplots"))
    figsize = tuple(config.pop("figsize"))
    _ = config.pop('func')
    expected_max_performance_data = {}
    average_times = {}
    output_file = config.pop("output_file")
    config = {config.pop("result_file"): config}    
    f, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    if subplots != (1, 1):
        axes_iter = zip(config.items(), np.ndenumerate(axes))
    else:
        axes_iter = zip(config.items(), enumerate([axes]))
    
    for ((data_file, _), (index, _)) in axes_iter:
        duration_field = config[data_file].pop('duration_field')
        model_field = config[data_file].pop('model_field')
        performance_metric_field = config[data_file].pop('performance_metric_field')
        master = pd.read_json(data_file, lines=True)
        data_sizes = [10000]
        for data_size in data_sizes:
            df = master
            avg_time = df.groupby(model_field)[duration_field].mean()
            sample_maxes = df.groupby(model_field)[performance_metric_field].apply(samplemax)
            expected_max_performance_data[data_file] = {data_size: sample_maxes}
            average_times[data_file] = {data_size: avg_time}
            if subplots == (1,1):
                axis = axes
            elif subplots[1] > 1:
                axis = axes[index[0], index[1]]
            else:
                axis = axes[index[0]]
            _one_plot(sample_maxes,
                        avg_time,
                        data_size,
                        axis,
                        **config[data_file])
    print("saving to {}".format(output_file))
    plt.savefig(output_file, dpi=300)
