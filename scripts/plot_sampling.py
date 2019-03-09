import argparse
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from rlutil.logging import log_processor

import scripts.plot_utils as plot_utils

def filter_fn(params):
    if params['sampling_policy'] != 'pi':
        return False
    if params['weighting_scheme'] != 'pi':
        return False
    return True


def main(log_dir):
    all_exps = list(plot_utils.max_itr(log_processor.iterate_experiments(log_dir, filter_fn=filter_fn), n=100000))
    plot_returns_over_time(all_exps)


def reduce_mean_partitions_over_time(partitions, split_key, expected_len=300):
    for partition_key in partitions:
        if isinstance(split_key, (list, tuple)):
            new_params = dict(zip(split_key, partition_key))
        else:
            new_params = {split_key: partition_key}

        exps = partitions[partition_key]
        new_data = {}
        for data_key in exps[0].progress.keys():
            agg_data = [exp.progress[data_key] for exp in exps]
            filtered_agg_data = [data for data in agg_data if len(data)==expected_len]
            new_data[data_key] = scipy.stats.trim_mean(filtered_agg_data, 0.1, axis=0)
        yield log_processor.ExperimentLog(new_params, new_data, None)


def plot_returns_over_time(all_exps):

    # do my processing manually (not using reduce_mean_keys), then arrange into a table
    split_exps = log_processor.partition_params(all_exps, ('layers', 'num_samples'))
    exps = list(reduce_mean_partitions_over_time(split_exps, ('layers', 'num_samples')))
    frame = log_processor.timewise_data_frame(exps, time_min=0, time_max=300)

    frame = log_processor.rename_partitions(frame, plot_utils.ARCH_NAME_MAP, col_key="layers")
    frame = log_processor.rename_values(frame, plot_utils.VALUE_NAME_MAP)

    sns.set(style="whitegrid")
    palette = dict(zip(sorted([float(x) for x in frame['Samples'].unique()]),
                   sns.color_palette("rocket_r", 6)))
    

    frame256 = frame[frame['Architecture'] == '(256, 256)']
    g = sns.relplot(x="Iteration", y="Normalized Returns",
                hue="Samples", 
                palette=palette,
                height=5, aspect=1.5, facet_kws=dict(sharex=False),
                kind="line", legend='brief', data=frame256)
    plt.legend(loc='best')
    plt.show()
    sns.relplot(x="Iteration", y="Normalized Returns",
                hue="Samples", col="Architecture",
                palette=palette,
                #col_order=['Tabular', '(256, 256)', '(16, 16)'],
                col_order=plot_utils.ARCH_ORDER,
                height=5, aspect=1.25, facet_kws=dict(sharex=False),
                kind="line", legend="full", data=frame)
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('log_dir', type=str)
    args = parser.parse_args()
    main(args.log_dir)
