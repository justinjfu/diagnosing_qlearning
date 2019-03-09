import argparse
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import collections
from rlutil.logging import log_processor

import scripts.plot_utils as plot_utils

def filter_fn(params):
    if params['log_sampling_type'] != 'buffer32':
        return False
    return True


def main(log_dir):
    all_exps = list(plot_utils.max_itr(log_processor.iterate_experiments(log_dir, filter_fn=filter_fn), n=100000000))
    plot_errors_over_time(all_exps)


def reduce_mean_partitions_over_time(partitions, split_key, expected_len=300):
    for partition_key in partitions:
        if isinstance(split_key, (list, tuple)):
            new_params = dict(zip(split_key, partition_key))
        else:
            new_params = {split_key: partition_key}

        exps = partitions[partition_key]
        new_data = {}
        for data_key in exps[0].progress.keys():
            if data_key in ['validation_stop_step']:
                continue
            agg_data = [exp.progress[data_key] for exp in exps]
            filtered_agg_data = [data for data in agg_data if len(data)==expected_len]
            new_data[data_key] = scipy.stats.trim_mean(filtered_agg_data, 0.1, axis=0)
        yield partition_key, log_processor.ExperimentLog(new_params, new_data, None)


def plot_errors_over_time(all_exps):

    # do my processing manually (not using reduce_mean_keys), then arrange into a table
    split_exps = log_processor.partition_params(all_exps, ('validation_stop', 'env_name'))
    exps = list(reduce_mean_partitions_over_time(split_exps, ('validation_stop', 'env_name'), expected_len=300))
    exps = dict(exps)
    split_exps = collections.defaultdict(list)
    for partition_key in exps:
        split_exps[partition_key[0]].append(exps[partition_key])
    exps = list(reduce_mean_partitions_over_time(split_exps, 'validation_stop', expected_len=300))
    exps = [exp[1] for exp in exps]
    frame = log_processor.timewise_data_frame(exps, time_min=0, time_max=600)

    frame = log_processor.rename_partitions(frame, {False: 'None', 'returns': 'Oracle Returns', 'bellman': 'Bellman Error'}, col_key="validation_stop")
    frame = log_processor.rename_values(frame, plot_utils.VALUE_NAME_MAP)
    frame = log_processor.rename_values(frame, {'validation_stop':'Stop Method'})

    sns.set(style="whitegrid")
    #import pdb; pdb.set_trace()
    #palette = dict(zip(sorted([float(x) for x in frame['Stop Method'].unique()]), sns.color_palette("rocket_r", 6)))
    

    #frame256 = frame[frame['Architecture'] == '(256, 256)']
    g = sns.relplot(x="Iteration", y="Normalized Returns",
                hue="Stop Method", 
                #palette=palette,
                height=5, aspect=1.5, facet_kws=dict(sharex=False),
                kind="line", legend='brief', data=frame)
    plt.legend(loc='best')
    plt.savefig('fig.png')
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('log_dir', type=str)
    args = parser.parse_args()
    main(args.log_dir)
