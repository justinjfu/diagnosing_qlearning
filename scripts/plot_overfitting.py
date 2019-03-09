"""
Use the replay_vs_onpolicy data dir
filter: layers=tabular (no function approx), sampling=pi, num_samples=32
Look at pendulum validation loss over time.
"""
import argparse
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from rlutil.logging import log_processor
import scripts.plot_utils as plot_utils
import scipy.stats
import tqdm
import collections
import pandas

def filter_fn(params):
    if params['layers'] != '(256, 256)':
        return False
    if params['sampling_policy'] != 'pi':
        return False
    return True

def max_itr(itr, n=100):
    n_ = 0
    for i in tqdm.tqdm(itr):
        yield i
        n_ += 1
        if n_>=n:
            break

def main(log_dir):
    all_exps = list(max_itr(log_processor.iterate_experiments(log_dir, filter_fn=filter_fn)))
    plot_coupled_validation_loss(all_exps)

def reduce_mean_over_time(exps, expected_len=300):
    new_data = {}
    for data_key in exps[0].progress.keys():
        agg_data = [exp.progress[data_key] for exp in exps]
        filtered_agg_data = [data for data in agg_data if len(data)==expected_len]
        new_data[data_key] = scipy.stats.trim_mean(filtered_agg_data, 0.1, axis=0)
    return log_processor.ExperimentLog({}, new_data, None)


def plot_coupled_validation_loss(all_exps):

    val_types = ['32', '64', '256', '1024', 'buffer']
    def normalize_errors(all_exps):
        for exp in all_exps:
            for name in val_types:
                key_val = 'tt_norm_validation_loss_sampling_%s' % name
                key_train = 'tt_norm_project_loss%s' % name
                normalized = exp.progress[key_val] / exp.progress['returns_expert']
                exp.progress['overfitting_%s' % name] = normalized
    normalize_errors(all_exps)

    avg_exp = reduce_mean_over_time(all_exps)

    new_rows = collections.defaultdict(list)
    for name in val_types:
        key = 'overfitting_%s' % name
        data = avg_exp.progress[key]
        new_rows['Normalized Validation Loss'].extend(data)
        if name == 'buffer':
            name = 'Replay Buffer'
        new_rows['Sampling Method'].extend([name] * len(data))
        new_rows['Iteration'].extend(np.arange(0, len(data)))
    frame = pandas.DataFrame(new_rows)

    sns.set(style="whitegrid")
    sns.lineplot(x="Iteration", y="Normalized Validation Loss",
             hue="Sampling Method", #style="event",
             data=frame)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('log_dir', type=str)
    args = parser.parse_args()
    main(args.log_dir)
