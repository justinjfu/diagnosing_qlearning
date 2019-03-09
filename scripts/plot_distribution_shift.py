
import argparse
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from rlutil.logging import log_processor
import tqdm

import scripts.plot_utils as plot_utils

def filter_fn(params):
    if params['layers'] != '(256, 256)':
        return False
    if params['target_mode'] == 'q*':
        return False
    if params['weight_states_only'] == True:
        return False
    return True

def max_itr(itr, n=10000):
    n_ = 0
    for i in tqdm.tqdm(itr):
        yield i
        n_ += 1
        if n_>=n:
            break

def main(log_dir):
    all_exps = list(max_itr(log_processor.iterate_experiments(log_dir, filter_fn=filter_fn), n=100000))
    plot_shift_vs_returns(all_exps)


def plot_entropy_vs_returns(all_exps):
    """ Make a scatterplot of weighting schemes vs returns """

    # Reduce the experiment logs values across **time**
    def reducer(values, key=None):
        if key == 'weight_entropy_normalized':
            return np.mean(values[1:])
        else:
            return values[-1] # return last
    frame = log_processor.to_data_frame(all_exps, reduce_fn=reducer)
    # Take the mean of weighting schemes across **experiments**
    frame = log_processor.reduce_mean_key(frame, col_key='weighting_scheme')

    # Rename axes to more readable names on the plot
    frame = log_processor.rename_partitions(frame, plot_utils.WEIGHT_NAME_MAP, col_key="weighting_scheme")
    frame = log_processor.rename_values(frame, plot_utils.VALUE_NAME_MAP)
    frame = log_processor.rename_values(frame, {'weight_entropy_normalized': 'Weight Entropy'})
    #print(frame)
    print(frame['Weighting Scheme'])
    print(frame['Normalized Returns'])
    print(frame["Weight Entropy"])

    sns.set(style="whitegrid")
    ax = sns.scatterplot(y="Normalized Returns", x="Weight Entropy", hue="Weighting Scheme", 
        legend=False,
        data=frame)
    log_processor.label_scatter_points(frame['Weight Entropy'], frame['Normalized Returns'], frame["Weighting Scheme"], ax,
        global_x_offset=0.005,
        offsets={
        })
    plt.savefig('fig.png')
    plt.show()


def plot_shift_vs_returns(all_exps):
    """ Make a scatterplot of weighting schemes vs returns """

    # Reduce the experiment logs values across **time**
    def reducer(values, key=None):
        if key == 'distributional_shift_tv':
            return np.mean(values[1:])
        else:
            return values[-1] # return last
    frame = log_processor.to_data_frame(all_exps, reduce_fn=reducer)
    # Take the mean of weighting schemes across **experiments**
    frame = log_processor.reduce_mean_key(frame, col_key='weighting_scheme')

    # Rename axes to more readable names on the plot
    frame = log_processor.rename_partitions(frame, plot_utils.WEIGHT_NAME_MAP, col_key="weighting_scheme")
    frame = log_processor.rename_values(frame, plot_utils.VALUE_NAME_MAP)
    frame = log_processor.rename_values(frame, {'Distributional Shift (TV)': 'Average Distributional Shift'})
    #print(frame)
    print(frame['Weighting Scheme'])
    print(frame['Normalized Returns'])
    print(frame["Average Distributional Shift"])

    sns.set(style="whitegrid")
    ax = sns.scatterplot(y="Normalized Returns", x="Average Distributional Shift", hue="Weighting Scheme", 
        legend=False,
        data=frame)
    log_processor.label_scatter_points(frame['Average Distributional Shift'], frame['Normalized Returns'], frame["Weighting Scheme"], ax,
        global_x_offset=0.005,
        offsets={
            'Prioritized': (-0.060, -0.03),
            'Replay(10)': (0, -0.00),
            'Random': (0, 0.000)
        })
    plt.savefig('fig.png')
    plt.show()


def plot_tv_loss_over_time(all_exps):

    def normalize_errors(all_exps):
        for exp in all_exps:
            exp.progress['distributional_shift_abs_diff_loss'] = exp.progress['distributional_shift_abs_diff_loss'] / exp.progress['returns_expert']
    normalize_errors(all_exps)

    # plot first 50
    frame = log_processor.timewise_data_frame(all_exps, time_min=1, time_max=50+2)
    frame['iteration'] -= 1  # shift iteration back by 1
    frame = log_processor.reduce_mean_keys(frame, col_keys=('weighting_scheme', 'iteration', 'env_name'))  # this step is slow

    frame = log_processor.rename_partitions(frame, plot_utils.WEIGHT_NAME_MAP, col_key="weighting_scheme")
    frame = log_processor.rename_values(frame, plot_utils.VALUE_NAME_MAP)

    sns.set(style="whitegrid")
    sns.lineplot(x="Iteration", y="Distributional Shift (TV)", hue="Weighting Scheme", data=frame)
    plt.show()
    sns.lineplot(x="Iteration", y="Normalized Loss Shift", hue="Weighting Scheme", data=frame)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('log_dir', type=str)
    args = parser.parse_args()
    main(args.log_dir)
