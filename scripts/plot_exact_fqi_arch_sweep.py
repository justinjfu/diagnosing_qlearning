import argparse
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from rlutil.logging import log_processor
import tqdm
import scripts.plot_utils as plot_utils

def filter_fn(params):
    if params['weighting_scheme'] != 'uniform':
        return False
    return True


def double_plot(frame, plot_keys, split_key='split_key', plot_name='a'):
    data = {'plot_val': [], plot_name: [], 'split_key': []}
    for plot_key in plot_keys.keys():
        plot_key_data = frame[plot_key]
        data['plot_val'].extend(plot_key_data)
        data[plot_name].extend([plot_keys[plot_key]] * len(plot_key_data))
        data['split_key'].extend(frame[split_key])
    frame = pandas.DataFrame(data=data)
    return frame


def main(log_dir):
    all_exps = list(log_processor.iterate_experiments(log_dir, filter_fn=filter_fn))
    returns_qstar_vs_arch(all_exps)


def returns_qstar_vs_arch(all_exps):
    sns.set(style="whitegrid")

    all_exps = log_processor.filter_params(all_exps, 'weighting_scheme', 'uniform')
    fqi_exps = log_processor.filter_params(all_exps, 'target_mode', 'tq')

    num_diverge = 0
    # normalize errors by expert returns so they are on the same scale across environments
    def normalize_q_errors(all_exps):
        for exp in all_exps:
            if exp.progress['q*_diff_abs_max'][-1] >= exp.progress['returns_expert'][0]*10:
                nonlocal num_diverge
                num_diverge += 1
            exp.progress['q*_diff_abs_max'] = exp.progress['q*_diff_abs_max'] / exp.progress['returns_expert']
            exp.progress['ground_truth_error_max'] = exp.progress['ground_truth_error_max'] / exp.progress['returns_expert']
    normalize_q_errors(fqi_exps)
    print('Num diverge:', num_diverge)
    print('Total exps:', len(fqi_exps))
    print('Fraction diverge:', float(num_diverge)/len(fqi_exps))

    split_wts = log_processor.partition_params(fqi_exps, 'layers')
    frame = log_processor.aggregate_partitions(split_wts, aggregate_fn=log_processor.reduce_trimmed_mean) 

    frame = double_plot(frame, plot_keys={'returns_normalized': "Normalized Returns", 
                                          'q*_diff_abs_max': "FQI Q* Error",
                                          'ground_truth_error_max': "Project Q* Error"},
                                          plot_name='')
    frame = log_processor.rename_partitions(frame, {"tabular": "Tabular"})

    g = sns.catplot(x='split_key', y='plot_val', hue='', data=frame, order=plot_utils.ARCH_ORDER,
        height=4, aspect=1.4, kind='bar', palette='muted', legend_out=False)
    g.despine(left=True)
    g.set_ylabels("Normalized Returns/Q-function Error")
    g.set_xlabels("Architecture")
    g.set(ylim=(-0.1,1))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('log_dir', type=str)
    args = parser.parse_args()
    main(args.log_dir)
