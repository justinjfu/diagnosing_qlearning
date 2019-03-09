import argparse
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from rlutil.logging import log_processor
import scripts.plot_utils as plot_utils

def filter_fn(params):
    if params['target_mode'] == 'q*':
        return False
    if params['weight_states_only'] == False:
        return False
    return True

def main(log_dir):
    all_exps = list(plot_utils.max_itr(log_processor.iterate_experiments(log_dir, filter_fn=filter_fn), n=20000))
    plot(all_exps)

def plot(all_exps):
    """ Make a scatterplot of weighting schemes vs returns """
    # Reduce the experiment logs values across **time**
    frame = log_processor.to_data_frame(all_exps, reduce_fn=log_processor.reduce_last)
    # Take the mean of weighting schemes across **experiments**
    frame = log_processor.reduce_mean_keys(frame, col_keys=('weighting_scheme', 'layers'))
     
    # Rename axes to more readable names on the plot
    #frame = log_processor.rename_partitions(frame, plot_utils.WEIGHT_NAME_MAP, col_key="weighting_scheme")
    #frame = log_processor.rename_values(frame, plot_utils.VALUE_NAME_MAP)

    sns.set(style="whitegrid")
    #f, ax = plt.subplots(figsize=(6, 15))
    arch_types = ['tabular', '(256, 256)', '(64, 64)', '(16, 16)', '(4, 4)']#frame['layers'].unique()
    weight_order = ['uniform', 'robust_prioritized', 'buffer_infinite', 'buffer10', 'random', 'pi', 'pi*']

    g = sns.catplot(x="weighting_scheme", y="returns_normalized_50_step_mean", hue="layers", data=frame,
                    hue_order=arch_types,
                    order=weight_order,
                    height=6, kind="bar", palette="muted", legend_out=False)
    g.despine(left=True)

    #colors = sns.mpl_palette("Blues", n_colors=len(arch_types))
    #for color, arch_type in zip(colors, arch_types):
    #    sns.barplot(x="returns_normalized_50_step_mean", y="weighting_scheme", data=frame[frame['layers']==arch_type],
    #                label=arch_type, color=color)

    #ax.legend(ncol=2, loc="lower right", frameon=True)
    #ax.set(xlim=(0, 24), ylabel="",
    #    xlabel="Automobile collisions per billion miles")
    sns.despine(left=True, bottom=True)


    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('log_dir', type=str)
    args = parser.parse_args()
    main(args.log_dir)
