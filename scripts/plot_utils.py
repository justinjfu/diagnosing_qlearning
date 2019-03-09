
VALUE_NAME_MAP = {
    "returns_normalized_50_step_mean": "Normalized Returns",
    "distributional_shift_tv": "Distributional Shift (TV)",
    "distributional_shift_abs_diff_loss": "Normalized Loss Shift",
    "weighting_scheme": "Weighting Scheme",
    "q*_diff_abs_max": "Normalized Q* Error",
    'iteration': 'Iteration',
    'smooth_target_tau': 'Alpha',
    'layers': 'Architecture',
    'num_samples': 'Samples'
}

WEIGHT_NAME_MAP = {
    'uniform': 'Uniform',
    'pi': 'Pi',
    'pi*': 'Pi*',
    'robust_prioritized': 'Prioritized',
    'random': 'Random',
    'buffer_infinite': 'Replay',
    'buffer10': 'Replay(10)'
}

ARCH_NAME_MAP = {
    'tabular': 'Tabular'
}

ARCH_ORDER = ('Tabular', '(256, 256)', '(64, 64)', '(16, 16)', '(4, 4)')


def max_itr(itr, n=100):
    import tqdm
    if n<0:
        n = float('inf')
    n_ = 0
    for i in tqdm.tqdm(itr):
        yield i
        n_ += 1
        if n_>=n:
            break