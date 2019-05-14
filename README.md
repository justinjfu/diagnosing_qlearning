# diagnosing_qlearning
Diagnosing Q-learning

This repository contains code for [Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250) by Justin Fu*, Aviral Kumar*, Matthew Soh, Sergey Levine.

This includes:
- Tabular/discrete environments useful for debugging deep RL algorithms.
- FQI and Q-iteration solvers.


## Colab Notebook

For those who prefer experimenting in Jupyter Notebooks, our algorithm prototyping notebook is available [here](https://drive.google.com/open?id=177mrb9B4rqNrdTLtZSgPRdCnrhVlJdEx) as a Colab notebook. This notebook contains a gridworld implementation, along with FQI and plotting code.


# Setup
Install dependencies
```
pip install -r requirements.txt
sudo apt-get install python-dev
```

Compile Cython environments (this must be run from the repo root directory)
```
make build
```

Run tests (this must be run from the repo root directory)
```
make test
```

# Running Experiments
Experiment scripts are located in the `scripts` folder. Each script runs a sweep over environments, and various hyperparameter settings across multiple seeds.

For example,
```
python scripts/run_weighted_exact_fqi.py
```

# Plotting
Plotting code is also located in the `scripts` folder. Each plotting script takes as argument the log directory for one of the experiment scripts

For example
```
python plot_exact_fqi.py <path-to-exact-fqi-logs>
```
