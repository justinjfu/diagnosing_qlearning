# diagnosing_qlearning
Diagnosing Q-learning

Code for [Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250) by Justin Fu*, Aviral Kumar*, Matthew Soh, Sergey Levine.

# Setup
Install dependencies
```
pip install -r requirements.txt
sudo apt-get install python-dev
```

Download and setup [rlutil](https://github.com/justinjfu/rlutil). Add it to your pythonpath.


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
python scripts/doodad_exact_fqi.py
```

# Plotting
Plotting code is also located in the `scripts` folder. Each plotting script takes as argument the log directory for one of the experiment scripts

For example
```
python plot_exact_fqi.py <path-to-exact-fqi-logs>
```
