import argparse

"""
Test rig

Goals:
- iterate through a long list of datasets/methods/hyperparams
- Save everything that might be useful - this is partially exploratory, i.e. looking for
differences between batch and minibatch GD
- Maybe have a 'run' class? That saves per-iteration values and has plot methods for easily comparing
runs over different measures


Things to save:
  - Experiment spec object, which should contain all data needed to reproduce the experiment
  - Git commit hash and diff
    - label = subprocess.check_output(["git", "describe", "--always"]).strip()
    - subprocess.call(["git diff > " + os.path.join(log_dir, filename_git_diff)], shell=True)
  - Versions of all dependencies??
    - pip freeze > requirements.txt


Experiment spec:
  - Dataset
    - String identifier (name)
  - Method
    - String identifier?
    - Hyperparams
  - Random seed


Working with the experiment spec:
  - DatasetLoader class
    - A big `if/elif/else` statement
  - 

On the server:
  - When you run a command in an SSH session, if you close the session the command will die. So you need to run it using
    something like `tmux`, or `screen`. Look into the details
  - Do we need to worry about not hogging 100% CPU? Can we limit our CPU usage? Do we need to?
    - Might want to chat to Scott Rose about this?

"""

# python test_rig.py --spec_list lots_of_specs.yaml


def run_experiment(experiment_spec):
    results_dir = create_results_directory()
    save_spec(experiment_spec, results_dir)
    save_git_commit_and_diff(results_dir)
    set_seed(experiment_spec.random_seed)

    dataset = _load_dataset(experiment_spec.dataset_name)
    method = _init_method(experiment_spec.method_name, experiment_spec.hyperparams)

    for i in range(experiment_spec.num_iterations):
        method.run_iteration(dataset)
        log_stuff(method, results_dir)


def _load_dataset(self, name):
    if name == 'thingy':
        """Dataset URL: """
        do_stuff('~/data/sajhdaksjdh/')
    elif name == 'other_thing':
        do_other_stuff()


def _init_method(self, name, hyperparams):
    return Method(hyperparams)


parser = argparse.ArgumentParser()
parser.add_argument('spec_list', type=str)
args = parser.parse_args()

experiment_specs = parse_list_of_specs(args.spec_list)

for spec in experiment_specs:
    run_experiment(spec)
