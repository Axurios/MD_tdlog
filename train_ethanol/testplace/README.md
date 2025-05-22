to complete
first run auxiliaire.py (to create the folders with different sets of hyperparams)
(might be useful to change some of them, or add different possibilities)

# Change this part in auxiliarie.py
if __name__ == "__main__":
    hyperparam_options = {
        "features": [16, 32],
        "max_degree": [2, 3],
        "learning_rate": [0.001, 0.01],
        "num_epochs": [10],
        "batch_size": [30],

        "num_iterations" : [3],
        "num_basis_functions" : [32],
        "cutoff" : [3.0],
        "num_train" : [200],
        "num_valid" : [25],
        "forces_weight" : [1.0],
        "run_num_train":900,
        "run_num_valid":100,
        "temperature" : 1000
    }
    run_all_combinations(hyperparam_options, use_xml=True)


then for each folder
    then run "intercept.py"
    then run "run.py"
they will return two xml with name describing the hyperparameters for each

then extract, all the xml files.
