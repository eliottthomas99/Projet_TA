import optuna

import RNN as rnn


def objective(trial, X_train, y_train, y_test, CV=5):
    """
    Objective function to be minimized.
    :param trial: A Trial object to optimize.
    :param X_train: The training data.
    :param y_train: The training labels.
    :param y_test: The test labels.
    :return: The value of the objective function.
    """
    # Generate the hyperparameters

    epochs = trial.suggest_int("epochs", 2, 4)
    batch_size = trial.suggest_int("batch_size", 16, 64)
    embedding_dim = trial.suggest_int("embedding_dim", 8, 64)
    units = trial.suggest_int("units", 128, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    n_neurons = trial.suggest_int("n_neurons", 32, 128)

    print(trial.params)  # print parameters to be tested for this trial

    # Create the model
    rnn_model = rnn.RNN(
                        X_train=X_train,
                        y_train=y_train,
                        y_test=y_test,
                        embedding_dim=embedding_dim,
                        units=units,
                        dropout=dropout,
                        n_neurons=n_neurons,
                        optimize=True
                        )
    accu_mean = 0
    # model training and evaluation
    for k in range(CV):
        opti_accuracy = rnn_model.train(epochs=epochs, batch_size=batch_size, optimize=True)
        accu_mean += opti_accuracy

    accu_mean /= CV
    return accu_mean


def optisearch(X_train, y_train, y_test, n_trials=100, CV=5):
    """
    Create a study object and run the Objective function.
    :param X_train: The training data.
    :param y_train: The training labels.
    :param y_test: The test labels.
    :param n_trials: The number of trials to run.
    :display: The results of the optimization.
    """
    objective_callback = lambda trial: objective(
        trial,
        X_train=X_train,
        y_train=y_train,
        y_test=y_test,
        CV=CV
    )

    study = optuna.create_study(direction="maximize")  # create a study object
    study.optimize(objective_callback, n_trials=n_trials)    # run the study

    print("Number of finished trials: ", len(study.trials))

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
