import optuna

import RNN as rnn


def objective(trial, X_train, y_train, y_test):
    """
    Objective function to be minimized.

    """
    # Generate the hyperparameters

    epochs = trial.suggest_int("epochs", 2, 4)
    batch_size = trial.suggest_int("batch_size", 16, 64)
    embedding_dim = trial.suggest_int("embedding_dim", 8, 64)
    units = trial.suggest_int("units", 128, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step = 0.1)
    n_neurons = trial.suggest_int("n_neurons", 32, 128)

    print(trial.params)  # print parameters to be tested for this trial

    rnn_model = rnn.RNN(X_train=X_train, y_train=y_train, y_test=y_test, embedding_dim=embedding_dim, units=units, dropout=dropout, n_neurons=n_neurons, optimize=True)

    # model training and evaluation
    opti_accuracy = rnn_model.train(epochs=epochs, batch_size=batch_size, optimize=True)

    return opti_accuracy


def optisearch(X_train, y_train, y_test, n_trials=100):
    """
    Create a study object and run the Objective function.
    """
    objective_callback = lambda trial: objective(
        trial,
        X_train=X_train,
        y_train=y_train,
        y_test=y_test
    )

    study = optuna.create_study(direction="maximize")  # create a study object
    study.optimize(objective_callback, n_trials=n_trials)    # run the study

    print("Number of finished trials: ", len(study.trials))

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
