from .lstm import LSTMModel

MODEL_PARAMS = {"hidden_size": 150, "num_layers": 3}
TRAINING_PARAMS = {"learning_rate": 1e-5, "epochs": 50, "patience": 5}


def build_model(n_features, n_horizons, **params):
    return LSTMModel(n_features, n_horizons, **params)