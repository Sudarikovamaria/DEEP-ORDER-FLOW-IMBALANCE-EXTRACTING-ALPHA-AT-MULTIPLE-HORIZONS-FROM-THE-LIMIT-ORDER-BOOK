"""Model registry.

Every model module defines:
    MODEL_PARAMS: dict           architecture hyperparameters (hidden sizes, layers, ...)
    TRAINING_PARAMS: dict        at least "learning_rate" (Table 2); optional "epochs", "patience"
    build_model(n_features, n_horizons, **MODEL_PARAMS) -> torch.nn.Module

The module maps x: Tensor[B, W, F] to Tensor[B, H], where W = 100 events, F = 20 (OF),
40 (LOB) or 10 (OFI), and H = 10 horizons.

A module may also define fit(model, train_loader, val_loader, **TRAINING_PARAMS) -> (model, history)
to replace the default SGD loop in engine.train (ARX uses closed-form ridge).
"""
from importlib import import_module

MODELS = {
    "arx": "arx",
    "cnn_lstm": "cnn_lstm",
    "mlp": "mlp",
    "lstm_mlp": "lstm_mlp",
    "lstm": "lstm",
    "lstm3": "lstm3",
}


def get_model(model_name, n_features=20, n_horizons=10):
    module = import_module(f".{MODELS[model_name]}", package=__package__)
    model = module.build_model(n_features=n_features, n_horizons=n_horizons, **module.MODEL_PARAMS)
    training_params = module.TRAINING_PARAMS.copy()
    return model, training_params


def get_fit(model_name):
    """The model's own fit function, or None if it trains with engine.train."""
    module = import_module(f".{MODELS[model_name]}", package=__package__)
    return getattr(module, "fit", None)
