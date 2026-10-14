from importlib import import_module

MODELS = {
    "arx": "arx",
    "cnn_lstm": "cnn_lstm",
    "mlp": "mlp",
    "lstm_mlp": "lstm_mlp",
    "lstm": "lstm",
    "lstm3": "lstm3",
}


def get_model(model_name):
    module = import_module(f".{MODELS[model_name]}", package=__package__)
    model = module.build_model(**module.MODEL_PARAMS)
    training_params = module.TRAINING_PARAMS.copy()
    return model, training_params