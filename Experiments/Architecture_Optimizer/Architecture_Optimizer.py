from train_gpu import *
from Architectures import *
import optuna


MODEL_REGISTRY = {
    "ficnn":      FiCNN,
    "ficnn_reg":  FiCNN_reg,
    "ficrnn":     FiCRNN,
    "ficrnn_gru": FiCRNN_GRU,
    "ficrnn_gru_opt": FiCRNN_GRU_OPT
}

def sample_architecture(trial, config):
    """Sample the FiCRNN_GRU_OPT kwargs. Every parameter has a unique name and a
    fixed distribution; the constraints (kernels stay/shrink, channels stay/grow)
    hold by construction."""
    N = trial.suggest_int("film_layers", config["film_layers"]["MIN"], config["film_layers"]["MAX"])

    def kernel_size(index:int, min:int):
        return trial.suggest_int(f"kernel_size_{index}f",  min, config["kernel_sizes"]["MAX"])

    kernels = [kernel_size(0, config["kernel_sizes"]["MIN"])]
    for i in range(1, N):
        kernels.append(kernel_size(i + 1, kernels[i-1]))

    channels = [trial.suggest_int("film_generator_layers", config["n_chanels"]["MIN"], config["n_chanels"]["MAX"])
                for i in range(N + 1)]

    return dict(
        kernels=kernels,
        channels=channels,
        film_generator_layers=trial.suggest_int("film_generator_layers",
                                                config["film_generator_layers"]["MIN"],
                                                config["film_generator_layers"]["MAX"]),
        classifier_hiden=trial.suggest_int("classifier_hiden",
                                           config["classifier_hiden"]["MIN"],
                                           config["classifier_hiden"]["MAX"]),
        dropouts_film_cnn=trial.suggest_categorical("dropout_film_cnn", config["dropouts"]),
        dropouts_classifier=trial.suggest_categorical("dropout_classifier", config["dropouts"]),
        dropout_film_generator=trial.suggest_categorical("dropout_film_generator", config["dropouts"]),
        window_len=config["window_len"],
    )

def optimize(args):
    with open("config/Architecture_Optimizer_config.yaml") as file:
        import yaml
        config = yaml.safe_load(file)

    def Objective(trial):
        # initialize optimized parameters
        lr = trial.suggest_float("learning_rate", config["lr"]["MIN"], config["lr"]["MAX"])

        film_generator_layers = trial.suggest_int("film_generator_layers", config["classifier_hiden"]["MIN"], config["classifier_hiden"]["MAX"])
        classifier_hiden      = trial.suggest_int("classifier_hiden", config["classifier_hiden"]["MIN"], config["classifier_hiden"]["MAX"])

        dropouts_film_cnn      = trial.suggest_categorical("learning_rate", config["dropout"])
        dropouts_classifier    = trial.suggest_categorical("learning_rate", config["dropout"])
        dropout_film_generator = trial.suggest_categorical("learning_rate", config["dropout"])

        def kernel_size(index:int, min:int = config["kernel_sizes"]["MIN"]):
            return trial.suggest_int(f"kernel_size_{index}f",  min, config["kernel_sizes"]["MAX"])

        kernels = [kernel_size(0)]
        N = trial.suggest_int("film_layers",  config["film_layers"]["MIN"], config["film_layers"]["MAX"])
        for i in range(1, N):
            kernels.append(kernel_size(i + 1, kernels[i-1]))

        channels = [trial.suggest_int("film_generator_layers", config["n_chanels"]["MIN"], config["n_chanels"]["MAX"])
                    for i in range(N + 1)]
