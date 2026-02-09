import itertools
import wandb 
import subprocess

def ML_Lambda_valid(Lambda,layers):
    if layers < 2:
        return False
    else:
        if layers <=  2 * Lambda + 1:
            return False
        elif Lambda < 1:
            return False
        else:
            return True
        
def create_sweeps(combinations,bayes_params,grid_keys):
    sweep_ids = []

    combonum = 1
    for combo in combinations:
        sweep_config = SWEEP_TEMPLATE.copy()
        sweep_config["parameters"] = SWEEP_TEMPLATE["parameters"].copy()

        # Add the fixed grid values
        for k, v in zip(grid_keys, combo):
            sweep_config["parameters"][k] = {"value": v}

        # Add the Bayesian/distribution parameters
        for k, v in bayes_params.items():
            sweep_config["parameters"][k] = v

        # Create the sweep
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            entity=ENTITY,
            project=PROJECT
        )

        print(f"{combonum} Created sweep: {sweep_id} with config: {sweep_config}")

        sweep_ids.append(sweep_id)
        combonum += 1

    return sweep_ids
        
if __name__ == "__main__":
    # Set your project and entity
    ENTITY = "suzannastep-university-of-chicago"
    PROJECT = "middle_linear"

    # Grid parameters: all possible combinations
    grid_params = {
        "Lambda": [0, 1, 2, 3],
        "layers": [15, 7, 5, 3, 2],
    }

    # Bayesian/distribution parameters
    bayes_params = {
        "wd": {
            "distribution": "log_uniform_values",
            "min": 1e-8,
            "max": 1e-3
        },
        "clip_grad_norm": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 2
        },
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2
        },
        "optimizer": {
            "values": ["adam", "sgd", "adam_cosine", "sgd_cosine"]
        }
    }

    SWEEP_TEMPLATE = {
        "program": "/home/sueparkinson/deeprelu/super_inrs/phantom_exp/run_exp.py",
        "method": "bayes",
        "metric": {
            "goal": "minimize",
            "name": "ValidateMSE"
        },
        "parameters": {
            "epochs": {
                "value":50000
            }
        }
    }

    # Generate all grid combinations
    grid_keys = list(grid_params.keys())
    combinations = [(0,15),(0,7),(0,5),(0,3),(0,2),(1,15),(2,15),(3,15)]

    sweep_ids = create_sweeps(combinations,bayes_params,grid_keys)