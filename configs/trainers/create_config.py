# import yaml
from yaml import load, dump
import ruamel.yaml
import sys

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


config = {
    "DATALOADER": {
        "TRAIN_X": {
            "BATCH_SIZE": 256
        },
        "TEST": {
            "BATCH_SIZE": 256
        },
        "NUM_WORKERS": 8
    },

    "OPTIM": {
      # NAME: "sgd"
      # LR: 0.002
        "NAME": "adam",
        "LR": 0.0001,
        "MAX_EPOCH": 10,
        "LR_SCHEDULER": "cosine",
        "WARMUP_EPOCH": 1,
        "WARMUP_TYPE": "constant",
        "WARMUP_CONS_LR": 1e-5
    },

    "TRAIN": {
        "PRINT_FREQ": 5
    },
}

if __name__ == "__main__":
    # with open('model.yaml', 'w') as file:
    #     yaml.dump(config, file, Dumper=MyDumper, default_flow_style=False)
    yaml = ruamel.yaml.YAML()
    yaml.indent(sequence=4, offset=2)
    with open('model.yaml', 'w') as file:
        yaml.dump(config, file)
