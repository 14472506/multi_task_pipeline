"""
Detials
"""
# import
import yaml
from loops import Train

# other shit.
with open("configs/classification/rotnet_cfg.yaml", "r") as file:
    config = yaml.safe_load(file)

program = Train(config)
program.train()