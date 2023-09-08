"""
Detials
"""
# import
import yaml
from loops import Train, Test

# other shit.
with open("configs/multi_task/rotmask_cfg.yaml", "r") as file:
    config = yaml.safe_load(file)

program = Test(config)
program.test()