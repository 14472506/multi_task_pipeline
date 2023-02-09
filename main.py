"""
Details
"""
# imports
from pipeline_loops import MainLoop
import json

# init experiment list
exp_list = ["./configs/mask_rcnn_dev_config.json"]


# looping through list
for exp in exp_list:
    # loading experiment
    with open(exp, "r") as f:
        cfg = json.load(f)

    # setting up loop
    loop =  MainLoop(cfg)
    if cfg["loop"]["train"]:
        loop.train()
    if cfg["loop"]["test"]:        
        loop.test()


    
    
