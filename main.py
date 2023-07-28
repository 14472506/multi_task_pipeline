"""
Details
"""
# imports
from pipeline_loops import MainLoop
import json

# init experiment list
exp_list = ["./configs/mask_rcnn_dev_config.json"]*10


count = 1
# looping through list
for exp in exp_list:
    # loading experiment
    with open(exp, "r") as f:
        cfg = json.load(f)

    cfg["logging"]["path"] = cfg["logging"]["path"] + "_" + str(count)

    # setting up loop
    loop =  MainLoop(cfg)
    if cfg["loop"]["train"]:
        loop.train()
        
    if cfg["loop"]["test"]:        
        loop.test()

    #cfg["model"]["load_source"] = "outputs/reduced_dataset_ssl/rotnet_200_50_epoch/rotnet_1_best.pth",
    #cfg["logging"]["path"] = cfg["logging"]["path"] + "_rotnet_" + str(count)
    #loop2 =  MainLoop(cfg)
    #if cfg["loop"]["train"]:
    #    loop2.train()
    

    count += 1


    
    
