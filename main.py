"""
main.py
-------------------
Description: This script provides a refactored and structured approach to execute model training and testing
pipelines based on provided configuration files. By utilizing command line arguments, users can specify the
experiment's configuration, thus allowing flexibility in managing different experiments without modifying the
script directly.

Last edited by: Bradley Hurst
-------------------
"""
import argparse
import json
from pipeline_loops import MainLoop
import logging

def validate_config(config):
    # A simple validation function. Add more checks as necessary
    required_sections = ["model", "dataset", "optimizer", "loop", "logging"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing section: {section}")

def main(config_file):
    #logging.basicConfig(filename="pipeline.log", level=logging.INFO)

    # loading experiment
    try:
        with open(config_file, "r") as f:
            cfg = json.load(f)
        validate_config(cfg)
    except Exception as e:
        #logging.error(f"Error loading config file: {str(e)}")
        return
    
    loop = MainLoop(cfg)
    if "train" in cfg["loop"]["actions"]:
        loop.train()
    if "test" in cfg["loop"]["actions"]:        
        loop.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training pipeline.')
    parser.add_argument('--config', required=True, help='Path to the config JSON file.')
    args = parser.parse_args()
    main(args.config)