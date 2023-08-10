"""
Detials
"""
# standard
import argparse
import json
import logging

# local
from tests import LoaderTest

# functions
def validate_config(config):
    # A simple validation function. Add more checks as necessary
    required_sections = ["model", "dataset", "optimizer", "loop", "logging"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing section: {section}")

def main(config_file, test):
    logging.basicConfig(filename="pipeline.log", level=logging.INFO)

    # loading experiment
    try:
        with open(config_file, "r") as f:
            cfg = json.load(f)
        validate_config(cfg)
    except Exception as e:
        logging.error(f"Error loading config file: {str(e)}")
        return

    test = LoaderTest(cfg, "multi_task")
    test.test_loader()

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Pipeline testing.')
    #parser.add_argument('--config', required=True, help='Path to the config JSON file.')
    #parser.add_argument('--test', required=True, help="specify test type")
    #args = parser.parse_args()
    main("configs/multi_task_rotnet_mask_rcnn_config.json", "dataloader")