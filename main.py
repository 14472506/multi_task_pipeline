"""
Script Detials:
This is a top level script for executing the training and testing of deep learning models
for specified classifier, instance segmentation, and multi task model training.

Usage: 
    - Specify the path to the experiment config file
    - Set the number of iterations by which the experiment should be executed
    - Specify weather training should take place
    - Specify weather testing should take place
"""
# imports
import argparse
import yaml
from loops import Train, Test

# main function
def main(args):
    """
    Main function for executing the experimental training and desting of 
    deep learning models.
    Args: 
        - (argparser.Namespace): parsing the command line
    """
    # Extract args
    config_root = args.config
    train_flag = args.train
    test_flag = args.test
    iterations = args.iters or 1

    # Load config file
    with open(config_root, 'r') as file:
        config = yaml.safe_load(file)

    # Loop over the number of loop iterations
    for i in range(iterations):
        # Modify experiment subdirectory
        amended_sub_dir = "model_" + str(i)
        config["logs"]["sub_dir"] = amended_sub_dir

        # Execute training
        if train_flag:
            trainer = Train(config)
            trainer.train()

        # Execute testing
        if test_flag:
            tester = Test(config)
            tester.test()

# excecution
if __name__ == "__main__":
    """
    Entry point of script:
        The following code defines the args parser, gathers the provided arguments, 
        them passes them to the main function
    """
    # Init parser
    parser = argparse.ArgumentParser(description="Retrieves the key parameters of training and testing models using the framework")

    # Define parsable arguments
    parser.add_argument("-config", type=str, required=True, help="Provide the path to the experiment config")
    parser.add_argument("-iters", type=int, default=1, help="Specify a number of execution iterations")
    parser.add_argument("-train", action="store_true", help="Specify if training should be executed")
    parser.add_argument("-test", action="store_true", help="Specify if testing should be executed")

    # Get parsed arguments
    args = parser.parse_args()

    # Call main
    main(args)