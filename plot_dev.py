# imports
import os
import json
from matplotlib import pyplot as plt

# code
def main():
    # target list
    target_list = ["mrcnn_800x500_bs1_no_awl_map_130",
                   "mrcnn_jigsaw_pt_800x1333_bs1_no_awl_map_130",
                   "rotmask_800x1333_bs1_bs8_no_awl_map_130",
                   "rotmask_800x1333_bs1_bs8_awl_map_130",
                   "mrcnn_jigsaw_pt_800x1333_bs1_awl_map_130"]
    
    legend_tags = {"mrcnn_800x500_bs1_no_awl_map_130" :"mrcnn",
                   "mrcnn_jigsaw_pt_800x1333_bs1_no_awl_map_130" :"jigsaw",
                   "rotmask_800x1333_bs1_bs8_no_awl_map_130" :"rotmask",
                   "rotmask_800x1333_bs1_bs8_awl_map_130" :"rotmask_awl",
                   "mrcnn_jigsaw_pt_800x1333_bs1_awl_map_130" :"jigsaw_awl"}

    # dir path
    dir_path = "outputs/logs"

    # figure init
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # loop through files
    for filename in os.listdir(dir_path):
        if filename in target_list:
            
            # get log root
            log_root = os.path.join(dir_path, filename, "model_0/log.json")

            # load log json
            with open(log_root) as log_file:
                output = json.load(log_file)
            
            # get correct val key format
            if "val_sup_loss" in output.keys():
                val_key = "val_sup_loss"
            else:
                val_key = "val_loss"

            # collect data
            val_data = output[val_key]
            epochs = output["epochs"]
            map_data = output["map"]
            pre_best_val = output["pre_best_val"][-1]
            pre_best_epoch = output["pre_best_epoch"][-1]
            pre_best_map = output["pre_best_map"][-1]
            pre_best_map_epoch = output["pre_best_map_epoch"][-1]
            post_best_val = output["post_best_val"][-1]
            post_best_epoch = output["post_best_epoch"][-1]
            post_best_map = output["post_best_map"][-1]
            post_best_map_epoch = output["post_best_map_epoch"][-1]


            # Plot training and validation loss
            ax1.plot(epochs, val_data, label= legend_tags[filename] + '_val', linewidth = 1)
            ax2.plot(epochs, map_data, label= legend_tags[filename] + '_mAP', linewidth = 1)

            # Mark the best pre and post values
            ax1.scatter([pre_best_epoch, post_best_epoch], [pre_best_val, post_best_val], color='black', s=20)
            ax2.scatter([pre_best_map_epoch, post_best_map_epoch], [pre_best_map, post_best_map], color='black', s=20)

    # Add labels and legend
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax2.set_ylabel("mAP", fontsize=14)

    ax1.tick_params(axis="both", labelsize=14)
    ax2.tick_params(axis="both", labelsize=14)

    # Get handles and labels from both axes
    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    plt.legend(handles, labels, loc="center", fontsize=8)

    plt.title('Training and Validation Loss Over Epochs', fontsize=18)

    # Instead of plt.show(), use plt.savefig()
    plt.savefig("plotter_dev.png", format='png', dpi=150)
    
    # Optionally, you can close the figure after saving
    plt.close()              

if __name__ == "__main__":
    main()