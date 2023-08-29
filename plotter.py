import json
import os
from matplotlib import pyplot as plt
from collections import OrderedDict

results_list = []
root = "outputs/reduced_dataset_ssl/"
dirname = "full_reduction_Jigsaw_pt_mask_rcnn_1"

#for root, dir, files in os.walk(root):
#for dirname in sorted(dir):
f_root = root + dirname + "/" + "log.json"
with open(f_root, "rb") as f:
    data = json.load(f)      
results_list.append((dirname, data))
 
results_dicts = OrderedDict(results_list)

val = results_dicts["full_reduction_Jigsaw_pt_mask_rcnn_1"]

figure = plt.figure(figsize=(10,5))
rows = 1
columns = 1
h = 50
w = 50
count = 1
#for key, val in results_dicts.items():
#    
epochs = list(range(1, len(val["epochs"])+1))
#   
ax = figure.add_subplot(rows, columns, count)
count += 1
ax.plot(epochs, val["train_loss"], label = "training_total", color = "#0000FF")
ax.plot(epochs, val["val_loss"], label = "val_total", color = "#EE4B2B")
ax.scatter(val["best_epoch"][-1]+1, val["best_val"][-1], c = "g", marker = "o")
ax.scatter(val["best_epoch"][-3]+1, val["best_val"][-3], c = "g", marker = "o")

ax.set_xlabel("Number of Epochs")
ax.set_ylabel("Loss Value")

plt.title("Multi-task RotNet Mask R-CNN bs1, sslbs4")
best_label = str(round(val["best_val"][-1], 3)) + " @ " + str(val["best_epoch"][-1])
plt.annotate(best_label, (val["best_epoch"][-1], val["best_val"][-1]))
step_label = str(round(val["best_val"][-3], 3)) + " @ " + str(val["best_epoch"][-3])
plt.annotate(step_label, (val["best_epoch"][-3], val["best_val"][-3]))
#
figure.legend(loc="upper right")
plt.savefig('plot2.png')