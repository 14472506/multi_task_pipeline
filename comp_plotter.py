import json
import os
from matplotlib import pyplot as plt
from collections import OrderedDict

# get logs
base_val_source = "outputs/multi_task_dev/DELET_THIS_1/log.json"
comp_val_source = "outputs/reduced_dataset_ssl/full_reduction_Jigsaw_pt_mask_rcnn_1/log.json"

# load comparison data
with open(base_val_source, "rb") as f:
    base_val_data = json.load(f)
with open(comp_val_source, "rb") as f:
    comp_val_data = json.load(f)

figure = plt.figure(figsize=(4,4))
rows = 1
columns = 1
count = 1
base_epochs = list(range(1, len(base_val_data["epochs"])+1)) 
comp_epcohs = list(range(1, len(comp_val_data["epochs"])+1))

ax = figure.add_subplot(rows, columns, count)
count += 1
ax.plot(base_epochs, base_val_data["val_loss"][0:45], label = "Multi Task", color = "#0000FF")
ax.plot(comp_epcohs[0:45], comp_val_data["val_loss"][0:45], label = "Jigsaw", color = "#EE4B2B")
ax.scatter(base_val_data["best_epoch"][-1]+1, base_val_data["best_val"][-1], c = "b", marker = "o")
#ax.scatter(comp_val_data["best_epoch"][-1]+1, comp_val_data["best_val"][-1], c = "r", marker = "o")
ax.scatter(comp_val_data["best_epoch"][-3]+1, comp_val_data["best_val"][-3], c = "r", marker = "o")

ax.set_xlabel("Number of Epochs")
ax.set_ylabel("Loss Value")

plt.title("Series vs Multi Task Training Comparison")
best_base = str(round(base_val_data["best_val"][-1], 3)) + " @ " + str(base_val_data["best_epoch"][-1])
plt.annotate(best_base, (base_val_data["best_epoch"][-1], base_val_data["best_val"][-1]), xytext=((base_val_data["best_epoch"][-1]-12, base_val_data["best_val"][-1]-0.02)))

best_comp = str(round(comp_val_data["best_val"][-1], 3)) + " @ " + str(comp_val_data["best_epoch"][-1])
plt.annotate(best_comp, (comp_val_data["best_epoch"][-1], comp_val_data["best_val"][-1]), xytext=((comp_val_data["best_epoch"][-1]-12, comp_val_data["best_val"][-1]-0.02)))
step_comp = str(round(comp_val_data["best_val"][-3], 3)) + " @ " + str(comp_val_data["best_epoch"][-3])
plt.annotate(step_comp, (comp_val_data["best_epoch"][-3], comp_val_data["best_val"][-3]))
#
figure.legend(loc="right")
plt.savefig('plot5.png')