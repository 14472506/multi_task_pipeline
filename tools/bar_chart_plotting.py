"""
Detials - summary bar chart plotting
"""
import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = [
    # baseline
    [0.579, 0.613, 0.580, 0.602, 0.605, 0.602, 0.589, 0.592, 0.584, 0.607],
    [0.603, 0.624, 0.605, 0.609, 0.622, 0.612, 0.596, 0.606, 0.607, 0.618],
    # rotnet full
    [0.617, 0.604, 0.593, 0.582, 0.619, 0.614, 0.609, 0.597, 0.590, 0.592],
    [0.623, 0.622, 0.615, 0.593, 0.630, 0.616, 0.628, 0.616, 0.597, 0.610],
    # jigsaw full
    [0.608, 0.624, 0.591, 0.586, 0.592, 0.613, 0.606, 0.593, 0.607, 0.605],
    [0.623, 0.626, 0.610, 0.604, 0.605, 0.610, 0.614, 0.619, 0.622, 0.616],
    # rotnet mini
    [0.614, 0.602, 0.596, 0.598, 0.592, 0.593, 0.578, 0.622, 0.605, 0.598],
    [0.616, 0.614, 0.617, 0.625, 0.602, 0.604, 0.595, 0.623, 0.613, 0.616],
    # Jigsaw mini
    [0.622, 0.595, 0.601, 0.599, 0.598, 0.615, 0.594, 0.622, 0.596, 0.599],
    [0.635, 0.596, 0.623, 0.615, 0.614, 0.630, 0.618, 0.632, 0.607, 0.612],
    # rot mini poor
    [0.585, 0.590, 0.608, 0.605, 0.610, 0.621, 0.581, 0.614, 0.590, 0.602],
    [0.627, 0.612, 0.615, 0.616, 0.628, 0.627, 0.591, 0.624, 0.604, 0.611],
    # jig mini poor
    [0.619, 0.603, 0.600, 0.621, 0.604, 0.596, 0.612, 0.621, 0.581, 0.605],
    [0.631, 0.611, 0.605, 0.632, 0.602, 0.606, 0.618, 0.631, 0.594, 0.606]
]

# Convert data to numpy array
data_np = np.array(data)

# Calculate means and standard deviations
means = data_np.mean(axis=1)
std_devs = data_np.std(axis=1)

# Number of lists
num_lists = len(data)

# Colors for each set
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'yellow']

fig, ax = plt.subplots()
bar_positions = np.arange(num_lists // 2)

# Create a bar chart with error bars for each set
for i in range(0, num_lists, 2):
    set_idx = i // 2
    major_color = colors[set_idx]
    minor_color = f"{major_color}80"
    offset = 0.15
    
    ax.bar(bar_positions[set_idx] - offset, means[i], yerr=std_devs[i], capsize=5, align='center', width=0.3,
           alpha=0.7, color=major_color, ecolor='black')
    ax.text(bar_positions[set_idx], means[i]//2, means[i], ha = 'center')
    
    ax.bar(bar_positions[set_idx] + offset, means[i+1], yerr=std_devs[i+1], capsize=5, align='center', width=0.3,
           alpha=0.3, color=major_color, ecolor='black')
    ax.text(bar_positions[set_idx], means[i+1]//2, means[i+1], ha = 'center')

# Customize the chart
ax.set_ylabel('Mean Average Precision')
ax.set_xlabel('Pre and Post Step Results')
ax.set_title('Baseline and Self Supervised mAP Results')
ax.set_xticks(bar_positions)
ax.set_xticklabels(['Baseline','RotNet_F','Jigsaw_F','RotNet_R','Jigsaw_R', "RotNet_PR", "Jigsaw_PR"])
ax.yaxis.grid(True)
ax.set_xlim(-0.5, 7)
ax.set_ylim(0.58, 0.64)

# Display the chart
plt.show()
plt.savefig('bar_chart2.png')
