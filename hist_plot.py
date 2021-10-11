import csv
import os
import matplotlib.pyplot as plt
import numpy as np


positive_diff_5 = "MontezumaRevengeNoFrameskip-v4_Sep23_20-16-57"
# ratio = "MontezumaRevengeNoFrameskip-v4_Sep23_19-32-31"
ratio_5 = "MontezumaRevengeNoFrameskip-v4_Sep24_05-33-58"

positive_diff_2 = "MontezumaRevengeNoFrameskip-v4_Sep24_12-53-19"
ratio_2 = "MontezumaRevengeNoFrameskip-v4_Sep24_13-00-22"

data_path = "MontezumaRevengeNoFrameskip-v4_Oct03_17-42-29"

folder_path = os.path.join(os.getcwd(), "runs", data_path)
path = os.path.join(folder_path, "hist_data.csv")

fig_save_path = os.path.join(folder_path, "figs")

if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

data = {}
nov_vals = []
nov_vals_T = []
nov_vals_notT = []


with open(path) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        exp = row[4] + row[5]
        if not exp in data.keys():
            data[exp] = ([], [])
        if 'True' in row[2]:
            data[exp][0].append(float(row[1]))
        else:
            data[exp][1].append(float(row[1]))


        # nov_vals.append(float(row[0]))
        # if 'True' in row[2]:
        #     nov_vals_T.append(float(row[0]))
        # else:
        #     nov_vals_notT.append(float(row[0]))


def plot_data(data, color, **kwargs):
    y, binEdges = np.histogram(data, bins=100)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y/float(len(data)), c=color, **kwargs)


# plot_data(nov_vals, 'blue')
# plt.show()
#
# plot_data(nov_vals_T, 'blue', label="target")
# plot_data(nov_vals_notT, 'red', label="non-target")
# plt.legend()
# plt.show()
# exit()

for exp in data.keys():
    plot_data(data[exp][0], 'blue', label="target")
    plot_data(data[exp][1], 'red', label="non-target")
    plt.title(exp)
    plt.legend()
    plt.savefig(os.path.join(fig_save_path,exp))
    # plt.show()
    plt.cla()
