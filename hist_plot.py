import csv
import os
import matplotlib.pyplot as plt
import numpy as np

import seaborn


data_path = "MontezumaRevengeNoFrameskip-v4_Oct11_18-55-42"

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
        # exp = row[4] + row[5]
        # if not exp in data.keys():
        #     data[exp] = ([], [])
        # if 'True' in row[2]:
        #     data[exp][0].append(float(row[1]))
        # else:
        #     data[exp][1].append(float(row[1]))


        nov_vals.append(float(row[0]))
        if 'True' in row[2]:
            nov_vals_T.append(float(row[0]))
        else:
            nov_vals_notT.append(float(row[0]))


def plot_data(data, color, **kwargs):
    y, binEdges = np.histogram(data, bins=100)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y/float(len(data)), c=color, **kwargs)


# seaborn.kdeplot(nov_vals)
# plt.show()

seaborn.kdeplot(nov_vals_T)
seaborn.kdeplot(nov_vals_notT)
plt.show()
exit()

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
