import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_file(path, name):
    with open(path) as json_file:
        data = json.load(json_file)
    smoothed = data['nov_vals']
    #smoothed = np.convolve(smoothed, np.ones(20)/20, mode='valid')

    plt.plot(data['nov_vals'], c='cyan', label='novelty')
    #plt.plot(smoothed, c='blue', label='novelty')

    # X = max_so_far(smoothed)
    # plt.scatter(X, smoothed[X], marker='x', c='red', label='calculated rel nov', zorder=10)

    plt.plot(data['rel_nov_vals'], c='orange', label='relative novelty')
    # plt.plot(get_rel_nov_vals(data['nov_vals']), c='red', label='relative novelty')
    plt.plot(data['current_room'], c='green', label='current room')
    plt.legend()
    plt.title(name)
    plt.show()

def max_so_far(nov_vals):
    M = np.maximum.accumulate(nov_vals)
    V = M[1:] > M[:-1]
    I = [x + 1 for x in np.where(V[1:] < V[:-1])[0]]
    return I[1:]

def get_rel_nov_vals(nov_vals, n_l=7):
    def get_rel_nov(i):
        if i < n_l or i >= len(nov_vals) - n_l:
            return 0

        visits_before = nov_vals[i - n_l : i]
        visits_after = nov_vals[i : i + n_l]

        # return np.sum(visits_before) / np.sum(visits_after)
        return max(np.sum(visits_after) - np.sum(visits_before), 0)

    return np.array([get_rel_nov(i) for i in range(len(nov_vals))])

def plot_all():
    TRAJECTORY_LOAD_PATH = "runs/MontezumaRevengeNoFrameskip-v4_Jun26_14-12-29/json_data"
    files = os.listdir(TRAJECTORY_LOAD_PATH)
    files.sort()
    for filename in files:
            plot_file(os.path.join(TRAJECTORY_LOAD_PATH, filename), filename)
plot_all()
# plot_file("runs/MontezumaRevengeNoFrameskip-v4_Jun08_15-19-38/json_data/subgoal_2800_0_1.json", "subgoal_2800_0_1.json" )
