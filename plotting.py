import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import argparse
import gzip

def plot_intrinsic_reward():
    with open('int_reward', 'rb') as f:
        pkl = pickle.load(f)

    fig, ax = plt.subplots()
    plt.plot(range(len(pkl)), pkl[:, 0])
    plt.savefig("reward_graph.png")

def load_data(path):
    data = []
    with gzip.open(path, 'rb') as f:
        try:
            while True:
                data.extend(pickle.load(f))
        except EOFError:
            pass
    return data

def plot_step_reward(env_id):
    #path = "logs/MontezumaRevengeNoFrameskip-v4_steprewards.pkl.gz"
    path = f"logs/{env_id}_steprewards.pkl.gz"
    data = load_data(path)
    #data = data[::5]

    fig, ax = plt.subplots()
    plt.plot(range(len(data)), data)
    plt.savefig(f"plots/{env_id}_steprewards_graph.png")

def plot_episode_reward(env_id):
    #data = "logs/MontezumaRevengeNoFrameskip-v4_episoderewards.pkl.gz"
    path = f"logs/{env_id}_episoderewards.pkl.gz"
    data = load_data(path)

    fig, ax = plt.subplots()
    plt.plot(range(len(data)), data)
    plt.savefig(f"plots/{env_id}_episoderewards_graph.png")

if __name__=="__main__":
    plot_step_reward("MontezumaRevengeNoFrameskip-v4")
    plot_episode_reward("MontezumaRevengeNoFrameskip-v4")
    plot_step_reward("PitfallNoFrameskip-v4")
    plot_episode_reward("PitfallNoFrameskip-v4")
    plot_step_reward("VentureNoFrameskip-v4")
    plot_episode_reward("VentureNoFrameskip-v4")
