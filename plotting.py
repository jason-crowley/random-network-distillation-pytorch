import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import argparse
import gzip

STEP_WINDOW_SZ = 100
EP_WINDOW_SZ = 10

def plot_intrinsic_reward():
    with open('int_reward', 'rb') as f:
        pkl = pickle.load(f)

    fig, ax = plt.subplots()
    plt.plot(range(len(pkl)), pkl[:, 0])
    plt.savefig("reward_graph.png")

def load_data(path):
    with gzip.open(path, 'rb') as f:
        try:
            data = pickle.load(f)
            while True:
                batch = pickle.load(f)
                for i in range(len(data)):
                    data[i].extend(batch[i])
        except EOFError:
            pass
    return data

def plot_step_reward(env_id):
    path = f"logs/{env_id}_steprewards.pkl.gz"
    data = load_data(path)
    data = np.apply_along_axis(lambda m: np.convolve(m, np.ones(STEP_WINDOW_SZ)/STEP_WINDOW_SZ, mode='valid'), axis=1, arr=data) 
    data = np.mean(data, axis=0)

    fig, ax = plt.subplots()
    plt.plot(range(len(data)), data)
    plt.savefig(f"plots/{env_id}_steprewards_graph.png")

def plot_episode_reward(env_id):
    path = f"logs/{env_id}_episoderewards.pkl.gz"
    data = load_data(path)
    m = min(map(len, data))
    data = [d[:m] for d in data]

    data = np.apply_along_axis(lambda m: np.convolve(m, np.ones(EP_WINDOW_SZ)/EP_WINDOW_SZ, mode='valid'), axis=1, arr=data) 
    data = np.mean(data, axis=0)

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
