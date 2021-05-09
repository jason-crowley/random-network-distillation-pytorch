import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import argparse

def plot_reward():
    with open('int_reward', 'rb') as f:
        pkl = pickle.load(f)

    fig, ax = plt.subplots()
    plt.plot(range(len(pkl)), pkl[:, 0])
    plt.savefig("reward_graph.png") 	

if __name__=="__main__":
    plot_reward()
