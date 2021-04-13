import numpy as np
from typing import List
from collections import defaultdict
import pdb
from matplotlib import pyplot as plt

class Log():
    def __init__(self):
        self.log_ = defaultdict(dict)
    def __call__(self, epoch:int, data:float, func:str):
        if func == "loss":
            self.log_[epoch]["loss"] = data
        elif func == "acc":
            self.log_[epoch]["acc"] = data
    def plot(self, path_dir="./"):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        epochs = []
        losses = []
        accs = []
        for epoch in self.log_:
            epochs.append(epoch)
            losses.append(self.log_[epoch]["loss"])
            accs.append(self.log_[epoch]["acc"])
        ax1.plot(epochs, accs,'g-')
        ax1.legend(('acc',))
        ax2.plot(epochs, losses,'r-')
        ax2.legend(('loss',))
        
        fig.savefig(path_dir+"acc_loss.png")