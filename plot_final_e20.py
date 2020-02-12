import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot

matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)


def parse_log(file_name):
    rounds = []
    accu = []
    loss = []
    sim = []
    weighted_avg=[]
    for line in open(file_name, 'r'):

        search_train_accu = re.search(r'At round (.*) training accuracy: (.*)', line, re.M | re.I)
        if search_train_accu:
            rounds.append(int(search_train_accu.group(1)))
        else:
            search_test_accu = re.search(r'At round (.*) accuracy: (.*)', line, re.M | re.I)
            if search_test_accu:
                accu.append(float(search_test_accu.group(2)))

        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M | re.I)
        if search_loss:
            loss.append(float(search_loss.group(2)))

        search_loss = re.search(r'gradient difference: (.*)', line, re.M | re.I)
        if search_loss:
            sim.append(float(search_loss.group(1)))
        
        weighted_average = re.search(r'At round (.*) weighted average: (.*)', line, re.M | re.I)
        if weighted_average:
            weighted_avg.append(float(weighted_average.group(1)))

    return rounds, sim, loss, accu, weighted_avg


f = plt.figure(figsize=[20, 10])

log = ["iris", "mnist","iris", "femnist", "shakespeare", "sent140_772user"]
titles = ["IRIS", "MNIST","IRIS", "FEMNIST", "Shakespeare", "Sent140"]
rounds = [100, 20, 20, 40, 800]
mus=[1, 1, 1, 0.001, 0.01]
drop_rates=[0, 0.5, 0.8]

sampling_rate = [1, 1, 1, 1, 10]
labels = ['FedAvg', r'FedProx ($\mu$=0)', r'FedProx ($\mu$>0)']

improv = 0

for drop_rate in range(2,3):
    for idx in range(1,3):

        ax = plt.subplot(1, 2, idx)
        #ax = plt.subplot(3, 2, 2*(drop_rate)+idx)

        if drop_rate == 0:
            rounds1, sim1, losses1, test_accuracies1, weighted_averages1 = parse_log(log[idx] + "/fedprox_drop0_mu0")
        else:
            rounds1, sim1, losses1, test_accuracies1, weighted_averages1 = parse_log(log[idx] + "/fedavg_drop"+str(drop_rates[drop_rate]))
        rounds2, sim2, losses2, test_accuracies2, weighted_averages2 = parse_log(log[idx] + "/fedprox_drop"+str(drop_rates[drop_rate])+"_mu0")
        rounds3, sim3, losses3, test_accuracies3, weighted_averages3 = parse_log(log[idx] + "/fedprox_drop"+str(drop_rates[drop_rate])+"_mu" + str(mus[idx]))

        if sys.argv[1] == 'loss':
                plt.plot(np.asarray(rounds1[:len(losses1):sampling_rate[idx]]), np.asarray(losses1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(losses2):sampling_rate[idx]]), np.asarray(losses2)[::sampling_rate[idx]], '--', linewidth=3.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(losses3):sampling_rate[idx]]), np.asarray(losses3)[::sampling_rate[idx]], linewidth=3.0, label=labels[2], color="#17becf")

        elif sys.argv[1] == 'accuracy':
                plt.plot(np.asarray(rounds1[:len(test_accuracies1):sampling_rate[idx]]), np.asarray(test_accuracies1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(test_accuracies2):sampling_rate[idx]]), np.asarray(test_accuracies2)[::sampling_rate[idx]], '--', linewidth=3.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(test_accuracies3):sampling_rate[idx]]), np.asarray(test_accuracies3)[::sampling_rate[idx]], linewidth=3.0, label=labels[2], color="#17becf")
        
        if sys.argv[1] == 'weighted_average':
                plt.plot(np.asarray(rounds1[:len(weighted_averages1):sampling_rate[idx]]), np.asarray(weighted_averages1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(weighted_averages2):sampling_rate[idx]]), np.asarray(weighted_averages2)[::sampling_rate[idx]], '--', linewidth=3.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(weighted_averages3):sampling_rate[idx]]), np.asarray(weighted_averages3)[::sampling_rate[idx]], linewidth=3.0, label=labels[2], color="#17becf")

        plt.xlabel("# Avg Iterations", fontsize=22)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)

        if sys.argv[1] == 'loss':
            plt.ylabel('Training Loss', fontsize=22)
        elif sys.argv[1] == 'accuracy':
            plt.ylabel('Testing Accuracy', fontsize=22)
        if sys.argv[1] == 'weighted_average':
            plt.ylabel('Weighted Average', fontsize=22)
        if drop_rate == 2:
            plt.title(titles[idx], fontsize=30, fontweight='bold')

        ax.tick_params(color='#dddddd')
        ax.spines['bottom'].set_color('#dddddd')
        ax.spines['top'].set_color('#dddddd')
        ax.spines['right'].set_color('#dddddd')
        ax.spines['left'].set_color('#dddddd')
        ax.set_xlim(0, rounds[idx])

        plt.legend(fontsize=22)
#f.legend(frameon=False, loc='lower center', ncol=3, prop=dict(weight='bold'), fontsize=100)  # note: different from plt.legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
f.savefig(sys.argv[1] + "_full.pdf")
