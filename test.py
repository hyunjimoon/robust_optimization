import itertools
from demand_generator import get_demand
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from profit import profit, calc_beta_dist_model_profit

p = 10
# c = 20 determined by beta
s = 1

mu = 1000
sd = 1000/3
mu1s = np.arange(700, 900, 100).tolist()
n = 1000


############
betas = [0.2, 0.5, 0.8]
models  = ['normal_ass', 'dist_free', 'interval_div']
model = 'interval_div'
############



for mu1 in mu1s:
    fig, ax = plt.subplots(figsize = (12, 9))
    dist = get_demand('mm', mu, sd, n, mu1 = mu1, sd1 = mu1/3, w1 = .75)
    pfs = [[m, calc_beta_dist_model_profit(betas, dist, m, p, s)] for m in models]

    for pf in pfs:
        plt.plot(betas, pf[1], label=str(pf[0]) + " model")
        print(str(pf[0]), "profit: ", pf[1])

    plt.xlabel("Beta")
    plt.ylabel("Profit")
    plt.xlim(0,1)
    plt.ylim(np.min(pf[1]), np.max(pf[1])) #TODO autoscale
    plt.title("Multimodal Distribution: mu1 = " + str(mu1))
    plt.legend(loc=3, bbox_to_anchor=(1, 0))
    Filename = 'mm_mu1_' + str(mu1) + '.png'
    plt.savefig(f'fig/{Filename}')
    plt.show()




