import numpy as np
import pandas as pd
import cmdstanpy # cmdstanpy.install_cmdstan()
import json
# performance measure
#from policy import dist_free_q, normal_ass_q, interval_div_q, quantile_q
from scipy.stats import norm
def calc_beta_dist_model_loss(betas, dist, model, p, s):
    pf = []
    for beta in betas:
        c = (p - s) * beta + s
        BoundaryIndex = int(len(dist) * 0.8)
        underage_cost = p - c
        overage_cost = c - s
        ratio = underage_cost / (underage_cost + overage_cost)

        train = dist[:BoundaryIndex]
        test = dist[BoundaryIndex:]

        if model == "normal_ass":
            q = dist_free_q(train, p, c, s)
            pf.append(loss(q, test, p, c, s))
            print('normal_ass_q',q)
        elif model == "dist_free":
            q = normal_ass_q(train, p, c, s)
            pf.append(loss_opt(q, test, p, c, s))
            print('dist_free_q', q)
        elif model == "interval_div":
            q = interval_div_q(train, p, c, s)
            pf.append(loss_opt(q, test, p, c, s))
            print('interval_div_q', q)
        elif model == "quantile_q":
            q = quantile_q(train, p, c, s)
            pf.append(loss_opt(q, test, p, c, s))
            print('quantile_q', q)
    return pf

# used in `argmin_lopt_emp` for pointwise loss
# lopt_[] should comply with `optTheta_[], optW_[]`.stan
def lopt_NV(W, input_type = "Y", X = None, Thetastar = None, Y = None):
    '''
    Compute optimization loss (a.k.a cost function)
    Parameters:
        real W: decision e.g. inventory amount
        matrix [n,p] X: predictor
        real Y: outcome e.g. demand
        real profit
        real cost
    Returns:
        real: objective value, $l_opt(W(thetahat, X), y)$
    '''
    profit, cost = 5, 1
    if input_type == "Y":
        return - np.mean([(profit * min(w, y) - cost * w) for w, y in zip(W, Y)])
    elif input_type == "Theta":
        sigma_y = 1
        meanstar = pd.Series(np.dot(X, Thetastar))
        z = (W -  meanstar) / sigma_y
        E_lost_sales = norm.pdf(z) - z * (1 - norm.cdf(z)) # L(z)=ϕ(z)−z(1−Φ(z))
        E_sales = meanstar - E_lost_sales     
        E_leftover = W - E_sales
        E_profit = (profit - cost) * E_sales - cost * E_leftover
        return - np.mean(E_profit)