import numpy as np
import cmdstanpy # cmdstanpy.install_cmdstan()
import json
# performance measure
#from policy import dist_free_q, normal_ass_q, interval_div_q, quantile_q

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
def lopt_NV(w, y, profit, cost):
    '''
    Compute optimization loss (a.k.a cost function)
    Parameters:
        real w: decision e.g. inventory amount
        real y: outcome e.g. demand
        real profit
        real cost
    Returns:
        real: objective value, $l_opt(W(thetahat, X), y)$
    '''
    return profit * np.min(w, yy) - cost * w
    profit, cost = kwargs.values()
    for yy, XX in zip(y, X):
        w = w_theta(P_ybarx, theta, y, XX) # w_theta(y, XX)
        loss += profit * np.min(w, yy) - cost * w
    return loss / len(y)