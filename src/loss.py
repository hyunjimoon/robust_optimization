import numpy as np
import cmdstanpy # cmdstanpy.install_cmdstan()
import json
# performance measure
from policy import dist_free_q, normal_ass_q, interval_div_q, quantile_q



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

def argmin_l_pred_NV(y, X):
  data = {'X': X, 'y':y, 'n': n, 'p': p}
  sm = cmdstanpy.CmdStanModel(stan_file="stan/LinReg.stan")
  fit = sm.sample(data)
  theta1 = fit.stan_variable(name='theta')
  return theta1

def loss_opt_NV(w_thetahat, y, X, thetahat, **kwargs):
    '''
    Compute loss of optimization (a.k.a cost function, objective value)
    Iterate over w gives optimal solution
    Parameters:
        function W_solver: decision function: (Theta, X) -> R, $\hat{W}(thetahat, x)$ 
        function w_thetahat: decision function: X -> R, $\hat{W}(thetahat, x)$ 
        array y: outcome array of dimension [n, 1]
        array X: predictor array of size [n, p]
        array thetahat: assumed model parameter to solve `argmin_l_opt`
        **kwargs: lost function coefficients
    Returns:
        real: objective value, $l_opt(W(thetahat, X), y)$
    '''
    p, c, s = kwargs.values()
    underage_cost = np.maximum(p - c, 0.)
    overage_cost = np.maximum(c - s, 0.)
    for xx, yy in zip(x, y):
        w = w_thetahat(xx)
        W_optmizer(theta, xx, x, y)
        loss += p * np.min(w, yy) - c * w
    return loss / len(y)














