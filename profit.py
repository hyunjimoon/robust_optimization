import numpy as np

# performance measure
from policy import dist_free_q, normal_ass_q, interval_div_q



def calc_beta_dist_model_profit(betas, dist, model, p, s):
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
            pf.append(profit(q, test, p, c, s))
            print('normal_ass_q',q)
        elif model == "dist_free":
            q = normal_ass_q(train, p, c, s)
            pf.append(profit(q, test, p, c, s))
            print('dist_free_q', q)
        elif model == "interval_div":
            q = interval_div_q(train, p, c, s)
            pf.append(profit(q, test, p, c, s))
            print('interval_div_q', q)
    return pf



def profit(q, d, p, c, s):
    underage_cost = np.maximum(p - c, 0.)
    overage_cost = np.maximum(c - s, 0.)

    profit = 0
    for dd in d:
        sales = np.minimum(dd, q)
        leftovers = q - sales
        sales_profit = sales * underage_cost
        leftovers_profit = leftovers * overage_cost
        profit += sales_profit - leftovers_profit
    return profit



