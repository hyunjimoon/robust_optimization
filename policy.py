import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.formula.api as smf

#d: demand(pd.series)

def interval_div_q(d, p, c, s):
    def cost(q, d, p, c, s):
        underage_cost = p - c
        overage_cost = c - s
        cost = max((q - d), 0) * overage_cost + max((d - q), 0) * underage_cost
        return cost

    def worst_interval_cand(d, p, c, s):
        I = int(int(d.shape[0]) * 0.3)  #TODO I = int(np.log(emp_dist.shape[0])) 등 실험
        se = [pd.qcut(d, I, duplicates='drop').values[i] for i in range(I)]
        worst_cand_lst = list()
        for interval in se:
            start, end = int(interval.left), int(interval.right)
            mid = ((c - s) / (p - s)) * start + ((p - c) / (p - s)) * end
            worst_cand_lst.append([start, mid, end])
        return np.unique(worst_cand_lst), se

    def E_cost_D(q, d, p, c, s):
        return np.mean([cost(q, i, p, c, s) for i in d])

    def min_E_worst_q(d, p, c, s):
        opt_q = 0
        opt_cost = np.inf
        cand = worst_interval_cand(d, p, c, s)[0]
        for q in cand:
            cand_cost = E_cost_D(q, d, p, c, s)
            if opt_cost > cand_cost:
                opt_cost = cand_cost
                opt_q = q
        return opt_q

    return min_E_worst_q(d, p, c, s)


def normal_ass_q(d, p, c, s):
    underage_cost = p - c
    overage_cost = c - s
    ratio = underage_cost / (underage_cost + overage_cost)

    return np.mean(d) + np.std(d) * norm.ppf(ratio)


def quantile_regression_q(d, p, c, s):
    underage_cost = p - c
    overage_cost = c - s
    ratio = underage_cost / (underage_cost + overage_cost)

    tmp = d.copy()
    tmp['ones'] = np.ones(tmp.shape)
    mod = smf.quantreg('y ~ ones', tmp)
    res = mod.fit(q=ratio)
    print(res.summary())
    return res


def dist_free_q(d, p, c, s):
    Mean = np.mean(d)
    sigma = np.std(d)
    m = p / c - 1
    dd = 1 - s / c
    return Mean + (sigma / 2) * (np.sqrt(m / dd) - np.sqrt(dd / m))
