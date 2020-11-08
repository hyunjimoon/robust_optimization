import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.formula.api as smf

#d: demand(pd.series)

def interval_div_q(d, p, c, s):
    def E_cost_I(q, se_I, p, c, s): #구간별 발생확률 동일
        cost = 0
        for interval in se_I:
            start, end = int(interval.left), int(interval.right)
            if q > end:
                cost += (q - start) * (c -s)
            elif q < start:
                cost += (end - q) * (p - c)
            else:
                cost += max((q - start) * (c -s), (end - q) * (p - c))
        return cost / len(se_I)

    L = int(int(d.shape[0]) * 0.3)  # TODO num_I = int(np.log(emp_dist.shape[0])) 등 실험
    se_I = [pd.qcut(d, L, duplicates='drop').values[i] for i in range(L)]

    worst_cand = dict()
    for interval in se_I:
        start, end = int(interval.left), int(interval.right)
        mid = ((c - s) / (p - s)) * start + ((p - c) / (p - s)) * end
        worst_cand[start] = E_cost_I(start, se_I, p, c, s)
        worst_cand[mid] = E_cost_I(mid, se_I, p, c, s)
        worst_cand[end] = E_cost_I(end, se_I, p, c, s)

    return min(worst_cand, key=worst_cand.get)



def normal_ass_q(d, p, c, s):
    underage_cost = p - c
    overage_cost = c - s
    ratio = underage_cost / (underage_cost + overage_cost)

    return np.mean(d) + np.std(d) * norm.ppf(ratio)


def quantile_q(d, p, c, s):
    underage_cost = p - c
    overage_cost = c - s
    ratio = underage_cost / (underage_cost + overage_cost)
    tmp = d.copy()
    tmp['y'] = d
    tmp['ones'] = np.ones(len(d))
    mod = smf.quantreg('y ~ ones', tmp)
    res = mod.fit(q=ratio)
    return res.params['Intercept']


def dist_free_q(d, p, c, s):
    Mean = np.mean(d)
    sigma = np.std(d)
    m = p / c - 1
    dd = 1 - s / c
    return Mean + (sigma / 2) * (np.sqrt(m / dd) - np.sqrt(dd / m))
