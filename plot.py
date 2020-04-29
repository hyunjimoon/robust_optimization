import numpy as np
from matplotlib import pyplot as plt

from policy import dist_free_q, normal_ass_q, interval_div_q
from profit import profit

#
# def plot_beta_profit(betas, dist, model, p, s, **kwargs):
#     pf = []
#     for beta in betas:
#         c = (p - s) * beta + s
#         BoundaryIndex = int(len(dist) * 0.8)
#         underage_cost = p - c
#         overage_cost = c - s
#         ratio = underage_cost / (underage_cost + overage_cost)
#
#         train = dist[:BoundaryIndex]
#         test = dist[BoundaryIndex:]
#
#         if model == "normal_ass":
#             q = dist_free_q(train, ratio)
#             pf.append(profit(q, test, p, c, s))
#         elif model == "dist_free":
#             q = normal_ass_q(train, p, c, s)
#             pf.append(profit(q, test, p, c, s))
#         elif model == "interval_div":
#             q = interval_div_q(train, p, c, s)
#             pf.append(profit(q, test, p, c, s))
#     return pf