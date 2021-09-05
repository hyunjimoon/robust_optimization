import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# 모든 분포는 동일한 1) mean (가능하다면 variance), 2) sample 수를 가진다.
np.random.seed(1)

def generator(theta_star, sigma_x, sigma_y, degree, dist, n):
    '''
    Generate data of 1.ground truth (deg=1, norm_dist), 2.(deg>1, norm_dist), 
                     3.(deg = 1, !norm_dist), 3.(deg > 1, !norm_dist)

    Parameters:
        array theta_star: true parameter value
        array sigma_x: array of length p, is the variance of each feature vector dimension, i.e. x_i ~ N(0, sigma_p)
        float sigma_y: noise of outcome sampled as N(y_true, sigma_y)
        int degree: `y_ture|x` str.  #MISSPEC1 degree 1 vs >1
                    y_true = x-linear if degree =1, polynomial degree prop.to amount of model misspecification
        chr dist: `y|y_ture` str. #MISSPEC2 normal_dist vs unif_dist, t_dist, gamma_dist, (TODO mm_dist)
        int n: number of data points to generate
        # x \sim N(0, sigma_x)
        # y|x \sim N(y_true, sigma_y)
        # (x,y) \sim Normal(??)
    Returns:
        np.array X: predictor data of dimension [n, p]
        np.array y: outcome data of dimension [n, 1] = (1,p) * (p, n)
    '''
    # metadata
    p = len(theta_star)
    # Generate predictor: iid MVN with each ith col. predictors share `sigma_x[i]`
    X = np.random.normal(loc = 0, scale = sigma_x, size = [n, p]) # each row is a training point of size p
    def normal_dist(mu,sd, n):
      return np.random.normal(mu, sd, size = n)
    def unif_dist(mu, sd, n):
      return np.random.uniform(mu - np.sqrt(3)*sd, mu + np.sqrt(3)*sd, n)
    if dist == 'unif':
      ys = [P(theta_star @ np.power(X, d).T, np.repeat(sigma_y, n), n) for P in (normal_dist, unif_dist) for d in (1, degree)]
    return X, ys

def gamma_dist(mu, sd, n):
    b = sd ** 2 / mu
    a = (mu / sd) ** 2
    return np.random.gamma(a, b, n)

def unif_dist(mu, sd, n):
   return np.random.uniform(mu - np.sqrt(3)*sd, mu + np.sqrt(3)*sd, n)

def mixture_dist(mu, sd, mu1, sd1, w1, n):
    mu2 = (mu - w1 * mu1) / (1 - w1)
    delta2 = mu2 - mu
    delta1 = mu1 - mu
    sd2 = np.sqrt((sd ** 2 - (w1 * (sd1 ** 2 + delta1 ** 2) - (1 - w1) * (delta2 ** 2))) / (1 - w1))
    print("mu1", mu1, "sigma1", sd1, "mu2", mu2, "sigma2", sd2)
    p = np.random.binomial(1, w1, size=n)
    dist1 = np.random.normal(loc=mu1, scale=sd1, size=n)
    dist2 = np.random.normal(loc=mu2, scale=sd2, size=n)
    ans = [p[i] * dist1[i] + (1 - p[i]) * dist2[i] for i in range(n)]

    plt.hist(ans)
    plt.title("Multimodal Distribution: mu1 = " + str(mu1))
    plt.legend(loc=3, bbox_to_anchor=(1, 0))
    Filename = 'mm_mu1_' + str(mu1) + 'demand'+'.png'
    plt.savefig(f'fig/{Filename}')
    return ans

def mixture_dist_12(mu1, sd1, mu2, sd2, w1, n):
    ans = np.concatenate((mu1 + np.random.randn(int(n*w1)) * sd1, mu2 + np.random.randn(int(n-n*w1))*sd2))
    
    plt.hist(ans)
    plt.title("Multimodal Distribution: mu1 = " + str(mu1))
    plt.legend(loc=3, bbox_to_anchor=(1, 0))
    Filename = 'mm_mu1_' + str(mu1) + 'demand'+'.png'
    plt.savefig(f'fig/{Filename}')
    return ans

def mixture_dist2(mu, sd, mu1, sd1, w1, n):
    mu2 = (mu - w1 * mu1) / (1 - w1)
    delta2 = mu2 - mu
    delta1 = mu1 - mu
    sd2 = np.sqrt((sd ** 2 - (w1 * (sd1 ** 2 + delta1 ** 2) - (1 - w1) * (delta2 ** 2))) / (1 - w1))
    print("mu1", mu1, "sigma1", sd1, "mu2", mu2, "sigma2", sd2)
    p = np.random.binomial(1, w1, size=n)
    dist1 = np.random.normal(loc=mu1, scale=sd1, size=n)
    dist2 = np.random.normal(loc=mu2, scale=sd2, size=n)
    ans = [p[i] * dist1[i] + (1 - p[i]) * dist2[i] for i in range(n)]

    plt.hist(ans)
    plt.title("Multimodal Distribution: mu1 = " + str(mu1))
    plt.legend(loc=3, bbox_to_anchor=(1, 0))
    Filename = 'mm_mu1_' + str(mu1) + 'demand'+'.png'
    plt.savefig(f'fig/{Filename}')
    return ans

def normal_dist(mu,sd, n):
    return np.random.normal(mu, sd, size = n)


def t_dist(mu, sd, n):
    # sd^2 = Var = v / (v-2)
    # v = (2*Var)/(Var-1) = (2*sd^2) / (sd^2-1)
    #  t 분포의 평균은 0이므로 random sampling한 후 mu 만큼 더하여 평행이동시켜줌. 그러면 평균 = mu 가 된다
    # v2 = 2 * (sd ** 2) / (sd ** 2 - 1)
    return np.random.standard_t(df= 2 * (sd ** 2) / (sd ** 2 - 1), size=n) * sd + mu

def exp_dist(mu, sd, n):
    return np.random.exp()
