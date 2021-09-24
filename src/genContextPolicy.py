import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm
import matplotlib.pyplot as plt
import logging
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn import mixture
## 3. Policies with generated predictors (unsupervised)
def fit_gmm_normalize_return_comp(df):
    dfy = df.y.copy()
    y = dfy #scaling todo (dfy - np.mean(dfy))/np.std(dfy) https://stackoverflow.com/questions/13161923/python-sklearn-mixture-gmm-not-robust-to-scale
    X = np.expand_dims(y, 1)
    N = np.arange(1, 8)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i], covariance_type = 'spherical').fit(X)
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]
    fig = plt.figure(figsize=(18, 6))

    # plot 1: data + best-fit mixture
    ax = fig.add_subplot(131)
    M_best = models[np.argmin(BIC)]

    x = np.linspace(-6, 6, 1000)
    logprob = M_best.score_samples(x.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    ax.hist(X, 30, density=True, histtype='stepfilled', alpha=0.4)
    ax.plot(x, pdf, '-k')
    ax.plot(x, pdf_individual, '--k')
    ax.text(0.04, 0.96, "Best-fit Mixture",
            ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p(x)$')
    print("best number of components", M_best.n_components)
    
    # plot 2: AIC and BIC
    ax = fig.add_subplot(132)
    ax.plot(N, AIC, '-k', label='AIC')
    ax.plot(N, BIC, '--k', label='BIC')
    ax.set_xlabel('n. components')
    ax.set_ylabel('information criterion')
    ax.legend(loc=2)

    # plot 3: posterior probabilities for each component
    ax = fig.add_subplot(133)

    p = M_best.predict_proba(x.reshape(-1, 1))
    #p = p[:, (1, 0, 2)]  # rearrange order so the plot looks better
    p = p.cumsum(1).T

    ax.fill_between(x, 0, p[0], color='gray', alpha=0.3)
    ax.fill_between(x, p[0], p[1], color='gray', alpha=0.5)
    ax.fill_between(x, p[1], 1, color='gray', alpha=0.7)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$p({\rm class}|x)$')


    plt.show()
    # print(M_best.weights_) #perfect
    # print(M_best.means_) #perfect
    # print(M_best.covariances_) #bad
    # print((M_best.means_+np.mean(dfy)) * np.std(dfy))
    return np.argmax(M_best.predict_proba(np.array(y).reshape(-1,1)), axis = 1)