import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# 모든 분포는 동일한 1) mean (가능하다면 variance), 2) sample 수를 가진다.
np.random.seed(1)
# extended sklearn `make_regression`
# plug-in optimizer wo closed form use this to find empirically find argmin_w with simulated data (x,y)
def sim_y_barThetax(mu2eta, family, Theta, x, sigma_y, n_gen=100, bias=0, random_state = None):
    '''
    Generate data on assumed Theta and x (used in approach2)
    Parameters:
        array Theta_star: true parameter value
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
    p = len(Theta)
    return np.random.normal(loc = np.dot(x, Theta), scale = sigma_y, size = n_gen)
# Trainset, Testset
def sim_ThetaXy(mu2eta = "lin", family = "Normal", n_gen=100, n_features=2, n_informative=10,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, sigma_y=0.0, random_state = None):
    """
    Generate data of 
    Theta
    X
    y: 1.ground truth (mu2eta= "lin", norm_dist), 2.(mu2eta= "exp", norm_dist), 
                     3.(mu2eta= "quad" , norm_dist), 4.(mu2eta= "lin" , unif_dist)
    Parameters:
        array Theta_star: true parameter value
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
        Theta : array of shape [n_features] or [n_features, n_targets], optional
        The coefficient of the underlying linear model. It is returned only if
        coef is True.
        
        X : array of shape [n_gen, n_features]
        The input samples.

        y : array of shape [n_gen] or [n_gen, n_targets]
            The output values.

    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile. See :func:`make_low_rank_matrix` for
    more details.

    Parameters
    ----------
    n_gen : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=100)
        The number of features.

    n_informative : int, optional (default=10)
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.

    n_targets : int, optional (default=1)
        The number of regression targets, i.e., the dimension of the y output
        vector associated with a sample. By default, the output is a scalar.

    bias : float, optional (default=0.0)
        The bias term in the underlying linear model.

    effective_rank : int or None, optional (default=None)
        if not None:
            The approximate number of singular vectors required to explain most
            of the input data by linear combinations. Using this kind of
            singular spectrum in the input allows the generator to reproduce
            the correlations often observed in practice.
        if None:
            The input set is well conditioned, centered and gaussian with
            unit variance.

    tail_strength : float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values
        profile if `effective_rank` is not None.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    coef : boolean, optional (default=False)
        If True, the coefficients of the underlying linear model are returned.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    """
    # Generate a ground truth model with only n_informative features being non
    # zeros (the other features are not correlated to y and should be ignored
    # by a sparsifying regularizers such as L1 or elastic net)
    n_informative = min(n_features, n_informative)
    generator = check_random_state(random_state)
    #Theta
    Theta = np.zeros((n_features, n_targets))
    Theta[:n_informative, :] =  np.array([[5  ],
                                       [6]])
    #abs(generator.rand(n_informative, n_targets))#100 *
    if effective_rank is None:
        # Randomly generate a well conditioned input set
        X = generator.randn(n_gen, n_features)  + 10
    else:
        # Randomly generate a low rank, fat tail input set
        X = make_low_rank_matrix(n_gen=n_gen,
                                 n_features=n_features,
                                 effective_rank=effective_rank,
                                 tail_strength=tail_strength,
                                 random_state=generator)
    
    ys = [P_ybarx(mu2eta, "Normal", Theta, X, sigma_y) for mu2eta in ("lin", "log")]  # for family in ("Normal", "Unif")
    return X, ys, np.squeeze(Theta)
    
def P_ybarx(mu2eta, family, Theta, X, sigma_y):
    '''
    Compute approach1 optimal solution; separate predict optimize
    Parameters:
         char mu2eta: model class type specificed with `P_ybarx`
            "lin": y_true = theta' * X
            "quad": y_true = theta' * X^2
          real family: distribution family
          real Theta: shape (p,) parameter vector
          real X: shape [n,p] predictor vector
            "fixed"
            real value of Theta (computing trainset error)
          np.array X: predictor data of size [n, p] 
          chr alg_type: Solver algorithm
    Returns:
        np.array what: predict then optimize optimal solution of size [n,1]
    '''
    mu = np.squeeze(np.dot(X, Theta))
    if mu2eta == "lin":
        eta = mu 
    elif mu2eta == "log":
        eta = np.exp(mu)
    elif mu2eta == "quadratic":
        eta =  np.dot(np.power(X, 2), Theta) 
        #= X.transpose() * diag(Theta) how to produce Theta2, Theta1 separately and fit
    if family == "Normal":
        return np.random.normal(eta, scale = sigma_y, size = len(X))
    elif family == "Exp":
        return np.random.exp(loc = eta, scale = sigma_y, size = len(X))
    elif family == "Cauchy":
        return np.random.Cauchy(loc = eta, scale = sigma_y, size = len(X))

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def make_low_rank_matrix(n_samples=100, n_features=100, *, effective_rank=10,
                         tail_strength=0.5, random_state=None):
    """Generate a mostly low rank matrix with bell-shaped singular values.
    Most of the variance can be explained by a bell-shaped curve of width
    effective_rank: the low rank part of the singular values profile is::
        (1 - tail_strength) * exp(-1.0 * (i / effective_rank) ** 2)
    The remaining singular values' tail is fat, decreasing as::
        tail_strength * exp(-0.1 * i / effective_rank).
    The low rank part of the profile can be considered the structured
    signal part of the data while the tail can be considered the noisy
    part of the data that cannot be summarized by a low number of linear
    components (singular vectors).
    This kind of singular profiles is often seen in practice, for instance:
     - gray level pictures of faces
     - TF-IDF vectors of text documents crawled from the web
    Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=100
        The number of features.
    effective_rank : int, default=10
        The approximate number of singular vectors required to explain most of
        the data by linear combinations.
    tail_strength : float, default=0.5
        The relative importance of the fat noisy tail of the singular values
        profile. The value should be between 0 and 1.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The matrix.
    """
    generator = check_random_state(random_state)
    n = min(n_samples, n_features)

    # Random (ortho normal) vectors
    u, _ = linalg.qr(generator.randn(n_samples, n), mode='economic',
                     check_finite=False)
    v, _ = linalg.qr(generator.randn(n_features, n), mode='economic',
                     check_finite=False)

    # Index of the singular values
    singular_ind = np.arange(n, dtype=np.float64)

    # Build the singular profile by assembling signal and noise components
    low_rank = ((1 - tail_strength) *
                np.exp(-1.0 * (singular_ind / effective_rank) ** 2))
    tail = tail_strength * np.exp(-0.1 * singular_ind / effective_rank)
    s = np.identity(n) * (low_rank + tail)

    return np.dot(np.dot(u, s), v.T)

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