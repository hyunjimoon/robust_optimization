import pandas as pd
import numpy as np
from scipy.stats import norm
import logging
import cmdstanpy
import jax
from jax.experimental import optimizers
import jax.scipy as jsc
import jax.numpy as jnp
from loss import lopt_NV
from generator import sim_y_barThetax, sim_ThetaXy, P_ybarx, check_random_state, make_low_rank_matrix
# approach 1 if y = None: W(Theta, X) else Thetahat, W(Thetahat, x)
def argmin_lopt_bar_argmin_lpred(link, family, X, Theta, y, input_type, alg_type = "sample"):
    '''
    Compute approach1 optimal solution; separate predict optimize
    Parameters:
         char MClass: model class type specificed with `P_ybarx`
            "lin": y_true = theta' * X
            "quad": y_true = theta' * X^2
          real Theta: coefficient vector of size [p, 1] 
            "fixed"
            real value of Theta (computing trainset error)
          np.array X: predictor data of size [n, p] 
          chr alg_type: Solver algorithm
    Returns:
        np.array what: predict then optimize optimal solution of size [n,1]
    '''
    # if not any([y, Theta]):
    #   print("Both Theta and y cannot be None")
    what = np.repeat(np.nan, len(X))
    if input_type == "Theta":
        for i in range(len(X)):
          what[i] = argmin_lopt_gen(link, family, X[i], Theta)
        return what
    else:
        Thetahat1 = argmin_lpred(link, family, X, y, alg_type = alg_type)
        for i in range(len(X)):
          what[i] = argmin_lopt_gen(link, family, X[i], Thetahat1)
        return what, Thetahat1

# approach 2
def argmin_lopt_emp(link, family, X, Theta, y, input_type, Thetastar = None, search_type = "autodiff"):
    '''
    Compute approach2 optimal solution; empirical objective minimization
    Parameters:
        char link: model class type specificed with
            "lin": y_true = Theta' * X
            "quad": y_true = x.transpose() @ Theta_mat @ x + Theta @ x
        char family: distribution type
        array y: outcome data of size [n, 1]
        array X: predictor data of size [n, p]
        char search_type: `grid`, `autodiff`
    Returns:
        array what: optimal solution for each x of size [n,1]
    '''
    # if not any([y, Theta]):
    #   print("Both Theta and y cannot be None")
    what = np.repeat(np.nan, len(X))
    if input_type == "Theta":
        for i in range(len(X)):
          what[i] = argmin_lopt_gen(link, family, X[i], Theta)
        return what
    else:
        if search_type == "grid":
            def lopt_sim(link, family, X, Theta):
                '''
                Loss when function is known upto P_Y|X^theta, but not w
                '''
                lopt = 0
                for i in range(len(X)):
                  what_x = argmin_lopt_gen(link, family, X[i], Theta)
                  lopt -= (5 * min(what_x, y[i]) - 1 * what_x)
                return lopt
            theta1 = np.linspace(Thetastar[0]*.8, Thetastar[0]*1.2, num = 10)
            theta2 = np.linspace(Thetastar[1]*.8, Thetastar[1]*1.2, num = 10)
            theta1, theta2 = np.meshgrid(theta1, theta2)
            M = 1000000
            for Theta in zip(theta1.ravel(), theta2.ravel()):
                #lopt_NV(what, y[i], profit, cost) 'numpy.float64' object cannot be interpreted as an integer
              if lopt_sim(link, family, X, Theta) < M:
                Thetahat2 = Theta
        elif search_type == "autodiff":
            def lopt_fn(param, X, y): #param = p*1, X:n*p, x= 1*p
                '''
                Loss when function is known upto w(x, P_Y|X^theta)
                '''
                lopt = 0
                for i in range(len(X)):
                  what_x = jsc.stats.norm.ppf(loc = param @ X[i], q =0.8) #TODO
                  lopt -= 5 * min(what_x, y[i]) - 1 * what_x
                return lopt
            Theta = Thetastar * 1.1 # np.random.normal(1) # init
            lr = 1
            opt_init, opt_update, get_params = optimizers.adagrad(lr)
            opt_state = opt_init(Theta)
            for i in range(50): # TODO cvg diag.
                loss, grads = jax.value_and_grad(lopt_fn)(get_params(opt_state), X, y)
                opt_state = opt_update(i, grads, opt_state)
                print("loss", loss, "Thetastar", Thetastar, "Thetahat", get_params(opt_state))
            Thetahat2 = get_params(opt_state)
            for i in range(len(X)):
                what[i] = argmin_lopt_gen(link, family, X[i], Thetahat2)
            return what, Thetahat2

# plug-in opt.solver
def argmin_lopt_gen(link, family, x, Theta, closed_type = "W_Theta", **kwargs):
    '''
    Compute optimal solution for each Theta and X assuming $Y|X \sim P_{\Theta}$
    Parameters:
        char link: "log", "linear", "quadratic" (, "logit") relation between eta and mu
        char family: "Normal", "Exp", "Cauchy"
        array Theta: model parameter of size [p, 1]
        array x: predictor data of size [1, p]
        function lopt: loss function, given w(x), y, X, outputs loss
    Returns:
        array w: optimal solution for each x of size [n,1]
    '''
    if closed_type == "P_ybarx":      
      y_sim = P_ybarx(link, family, x, Theta, sigma_y)
      data = {'y': list(y_sim), 'n': len(y_sim)}
      data = {**data, **kwargs}
      sm = cmdstanpy.CmdStanModel(stan_file="/content/drive/MyDrive/Colab Notebooks/robust_optimization/src/stan/optW_lopt_NV.stan")
      W_Thetax = sm.optimize(data).stan_variable('w')
    elif closed_type == "W_Theta":
      print(Theta)
      print(x.shape)
      W_Thetax = np.dot(x, Theta) + norm.ppf(.8) * sigma_y
    return W_Thetax
def argmin_lpred(link, family, X, y, alg_type = "sample"):
    '''
    Solve parameter that best predicts y given X assuming $y\sim N(X * Theta, 1)$
    Parameters:
        char link: model class type specificed with `P_ybarx` which is user-defined
             input for `argmin_lopt_gen`, `argmin_lopt_emp`
             SHOULD comply with "P_ybarx_f{link}.stan"
             "lin": y_true = Theta' * X
             "quad": y_true = Theta' * X^2
        char family: distribution type specificed e.g. "Normal", "Cauchy"
        np.array X: predictor data of size [n, p]
        np.array y: outcome data of size [n, 1]
        chr alg_type: Solver algorithm
             "sample": MAP with improper uniform priors (undeclared) in stan file with HMC
             "optimize": Maximum likelihood estimate () with "lbfgs", "bfgs", "newton"
             "varaiational": variational inference with ADVI, RVI
             https://mc-stan.org/cmdstanpy/examples/Maximum%20Likelihood%20Estimation.html
             https://mc-stan.org/cmdstanpy/examples/Variational%20Inference.html
    Returns:
        np.array Thetahat: optimal parameter value of size [p, 1]
    '''
    data = {'X':X.tolist(), 'y':y, 'n': len(X), 'p': len(X[1]), 'sigma_y': 1}
    sm = cmdstanpy.CmdStanModel(stan_file="/content/drive/MyDrive/Colab Notebooks/robust_optimization/src/stan/optTheta_lpred.stan") # true Theta: 2, 2
    if alg_type == "sample":
      Thetahat = np.mean(sm.sample(data).stan_variable('beta'), axis =0) # 2.00, 1.98
    if alg_type == "optimize":
      Thetahat = sm.optimize(data).stan_variable('beta') # 2.00 , 1.98
    if alg_type == "variational":
      Thetahat = np.mean(sm.sample(data).stan_variable('beta'), axis =0) # 1.99, 1.99
    return Thetahat
